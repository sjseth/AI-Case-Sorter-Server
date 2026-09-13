"""ConvNeXt training worker (server port of the desktop client's trainer).

Kept behaviourally identical to ``sorter/training/train_convnext.py`` in the
AI-Case-Sorter-Py client -- same dataset convention, augmentations, head
layout, optimizer, SWA logic, and checkpoint payload -- so a model trained
here is indistinguishable from one trained in the desktop app. The
differences are all operational:

* ``--device auto|cpu|cuda`` (the client always picks CUDA when present).
* SIGTERM / SIGINT stop the run cleanly at the next batch and exit 130 with a
  ``cancelled`` marker, instead of dying mid-write.
* Pretrained ImageNet weights fall back to random init (with a log line) when
  the torchvision download fails, so an air-gapped server can still train.
* Checkpoints are written atomically (tmp + replace).
* Extra ``batch`` progress markers give the UI intra-epoch feedback.
* ``Infinity`` never reaches the JSON markers (it isn't valid JSON).

Progress markers on stdout, one per line:

    [PROGRESS] {"event": "start", "epochs": 15, "classes": 7, "images": 320, ...}
    [PROGRESS] {"event": "batch", "epoch": 1, "batch": 12, "batches": 40, ...}
    [PROGRESS] {"event": "epoch", "epoch": 1, "train_loss": 0.42, ...}
    [PROGRESS] {"event": "done", "best_val_acc": 0.94, "env": {...}}
    [PROGRESS] {"event": "cancelled"}
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import random
import signal
import sys
import time
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    from torch.optim.swa_utils import SWALR, AveragedModel, update_bn
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import models, transforms
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(f"[train_convnext] PyTorch is required ({exc}). Run setup.py first.\n")
    raise


SUPPORTED_MODELS = ("convnext_tiny", "convnext_small", "convnext_base", "convnext_large")

_cancel_requested = False


class Cancelled(Exception):
    pass


def _request_cancel(signum, _frame) -> None:  # pragma: no cover - signal path
    global _cancel_requested
    _cancel_requested = True
    _say(f"[INFO] Cancel requested (signal {signum}); stopping after the current batch.")


def _check_cancel() -> None:
    if _cancel_requested:
        raise Cancelled()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _emit(event: str, **payload: Any) -> None:
    payload["event"] = event
    sys.stdout.write("[PROGRESS] " + json.dumps(_json_safe(payload), separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _say(line: str = "") -> None:
    print(line, flush=True)


def _describe_device(device: Any) -> list[str]:
    import numpy
    import torchvision

    lines = [
        f"[SETUP] PyTorch {torch.__version__} (torchvision {torchvision.__version__}, numpy {numpy.__version__})",
        f"[SETUP] CUDA available: {torch.cuda.is_available()}",
    ]
    if device.type == "cuda":
        index = device.index or 0
        major, minor = torch.cuda.get_device_capability(index)
        total_gb = torch.cuda.get_device_properties(index).total_memory / (1024**3)
        lines += [
            f"[SETUP] CUDA runtime: {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}",
            "[INFO] Device: CUDA",
            f"[INFO] Detected GPU: {torch.cuda.get_device_name(index)} ({total_gb:.1f} GB, compute sm_{major}{minor})",
        ]
    else:
        lines += [
            "[INFO] Device: CPU",
            "[INFO] No CUDA device in use -- training on the CPU is much slower.",
        ]
    return lines


def _describe_config(args: argparse.Namespace) -> list[str]:
    settings = [
        ("Model", args.model_name),
        ("Batch Size", args.batch_size),
        ("Initial LR", args.lr),
        ("Weight Decay", args.weight_decay),
        ("Dropout", args.dropout),
        ("Validation Split", args.val_split),
        ("Full Dataset Training", bool(args.trainall)),
        ("Image Size", f"{args.imgsize}x{args.imgsize}"),
        ("Freeze Backbone", bool(args.freeze_backbone)),
        ("Focal Loss", f"gamma {args.focal_gamma}" if args.use_focal_loss else False),
        (
            "Stochastic Depth",
            args.stochastic_depth_prob if args.stochastic_depth_prob >= 0 else "torchvision default",
        ),
        ("SWA", f"from {args.swa_start:.0%} ({args.swa_mode})" if args.use_swa else False),
        ("Target Epochs", args.epochs),
    ]
    return ["[INFO] Configuration:"] + [f"       {name}: {value}" for name, value in settings]


def checkpoint_env() -> dict[str, str]:
    import numpy
    import torchvision

    return {
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "numpy_version": str(numpy.__version__),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ConvNeXt trainer")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_model", required=True)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--trainall", action="store_true")

    p.add_argument("--model_name", default="convnext_tiny", choices=SUPPORTED_MODELS)
    p.add_argument("--imgsize", type=int, default=232)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--max_workers", type=int, default=-1)

    p.add_argument("--stochastic_depth_prob", type=float, default=-1.0)
    p.add_argument("--use_focal_loss", action="store_true")
    p.add_argument("--focal_gamma", type=float, default=1.0)
    p.add_argument("--use_swa", action="store_true")
    p.add_argument("--swa_start", type=float, default=0.75)
    p.add_argument("--swa_mode", default="scheduled", choices=("scheduled", "adaptive"))
    p.add_argument("--swa_acc_threshold", type=float, default=0.96)
    p.add_argument("--swa_patience", type=int, default=5)
    p.add_argument("--swa_min_epoch", type=int, default=10)

    # Server additions
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--no_pretrained", action="store_true", help="Skip the ImageNet weight download")
    p.add_argument("--batch_progress_every", type=int, default=5)

    return p.parse_args(argv)


# ----- Dataset (module-level so DataLoader workers can pickle) --------------


class FilenameLabelDataset(Dataset):
    """Reads images from a flat directory; label = filename prefix before '__'."""

    def __init__(self, image_dir: str) -> None:
        self.image_dir = image_dir
        self.image_paths: list[str] = []
        labels_str: list[str] = []
        for fname in sorted(os.listdir(image_dir)):
            if "__" not in fname:
                continue
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            self.image_paths.append(os.path.join(image_dir, fname))
            labels_str.append(fname.split("__", 1)[0])
        if not self.image_paths:
            raise RuntimeError(f"No valid images found in {image_dir}")
        self.classes = sorted(set(labels_str))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = [self.class_to_idx[lbl] for lbl in labels_str]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return img, self.labels[idx]


class TransformSubset(Dataset):
    def __init__(self, subset, transform) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        img, lbl = self.subset[idx]
        return self.transform(img), lbl


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.0, alpha: float = 1.0, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none", label_smoothing=self.label_smoothing)
        log_pt = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(log_pt.gather(1, targets.unsqueeze(1)).squeeze(1))
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


def _get_transforms(image_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def _get_model_weights(model_name: str):
    weights_map = {
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "convnext_small": models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        "convnext_base": models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
        "convnext_large": models.ConvNeXt_Large_Weights.IMAGENET1K_V1,
    }
    return getattr(models, model_name), weights_map[model_name]


def _build_model(args: argparse.Namespace):
    model_ctor, model_weights = _get_model_weights(args.model_name)
    kwargs: dict[str, Any] = {}
    if args.stochastic_depth_prob >= 0:
        kwargs["stochastic_depth_prob"] = args.stochastic_depth_prob
    if args.no_pretrained or os.environ.get("CASESORTER_NO_PRETRAINED", "").strip().lower() in ("1", "true", "yes"):
        _say("[INFO] Pretrained weights disabled: initialising the backbone randomly")
        return model_ctor(weights=None, **kwargs)
    try:
        return model_ctor(weights=model_weights, **kwargs)
    except Exception as exc:  # download failure on an offline box
        _say(f"[WARN] Could not load pretrained ImageNet weights ({exc}).")
        _say("[WARN] Falling back to random initialisation; accuracy will be lower.")
        return model_ctor(weights=None, **kwargs)


def _atomic_save(payload: dict[str, Any], path: str) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _pick_device(choice: str) -> Any:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            _say("[WARN] --device cuda requested but CUDA is unavailable; using CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_epoch(
    model, loader, criterion, optimizer, scaler, device, is_train: bool, use_channels_last: bool,
    *, epoch: int, total_epochs: int, progress_every: int,
) -> tuple[float, float]:
    if is_train:
        model.train()
    else:
        model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    n_batches = len(loader)
    started = time.time()
    with torch.set_grad_enabled(is_train):
        for b_idx, (imgs, labels) in enumerate(loader, 1):
            _check_cancel()
            imgs = imgs.to(device)
            labels = labels.to(device)
            if use_channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            if progress_every > 0 and (b_idx % progress_every == 0 or b_idx == n_batches):
                elapsed = time.time() - started
                _emit(
                    "batch",
                    epoch=epoch,
                    total=total_epochs,
                    phase="train" if is_train else "val",
                    batch=b_idx,
                    batches=n_batches,
                    loss=float(running_loss / max(total, 1)),
                    acc=float(correct / max(total, 1)),
                    elapsed=round(elapsed, 1),
                )
    return running_loss / max(total, 1), correct / max(total, 1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, _request_cancel)
    signal.signal(signal.SIGINT, _request_cancel)
    if hasattr(signal, "SIGBREAK"):  # Windows: CTRL_BREAK_EVENT
        signal.signal(signal.SIGBREAK, _request_cancel)  # type: ignore[attr-defined]

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(42)
    torch.manual_seed(42)

    device = _pick_device(args.device)
    for line in _describe_device(device):
        _say(line)
    _say()
    for line in _describe_config(args):
        _say(line)
    _say()

    dataset = FilenameLabelDataset(args.image_dir)
    _say(f"[INFO] Loaded {len(dataset)} images. Classes: {len(dataset.classes)}")
    _emit(
        "start",
        epochs=args.epochs,
        classes=len(dataset.classes),
        class_names=dataset.classes,
        images=len(dataset),
        device=device.type,
        model=args.model_name,
    )

    val_split = float(args.val_split or 0.0)
    if not (0.0 <= val_split < 1.0):
        raise ValueError("--val_split must be in [0.0, 1.0).")
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    if n_val == 0 or args.trainall:
        if args.trainall:
            _say("[INFO] --trainall specified -> using the full dataset for training")
        else:
            _say("[INFO] Validation split rounds to zero images -> training on everything")
        train_sub = dataset
        val_sub = None
    else:
        _say(f"[INFO] Split: {n_train} training / {n_val} validation images")
        train_sub, val_sub = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )

    train_tf, val_tf = _get_transforms(args.imgsize)

    num_workers = args.max_workers
    if num_workers < 0:
        cpu = os.cpu_count() or 1
        if sys.platform == "win32":
            num_workers = min(2, max(0, cpu - 1))
        else:
            num_workers = min(8, max(1, cpu - 1))
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    train_loader = DataLoader(TransformSubset(train_sub, train_tf), shuffle=True, **loader_args)
    val_loader = None
    if val_sub is not None:
        val_loader = DataLoader(TransformSubset(val_sub, val_tf), shuffle=False, **loader_args)

    model = _build_model(args)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=args.dropout),
        nn.Linear(in_features, len(dataset.classes)),
    )
    model.to(device)

    use_channels_last = False
    if device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 7:
        model = model.to(memory_format=torch.channels_last)
        use_channels_last = True
    _say(f"[INFO] Data loader workers: {num_workers}")
    _say(f"[INFO] channels_last memory format: {'enabled' if use_channels_last else 'disabled'}")

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.use_focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma, label_smoothing=0.1)
        _say(f"[INFO] Using FocalLoss (gamma {args.focal_gamma}) with label smoothing 0.1")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        _say("[INFO] Using CrossEntropyLoss with label smoothing 0.1")

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    swa_model = None
    swa_scheduler = None
    swa_active = False
    swa_fallback_epoch = int(args.epochs * args.swa_start)
    if args.use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.1)

    best_acc = -1.0
    best_loss = float("inf")
    best_plateau_loss = float("inf")
    epochs_no_improve = 0
    run_started = time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.time()
            train_loss, train_acc = _run_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                is_train=True, use_channels_last=use_channels_last,
                epoch=epoch, total_epochs=args.epochs, progress_every=args.batch_progress_every,
            )
            val_loss: float | None = None
            val_acc: float | None = None
            if val_loader is not None:
                val_loss, val_acc = _run_epoch(
                    model, val_loader, criterion, None, None, device,
                    is_train=False, use_channels_last=use_channels_last,
                    epoch=epoch, total_epochs=args.epochs, progress_every=args.batch_progress_every,
                )
                if val_loss < best_plateau_loss:
                    best_plateau_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            if args.use_swa and not swa_active:
                if args.swa_mode == "scheduled":
                    should_start = epoch >= swa_fallback_epoch
                else:
                    hit_threshold = val_acc is not None and val_acc >= args.swa_acc_threshold
                    plateaued = epochs_no_improve >= args.swa_patience and epoch >= args.swa_min_epoch
                    fallback = epoch >= swa_fallback_epoch
                    should_start = hit_threshold or plateaued or fallback
                if should_start:
                    swa_active = True

            if args.use_swa and swa_active:
                assert swa_model is not None and swa_scheduler is not None
                swa_model.update_parameters(model)
                swa_scheduler.step()
                current_lr = swa_scheduler.get_last_lr()[0]
            else:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

            save_payload = {
                "model_state_dict": model.state_dict(),
                "classes": dataset.classes,
                "base": args.model_name,
                "image_size": int(args.imgsize),
                **checkpoint_env(),
            }
            if val_acc is None or val_loss is None:
                _atomic_save(save_payload, args.output_model)
                saved_reason = "no-validation"
            else:
                save_payload["val_acc"] = val_acc
                save_payload["val_loss"] = val_loss
                if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                    best_acc = val_acc
                    best_loss = val_loss
                    _atomic_save(save_payload, args.output_model)
                    saved_reason = "new-best"
                else:
                    saved_reason = "skipped"

            _emit(
                "epoch",
                epoch=epoch,
                total=args.epochs,
                train_loss=float(train_loss),
                train_acc=float(train_acc),
                val_loss=(float(val_loss) if val_loss is not None else None),
                val_acc=(float(val_acc) if val_acc is not None else None),
                lr=float(current_lr),
                swa_active=bool(swa_active),
                saved=saved_reason,
                epoch_seconds=round(time.time() - epoch_started, 1),
                elapsed_seconds=round(time.time() - run_started, 1),
            )
    except Cancelled:
        _say("[INFO] Training cancelled.")
        _emit("cancelled", epochs_completed=max(0, epoch - 1) if "epoch" in locals() else 0)
        return 130

    if args.use_swa and swa_active:
        assert swa_model is not None
        update_bn(train_loader, swa_model, device=device)
        swa_path = args.output_model.replace(".pth", "_swa.pth")
        _atomic_save(
            {
                "model_state_dict": swa_model.state_dict(),
                "classes": dataset.classes,
                "base": args.model_name,
                "image_size": int(args.imgsize),
                **checkpoint_env(),
            },
            swa_path,
        )
        _say(f"[INFO] SWA checkpoint written: {swa_path}")

    _emit(
        "done",
        best_val_acc=best_acc if best_acc >= 0 else None,
        best_val_loss=best_loss if best_loss != float("inf") else None,
        classes=dataset.classes,
        images=len(dataset),
        duration_seconds=round(time.time() - run_started, 1),
        env=checkpoint_env(),
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
