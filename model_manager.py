"""Loads trained ConvNeXt checkpoints and runs inference against them.

A PyTorch-only port of the inference path used by the CaseSorter desktop
client (``sorter/ml/local_inference.py``). Checkpoints produced by the
desktop trainer, this server's trainer, the legacy Windows app, and their
SWA variants all load here.

Models are registered under an *alias* -- the name the OpenAI ``model`` field
uses. Static aliases come from ``config.MODELS``; the registry (models the
server trained, imported or downloaded) registers and unregisters aliases at
runtime when serving is toggled.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            probe = torch.randn(1, 3, 8, 8, device="cuda")
            _ = probe.sum().item()
            log.info("Using CUDA (%s)", torch.cuda.get_device_name(0))
            return torch.device("cuda")
        except Exception as exc:
            log.warning("CUDA reported available but test op failed (%s); falling back to CPU", exc)
            return torch.device("cpu")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        try:
            probe = torch.randn(1, 3, 8, 8, device="mps")
            _ = probe.sum().item()
            log.info("Using MPS (Apple GPU)")
            return torch.device("mps")
        except Exception as exc:
            log.warning("MPS probe failed (%s); falling back to CPU", exc)
    log.info("CUDA unavailable; using CPU")
    return torch.device("cpu")


DEVICE: torch.device = _select_device()


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": DEVICE.type,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "gpu_memory_gb": None,
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:  # pragma: no cover
        info["torchvision"] = None
    if DEVICE.type == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 1)
            cap = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"{cap[0]}.{cap[1]}"
        except Exception:  # pragma: no cover
            pass
    return info


# ---------------------------------------------------------------------------
# ConvNeXt variants
# ---------------------------------------------------------------------------

_CONVNEXT_VARIANTS: Dict[str, Tuple[Callable[..., nn.Module], int]] = {
    "convnext_tiny": (models.convnext_tiny, 224),
    "convnext_small": (models.convnext_small, 224),
    "convnext_base": (models.convnext_base, 224),
    "convnext_large": (models.convnext_large, 224),
}

_BASE_ALIASES = {
    "tiny": "convnext_tiny",
    "small": "convnext_small",
    "base": "convnext_base",
    "large": "convnext_large",
}


def _resolve_base(name: str) -> str:
    key = (name or "").lower().replace("-", "_")
    return _BASE_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _build_model(base: str, num_classes: int, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    """Construct a ConvNeXt with the classifier shape implied by ``state_dict``."""
    base = _resolve_base(base)
    if base not in _CONVNEXT_VARIANTS:
        raise ValueError(f"Unsupported model base: {base!r}")

    factory, _ = _CONVNEXT_VARIANTS[base]
    model = factory(weights=None)
    in_features = model.classifier[2].in_features

    if "classifier.3.weight" in state_dict:
        model.classifier = nn.Sequential(
            model.classifier[0],
            model.classifier[1],
            nn.Dropout(p=0.0),
            nn.Linear(in_features, num_classes),
        )
    elif "classifier.2.1.weight" in state_dict:
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def _torch_load(path: Path, trusted: bool) -> Any:
    """``weights_only=True`` first (safe for untrusted files); trusted files may fall back."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not trusted:
            raise RuntimeError(
                f"{path.name}: refusing to load with weights_only=False (untrusted checkpoint). {exc}"
            ) from exc
        log.warning("weights_only load failed for %s (%s); retrying as trusted file", path.name, exc)
        return torch.load(path, map_location="cpu", weights_only=False)


def read_checkpoint_meta(path: Path | str) -> dict[str, Any]:
    """Classes / base / image size / env recorded in a checkpoint, without building the net."""
    ckpt = _torch_load(Path(path), trusted=False)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint is not a dict payload")
    return {
        "classes": list(ckpt.get("classes") or []),
        "base": _resolve_base(ckpt.get("base") or "convnext_tiny"),
        "image_size": int(ckpt.get("image_size") or 0) or None,
        "val_acc": ckpt.get("val_acc"),
        "val_loss": ckpt.get("val_loss"),
        "torch_version": ckpt.get("torch_version"),
        "torchvision_version": ckpt.get("torchvision_version"),
        "numpy_version": ckpt.get("numpy_version"),
    }


def _load_checkpoint(path: Path, trusted: bool) -> Tuple[nn.Module, List[str], str, Optional[int], dict]:
    log.info("Loading checkpoint: %s", path)
    ckpt = _torch_load(path, trusted)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: checkpoint is not a dict payload")

    base = ckpt.get("base", "convnext_base")
    classes = list(ckpt.get("classes") or [])
    if not classes:
        raise ValueError(f"{path}: no 'classes' in checkpoint")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    state_dict.pop("n_averaged", None)

    model = _build_model(base, num_classes=len(classes), state_dict=state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()

    ckpt_size = int(ckpt.get("image_size") or 0) or None
    meta = {
        "val_acc": ckpt.get("val_acc"),
        "val_loss": ckpt.get("val_loss"),
        "torch_version": ckpt.get("torch_version"),
    }
    log.info("Loaded %s (%d classes) on %s", _resolve_base(base), len(classes), DEVICE)
    return model, classes, _resolve_base(base), ckpt_size, meta


# ---------------------------------------------------------------------------
# Public manager
# ---------------------------------------------------------------------------

class ModelManager:
    """Caches loaded checkpoints keyed by alias (and by path for ad-hoc use)."""

    def __init__(
        self,
        aliases: Dict[str, str],
        options: Optional[Dict[str, dict]] = None,
        base_dir: Optional[Path] = None,
    ):
        self._base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
        self._aliases: Dict[str, str] = {}
        self._options: Dict[str, dict] = {}
        self._trusted: Dict[str, bool] = {}
        self._sources: Dict[str, str] = {}
        self._cache: Dict[str, dict] = {}
        self._path_cache: Dict[Tuple[str, int], dict] = {}
        self._lock = threading.RLock()
        self._infer_lock = threading.Lock()
        for alias, raw in dict(aliases).items():
            self.register(alias, raw, (options or {}).get(alias), trusted=True, source="config")

    # -- registry -----------------------------------------------------------

    def register(
        self,
        alias: str,
        path: str | Path,
        options: Optional[dict] = None,
        *,
        trusted: bool = False,
        source: str = "registry",
    ) -> None:
        alias = alias.strip()
        with self._lock:
            self._aliases[alias] = str(path)
            self._options[alias] = dict(options or {})
            self._trusted[alias] = bool(trusted)
            self._sources[alias] = source
            self._cache.pop(alias, None)

    def unregister(self, alias: str) -> None:
        with self._lock:
            self._aliases.pop(alias, None)
            self._options.pop(alias, None)
            self._trusted.pop(alias, None)
            self._sources.pop(alias, None)
            self._cache.pop(alias, None)

    def invalidate(self, alias: Optional[str] = None) -> None:
        with self._lock:
            if alias is None:
                self._cache.clear()
                self._path_cache.clear()
            else:
                self._cache.pop(alias, None)

    def aliases(self) -> List[str]:
        with self._lock:
            return list(self._aliases)

    def has(self, alias: str) -> bool:
        return alias in self._aliases

    def source(self, alias: str) -> Optional[str]:
        return self._sources.get(alias)

    def describe(self) -> List[dict]:
        with self._lock:
            out = []
            for alias in self._aliases:
                entry = self._cache.get(alias)
                out.append(
                    {
                        "alias": alias,
                        "path": self._aliases[alias],
                        "source": self._sources.get(alias),
                        "loaded": entry is not None,
                        "classes": len(entry["classes"]) if entry else None,
                        "base": entry["base"] if entry else None,
                        "image_size": entry["image_size"] if entry else self._options.get(alias, {}).get("image_size"),
                    }
                )
            return out

    def preload(self) -> None:
        for alias in self.aliases():
            try:
                self._load(alias)
            except Exception as exc:  # noqa: BLE001
                log.error("Preload of %r failed: %s", alias, exc)

    def classes(self, alias: str) -> List[str]:
        return list(self._load(alias)["classes"])

    # -- loading ------------------------------------------------------------

    def _resolve_path(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = self._base_dir / path
        return path

    @staticmethod
    def _make_entry(model: nn.Module, classes: List[str], base: str, size: int, meta: dict) -> dict:
        return {
            "model": model,
            "classes": classes,
            "base": base,
            "image_size": size,
            "meta": meta,
            "transform": transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        }

    def _load(self, alias: str) -> dict:
        with self._lock:
            if alias in self._cache:
                return self._cache[alias]
            if alias not in self._aliases:
                raise KeyError(f"Unknown model alias: {alias!r}")

            path = self._resolve_path(self._aliases[alias])
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            model, classes, base, ckpt_size, meta = _load_checkpoint(path, self._trusted.get(alias, False))
            override = self._options.get(alias, {}).get("image_size")
            if override:
                size = int(override)
            elif ckpt_size:
                size = ckpt_size
            else:
                size = _CONVNEXT_VARIANTS[base][1]
            entry = self._make_entry(model, classes, base, size, meta)
            self._cache[alias] = entry
            return entry

    def _load_path(self, path: Path, image_size: Optional[int], trusted: bool) -> dict:
        path = Path(path).resolve()
        key = (str(path), int(path.stat().st_mtime))
        with self._lock:
            entry = self._path_cache.get(key)
            if entry is not None and (image_size is None or entry["image_size"] == int(image_size)):
                return entry
            model, classes, base, ckpt_size, meta = _load_checkpoint(path, trusted)
            size = int(image_size) if image_size else (ckpt_size or _CONVNEXT_VARIANTS[base][1])
            entry = self._make_entry(model, classes, base, size, meta)
            # keep the path cache small: one entry per file
            for k in [k for k in self._path_cache if k[0] == str(path)]:
                self._path_cache.pop(k, None)
            self._path_cache[key] = entry
            return entry

    # -- inference ----------------------------------------------------------

    def _run(self, entry: dict, image: Image.Image, topk: int) -> dict:
        tensor = entry["transform"](image.convert("RGB")).unsqueeze(0).to(DEVICE)
        with self._infer_lock, torch.inference_mode():
            output = entry["model"](tensor)
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.softmax(output, dim=1)[0]
            k = max(1, min(int(topk), probs.numel()))
            top_probs, top_idx = torch.topk(probs, k)
            top_probs = top_probs.tolist()
            top_idx = top_idx.tolist()
        classes = entry["classes"]
        return {
            "label": classes[top_idx[0]],
            "score": float(top_probs[0]),
            "model_base": entry["base"],
            "image_size": entry["image_size"],
            "topk": [{"label": classes[i], "score": float(p)} for i, p in zip(top_idx, top_probs)],
        }

    def predict(self, alias: str, image: Image.Image, *, topk: int = 5) -> dict:
        return self._run(self._load(alias), image, topk)

    def predict_path(
        self, path: str | Path, image: Image.Image, *, image_size: Optional[int] = None, trusted: bool = False, topk: int = 5
    ) -> dict:
        return self._run(self._load_path(Path(path), image_size, trusted), image, topk)

    def classes_for_path(self, path: str | Path, *, trusted: bool = False) -> List[str]:
        return list(self._load_path(Path(path), None, trusted)["classes"])
