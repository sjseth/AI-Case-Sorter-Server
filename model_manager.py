"""Loads trained ConvNeXt checkpoints and runs inference against them.

This is a slimmed-down, PyTorch-only port of the inference path used by the
parent CaseSorter project (see python_env/inference_worker.py). It is kept
self-contained so this folder can be lifted into its own repo without
dragging the rest of the application along. Only checkpoints produced by
python_env/train_convnext.py (and its SWA variants) are supported.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        log.info("CUDA unavailable; using CPU")
        return torch.device("cpu")
    try:
        probe = torch.randn(1, 3, 8, 8, device="cuda")
        _ = probe.sum().item()
        log.info("Using CUDA (%s)", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    except Exception as exc:
        log.warning("CUDA reported available but test op failed (%s); falling back to CPU", exc)
        return torch.device("cpu")


DEVICE: torch.device = _select_device()


# ---------------------------------------------------------------------------
# ConvNeXt variants
# ---------------------------------------------------------------------------

# (torchvision factory, default image size). Weights intentionally default to
# None: the trained state_dict overwrites every parameter anyway, and using
# None avoids the torchvision pretrained-weight download on first run (useful
# on air-gapped machines).
_CONVNEXT_VARIANTS: Dict[str, Tuple[callable, int]] = {
    "convnext_tiny":  (models.convnext_tiny,  224),
    "convnext_small": (models.convnext_small, 224),
    "convnext_base":  (models.convnext_base,  224),
    "convnext_large": (models.convnext_large, 224),
}

# Older training runs sometimes wrote the short form into the checkpoint.
_BASE_ALIASES = {
    "tiny":  "convnext_tiny",
    "small": "convnext_small",
    "base":  "convnext_base",
    "large": "convnext_large",
}


def _resolve_base(name: str) -> str:
    key = (name or "").lower()
    return _BASE_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _build_model(base: str, num_classes: int, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    """Construct a ConvNeXt with the classifier shape implied by ``state_dict``.

    train_convnext.py wraps the head in ``Sequential(Dropout, Linear)`` at
    classifier index 2. Some checkpoints we've seen in the wild use a bare
    Linear at the same index, and a third layout puts an extra Dropout/Linear
    at classifier index 3. We detect each by looking at the keys.
    """
    base = _resolve_base(base)
    if base not in _CONVNEXT_VARIANTS:
        raise ValueError(f"Unsupported model base: {base!r}")

    factory, _ = _CONVNEXT_VARIANTS[base]
    model = factory(weights=None)
    in_features = model.classifier[2].in_features

    if "classifier.3.weight" in state_dict:
        model.classifier = nn.Sequential(
            model.classifier[0],  # LayerNorm
            model.classifier[1],  # Flatten
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


def _load_checkpoint(path: Path) -> Tuple[nn.Module, List[str], str]:
    log.info("Loading checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    base = ckpt.get("base", "convnext_base")
    classes = list(ckpt["classes"])
    state_dict: Dict[str, torch.Tensor] = ckpt["model_state_dict"]

    # SWA / DataParallel wrappers prepend "module." to every key.
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    # SWA also stores an n_averaged counter that we don't want to load.
    state_dict.pop("n_averaged", None)

    model = _build_model(base, num_classes=len(classes), state_dict=state_dict)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()

    log.info("Loaded %s (%d classes) on %s", _resolve_base(base), len(classes), DEVICE)
    return model, classes, _resolve_base(base)


# ---------------------------------------------------------------------------
# Public manager
# ---------------------------------------------------------------------------

class ModelManager:
    """Caches loaded checkpoints keyed by the alias from config.MODELS."""

    def __init__(
        self,
        aliases: Dict[str, str],
        options: Optional[Dict[str, dict]] = None,
        base_dir: Optional[Path] = None,
    ):
        self._aliases = dict(aliases)
        self._options = dict(options or {})
        self._base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
        self._cache: Dict[str, dict] = {}
        self._lock = threading.RLock()

    def aliases(self) -> List[str]:
        return list(self._aliases)

    def has(self, alias: str) -> bool:
        return alias in self._aliases

    def preload(self) -> None:
        for alias in self._aliases:
            self._load(alias)

    def _resolve_path(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = self._base_dir / path
        return path

    def _image_size_for(self, alias: str, base: str) -> int:
        override = self._options.get(alias, {}).get("image_size")
        if override is not None:
            return int(override)
        _, default_size = _CONVNEXT_VARIANTS[base]
        return default_size

    def _load(self, alias: str) -> dict:
        with self._lock:
            if alias in self._cache:
                return self._cache[alias]
            if alias not in self._aliases:
                raise KeyError(f"Unknown model alias: {alias!r}")

            path = self._resolve_path(self._aliases[alias])
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            model, classes, base = _load_checkpoint(path)
            size = self._image_size_for(alias, base)
            entry = {
                "model": model,
                "classes": classes,
                "base": base,
                "image_size": size,
                "transform": transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]),
            }
            self._cache[alias] = entry
            return entry

    def predict(self, alias: str, image: Image.Image) -> dict:
        entry = self._load(alias)
        tensor = entry["transform"](image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            output = entry["model"](tensor)
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.softmax(output, dim=1)[0]

        top_prob, top_idx = torch.max(probs, 0)
        return {
            "label": entry["classes"][int(top_idx.item())],
            "score": float(top_prob.item()),
            "model_base": entry["base"],
        }
