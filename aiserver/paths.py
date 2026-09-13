"""On-disk layout for everything the server writes.

    <data root>/
    ├── config/
    │   ├── server.db          <- SQLite database (models, headstamps, jobs, clients)
    │   └── msal_cache.bin     <- community login token cache
    ├── models/<model_id>/
    │   ├── images/            <- training images ({label}__{ticks}.jpg)
    │   ├── trainedmodel/      <- <model_id>.pth checkpoint (+ _swa variant)
    │   ├── reports/           <- evaluation results (JSON)
    │   └── uploads/           <- scratch for evaluation uploads
    ├── downloads/             <- community archives before import
    └── logs/                  <- training-<stamp>.log

The data root resolves, in order, to ``CASESORTER_SERVER_DATA_DIR`` (env),
``config.DATA_DIR`` (when set), then ``<repo>/data``. Keeping it inside the
repo by default matches how this server is deployed: a plain clone that the
user runs from where it landed.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_DATA_DIR = "CASESORTER_SERVER_DATA_DIR"

_repo_root = Path(__file__).resolve().parent.parent
_data_root: Path | None = None


def repo_root() -> Path:
    return _repo_root


def set_data_root(path: Path | str | None) -> None:
    """Pin the data root (tests and ``config.DATA_DIR`` use this)."""
    global _data_root
    _data_root = Path(path).resolve() if path else None


def data_root() -> Path:
    if _data_root is not None:
        return _data_root
    env = os.environ.get(ENV_DATA_DIR)
    if env:
        return Path(env).resolve()
    return _repo_root / "data"


def ensure_layout() -> Path:
    root = data_root()
    for sub in ("config", "models", "downloads", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def config_dir() -> Path:
    return data_root() / "config"


def db_path() -> Path:
    return config_dir() / "server.db"


def msal_cache_path() -> Path:
    return config_dir() / "msal_cache.bin"


def models_dir() -> Path:
    return data_root() / "models"


def model_dir(model_id: int) -> Path:
    return models_dir() / str(int(model_id))


def model_images_dir(model_id: int) -> Path:
    return model_dir(model_id) / "images"


def model_trained_dir(model_id: int) -> Path:
    return model_dir(model_id) / "trainedmodel"


def model_reports_dir(model_id: int) -> Path:
    return model_dir(model_id) / "reports"


def model_uploads_dir(model_id: int) -> Path:
    return model_dir(model_id) / "uploads"


def model_checkpoint_path(model_id: int) -> Path:
    return model_trained_dir(model_id) / f"{int(model_id)}.pth"


def downloads_dir() -> Path:
    return data_root() / "downloads"


def logs_dir() -> Path:
    return data_root() / "logs"
