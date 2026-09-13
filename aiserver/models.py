"""Dataclasses shared by the registry, the API and the ZIP import/export.

They mirror the desktop client's ``sorter.data.models`` closely enough that a
model exported by either side round-trips through the other: same field
names, same defaults, same PascalCase aliases for archives written by the
legacy Windows app.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

SUPPORTED_MODEL_MODES = (
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
)
OPENAI_MODEL_MODE = "openai"
MODEL_MODES = (*SUPPORTED_MODEL_MODES, OPENAI_MODEL_MODE)

MODEL_MODE_LABELS = {
    "convnext_tiny": "ConvNeXt-Tiny",
    "convnext_small": "ConvNeXt-Small",
    "convnext_base": "ConvNeXt-Base",
    "convnext_large": "ConvNeXt-Large",
    OPENAI_MODEL_MODE: "OpenAI",
}

MODEL_TYPES = ("Standard", "ReadOnly", "CommunityManaged")
FOREIGN_MODEL_TYPES = frozenset({"CommunityManaged", "ReadOnly"})
FEEDBACK_UPLOAD_MODES = ("Instant", "OnRunComplete", "Manual")


def normalize_upload_mode(raw: Any, *, feedback_enabled: bool) -> str:
    if isinstance(raw, bool):
        raw = None
    if isinstance(raw, str):
        s = raw.strip()
        for mode in FEEDBACK_UPLOAD_MODES:
            if s.lower() == mode.lower():
                return mode
        if s.isdigit():
            raw = int(s)
    if isinstance(raw, int) and not isinstance(raw, bool):
        if 0 <= raw < len(FEEDBACK_UPLOAD_MODES):
            return FEEDBACK_UPLOAD_MODES[raw]
    return "Instant" if feedback_enabled else "Manual"


def _coerce(cls, data: dict[str, Any] | None, aliases: dict[str, str]):
    if not data:
        return cls()
    known = {f.name for f in fields(cls)}
    normalised: dict[str, Any] = {}
    for k, v in data.items():
        if k in known:
            normalised[k] = v
        elif k in aliases:
            normalised[aliases[k]] = v
    try:
        return cls(**normalised)
    except TypeError:
        return cls(**{k: v for k, v in normalised.items() if k in known})


@dataclass
class ImageProcessingConfig:
    strategy: str = "hough"
    primer_mode: str = "hide"
    primer_radius: int = 135
    hough: dict[str, Any] = field(
        default_factory=lambda: {
            "dp": 2.0,
            "min_dist": 500,
            "param1": 100,
            "param2": 60,
            "min_radius": 150,
            "max_radius": 250,
        }
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ImageProcessingConfig":
        if not data:
            return cls()
        return cls(
            strategy=data.get("strategy", "hough"),
            primer_mode=data.get("primer_mode", "hide"),
            primer_radius=int(data.get("primer_radius", 135)),
            hough=dict(data.get("hough", cls().hough)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointEnv:
    torch: str = ""
    torchvision: str = ""
    numpy: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CheckpointEnv":
        if not data:
            return cls()

        def _pick(name: str) -> str:
            return str(data.get(name) or data.get(f"{name}_version") or "")

        return cls(torch=_pick("torch"), torchvision=_pick("torchvision"), numpy=_pick("numpy"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_empty(self) -> bool:
        return not (self.torch or self.torchvision or self.numpy)


@dataclass
class AIModelConfig:
    endpoint_url: str = ""
    api_key: str = ""
    model: str = ""
    prompt: str = ""
    image_quality: int = 100
    image_scale: int = 100

    _ALIASES = {
        "OpenAI_EndpointUrl": "endpoint_url",
        "OpenAI_APIKey": "api_key",
        "OpenAI_Model": "model",
        "OpenAI_SystemPrompt": "prompt",
        "ImageQuality": "image_quality",
        "ImageScale": "image_scale",
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AIModelConfig":
        return _coerce(cls, data, cls._ALIASES)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Mirrors the client's 27-field training config (legacy defaults)."""

    model_name: str = "convnext_tiny"
    image_directory: str = ""
    output_model_path: str = ""

    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 32
    weight_decay: float = 1e-4
    val_split: float = 0.2
    dropout_rate: float = 0.0
    freeze_backbone: bool = False
    use_workspace: bool = False
    allow_gpu: bool = True
    max_workers: int = -1
    image_size: int = 232
    train_all: bool = False

    use_swa: bool = False
    swa_start: float = 0.75
    swa_mode: str = "scheduled"
    swa_acc_threshold: float = 0.96
    swa_patience: int = 5
    swa_min_epoch: int = 10

    use_focal_loss: bool = False
    focal_gamma: float = 1.0
    stochastic_depth_prob: float = -1.0

    use_parent_classifications: bool = False

    _ALIASES = {
        "ModelName": "model_name",
        "ImageDirectory": "image_directory",
        "OutputModelPath": "output_model_path",
        "Epochs": "epochs",
        "LearningRate": "learning_rate",
        "BatchSize": "batch_size",
        "WeightDecay": "weight_decay",
        "ValSplit": "val_split",
        "DropoutRate": "dropout_rate",
        "FreezeBackbone": "freeze_backbone",
        "UseWorkspace": "use_workspace",
        "AllowGPU": "allow_gpu",
        "MaxWorkers": "max_workers",
        "ImageSize": "image_size",
        "TrainAll": "train_all",
        "UseSWA": "use_swa",
        "SWAStart": "swa_start",
        "SWAMode": "swa_mode",
        "SWAAccThreshold": "swa_acc_threshold",
        "SWAPatience": "swa_patience",
        "SWAMinEpoch": "swa_min_epoch",
        "UseFocalLoss": "use_focal_loss",
        "FocalGamma": "focal_gamma",
        "StochasticDepthProb": "stochastic_depth_prob",
        "UseParentClassifications": "use_parent_classifications",
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrainingConfig":
        cfg = _coerce(cls, data, cls._ALIASES)
        return cfg.normalized()

    def normalized(self) -> "TrainingConfig":
        """Clamp types so a JSON body with strings/ints still yields a valid config."""
        self.model_name = str(self.model_name or "convnext_tiny").lower().replace("-", "_")
        if self.model_name not in SUPPORTED_MODEL_MODES:
            self.model_name = "convnext_tiny"
        self.epochs = max(1, int(self.epochs))
        self.learning_rate = float(self.learning_rate)
        self.batch_size = max(1, int(self.batch_size))
        self.weight_decay = float(self.weight_decay)
        self.val_split = min(0.95, max(0.0, float(self.val_split)))
        self.dropout_rate = min(0.95, max(0.0, float(self.dropout_rate)))
        self.freeze_backbone = bool(self.freeze_backbone)
        self.use_workspace = bool(self.use_workspace)
        self.allow_gpu = bool(self.allow_gpu)
        self.max_workers = int(self.max_workers)
        self.image_size = max(32, int(self.image_size))
        self.train_all = bool(self.train_all)
        self.use_swa = bool(self.use_swa)
        self.swa_start = min(1.0, max(0.0, float(self.swa_start)))
        self.swa_mode = "adaptive" if str(self.swa_mode).lower() == "adaptive" else "scheduled"
        self.swa_acc_threshold = float(self.swa_acc_threshold)
        self.swa_patience = max(1, int(self.swa_patience))
        self.swa_min_epoch = max(1, int(self.swa_min_epoch))
        self.use_focal_loss = bool(self.use_focal_loss)
        self.focal_gamma = float(self.focal_gamma)
        self.stochastic_depth_prob = float(self.stochastic_depth_prob)
        self.use_parent_classifications = bool(self.use_parent_classifications)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Model:
    id: int = 0
    name: str = ""
    cartridge_name: str = ""
    model_mode: str = "convnext_tiny"
    model_type: str = "Standard"
    community_model_uid: str | None = None
    model_version: int = 1
    enable_image_processing: bool = True
    image_processing: ImageProcessingConfig = field(default_factory=ImageProcessingConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    ai_model_config: AIModelConfig = field(default_factory=AIModelConfig)
    use_primer_mask: bool = False
    hide_primer: bool = True
    primer_mask_size: int = 135
    last_training_date: str | None = None
    last_training_duration: int = 0
    trained_image_count: int = 0
    training_confusion_table: str | None = None
    feedback_loop_enabled: bool = False
    feedback_loop_confidence_floor: int = 95
    feedback_loop_upload_mode: str = "Manual"
    model_path: str | None = None
    checkpoint_env: CheckpointEnv = field(default_factory=CheckpointEnv)
    # Server-only fields
    serve_enabled: bool = False
    serve_alias: str | None = None
    owner_client_id: int | None = None
    notes: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    last_val_acc: float | None = None

    @classmethod
    def from_row(cls, row: Any) -> "Model":
        def _parse(s: str | None) -> dict[str, Any] | None:
            if not s:
                return None
            try:
                return json.loads(s)
            except (TypeError, ValueError):
                return None

        keys = row.keys()

        def _opt(name: str, default: Any = None) -> Any:
            return row[name] if name in keys else default

        fb_enabled = bool(row["feedback_loop_enabled"])
        return cls(
            id=row["id"],
            name=row["name"],
            cartridge_name=row["cartridge_name"] or "",
            model_mode=row["model_mode"],
            model_type=row["model_type"],
            community_model_uid=row["community_model_uid"],
            model_version=row["model_version"],
            enable_image_processing=bool(row["enable_image_processing"]),
            image_processing=ImageProcessingConfig.from_dict(_parse(row["image_processing_json"])),
            training_config=TrainingConfig.from_dict(_parse(row["training_config_json"])),
            ai_model_config=AIModelConfig.from_dict(_parse(row["ai_model_config_json"])),
            use_primer_mask=bool(row["use_primer_mask"]),
            hide_primer=bool(row["hide_primer"]),
            primer_mask_size=row["primer_mask_size"],
            last_training_date=row["last_training_date"],
            last_training_duration=row["last_training_duration"],
            trained_image_count=row["trained_image_count"],
            training_confusion_table=row["training_confusion_table"],
            feedback_loop_enabled=fb_enabled,
            feedback_loop_confidence_floor=row["feedback_loop_confidence_floor"],
            feedback_loop_upload_mode=normalize_upload_mode(
                row["feedback_loop_upload_mode"], feedback_enabled=fb_enabled
            ),
            model_path=row["model_path"],
            checkpoint_env=CheckpointEnv.from_dict(_parse(row["checkpoint_env_json"])),
            serve_enabled=bool(_opt("serve_enabled", 0)),
            serve_alias=_opt("serve_alias"),
            owner_client_id=_opt("owner_client_id"),
            notes=_opt("notes", "") or "",
            created_at=_opt("created_at"),
            updated_at=_opt("updated_at"),
            last_val_acc=_opt("last_val_acc"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def alias(self) -> str:
        """The name this model is served under on the OpenAI endpoints."""
        return (self.serve_alias or self.name or f"model-{self.id}").strip()


def is_foreign_model(model: Model | None) -> bool:
    return bool(model is not None and model.model_type in FOREIGN_MODEL_TYPES)


def is_openai_model(model: Model | None) -> bool:
    return bool(model is not None and model.model_mode == OPENAI_MODEL_MODE)


def is_trainable(model: Model | None) -> bool:
    return model is not None and not is_foreign_model(model) and not is_openai_model(model)


@dataclass
class Headstamp:
    id: int = 0
    model_id: int = 0
    name: str = ""
    slot: int = 0
    parent_name: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> "Headstamp":
        keys = row.keys()
        return cls(
            id=row["id"],
            model_id=row["model_id"],
            name=row["name"],
            slot=row["slot"] if "slot" in keys else 0,
            parent_name=row["parent_name"] if "parent_name" in keys else None,
        )


@dataclass
class BoundClient:
    id: int = 0
    name: str = ""
    token_hash: str = ""
    created_at: str = ""
    last_seen_at: str | None = None
    revoked: bool = False
    client_version: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> "BoundClient":
        return cls(
            id=row["id"],
            name=row["name"],
            token_hash=row["token_hash"],
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
            revoked=bool(row["revoked"]),
            client_version=row["client_version"],
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "revoked": self.revoked,
            "client_version": self.client_version,
        }
