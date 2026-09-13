"""Model ZIP export/import, byte-compatible with the desktop client and the
legacy Windows app.

    manifest.json
    model/<filename>.pth        (ModelOnly | ModelAndImages)
    images/<label>__<ticks>.jpg (ImagesOnly | ModelAndImages)

Manifest top level is PascalCase (``ModelName``, ``CartridgeName``,
``Headstamps``, ``ExportMode``, ``ModelInfo``); ``ModelInfo`` is written in
snake_case and read in either spelling. Security checks on import mirror the
client: traversal entries, decompression bombs and unexpected extensions are
rejected before anything touches the filesystem.
"""

from __future__ import annotations

import enum
import json
import os
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from . import paths
from .models import (
    MODEL_MODES,
    OPENAI_MODEL_MODE,
    AIModelConfig,
    CheckpointEnv,
    ImageProcessingConfig,
    Model,
    TrainingConfig,
    is_foreign_model,
    normalize_upload_mode,
)
from .store import HeadstampRepo, ModelRepo


class ExportMode(str, enum.Enum):
    MODEL_ONLY = "ModelOnly"
    MODEL_AND_IMAGES = "ModelAndImages"
    IMAGES_ONLY = "ImagesOnly"

    @classmethod
    def parse(cls, raw: str | None) -> "ExportMode":
        text = (raw or "").replace("_", "").replace(" ", "").lower()
        for mode in cls:
            if mode.value.lower() == text:
                return mode
        aliases = {"model": cls.MODEL_ONLY, "images": cls.IMAGES_ONLY, "both": cls.MODEL_AND_IMAGES, "": cls.MODEL_AND_IMAGES}
        if text in aliases:
            return aliases[text]
        raise ValueError(f"Unknown export mode {raw!r}")


_VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VALID_MODEL_EXTS = {".pth", ".pt", ""}
_LEGACY_ZIP_MODEL_NAME = "trainedmodel.zip"
_MAX_COMPRESSION_RATIO = 100
_RATIO_CHECK_MIN_BYTES = 1_000_000

WINFORMS_MODELMODE_OPENAI = 2
_WINFORMS_MODELMODE_INT_TO_STR = {
    0: "convnext_tiny", 1: "convnext_tiny", WINFORMS_MODELMODE_OPENAI: OPENAI_MODEL_MODE,
    3: "convnext_large", 4: "convnext_tiny", 5: "convnext_tiny", 6: "convnext_base",
    7: "convnext_small", 8: "convnext_tiny",
}


def _normalize(entry_name: str) -> str:
    return entry_name.replace("\\", "/").lstrip("/")


def _is_traversal(rel: str) -> bool:
    return any(part == ".." for part in PurePosixPath(rel).parts)


def model_to_export_dict(m: Model) -> dict[str, Any]:
    d = asdict(m)
    for key in ("model_path", "serve_enabled", "serve_alias", "owner_client_id", "notes",
                "created_at", "updated_at", "last_val_acc", "cartridge_name"):
        d.pop(key, None)
    d["cartridge_id"] = 0
    if d.get("training_config"):
        d["training_config"].pop("image_directory", None)
        d["training_config"].pop("output_model_path", None)
    if d.get("ai_model_config"):
        d["ai_model_config"].pop("api_key", None)
    return d


def _normalize_model_mode(raw: Any) -> str:
    if isinstance(raw, str):
        rl = raw.strip().lower().replace("-", "_").replace(" ", "_")
        if rl in MODEL_MODES:
            return rl
        if rl.startswith("convnext") and not rl.startswith("convnext_"):
            candidate = f"convnext_{rl[len('convnext'):]}"
            if candidate in MODEL_MODES:
                return candidate
        return "convnext_tiny"
    if isinstance(raw, int) and not isinstance(raw, bool):
        return _WINFORMS_MODELMODE_INT_TO_STR.get(raw, "convnext_tiny")
    return "convnext_tiny"


def _normalize_model_type(raw: Any) -> str:
    if isinstance(raw, str) and raw in ("Standard", "ReadOnly", "CommunityManaged"):
        return raw
    if isinstance(raw, int) and not isinstance(raw, bool):
        return ("Standard", "ReadOnly", "CommunityManaged")[raw] if 0 <= raw <= 2 else "Standard"
    return "Standard"


def _g(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def model_from_export_dict(d: dict[str, Any]) -> Model:
    if not d:
        return Model()
    fb_enabled = bool(_g(d, "feedback_loop_enabled", "FeedbackLoopEnabled", default=False))
    return Model(
        id=0,
        name=_g(d, "name", "Name", default=""),
        model_mode=_normalize_model_mode(_g(d, "model_mode", "ModelMode")),
        model_type=_normalize_model_type(_g(d, "model_type", "ModelType", default="Standard")),
        community_model_uid=_g(d, "community_model_uid", "CommunityModelUID"),
        model_version=int(_g(d, "model_version", "ModelVersion", default=1)),
        enable_image_processing=bool(_g(d, "enable_image_processing", "EnableImageProcessing", default=True)),
        image_processing=ImageProcessingConfig.from_dict(_g(d, "image_processing", "ImageProcessingConfig")),
        training_config=TrainingConfig.from_dict(
            _g(d, "training_config", "PythonTrainingConfig", "ModelTrainingConfig")
        ),
        ai_model_config=AIModelConfig.from_dict(_g(d, "ai_model_config", "AIModelConfig")),
        use_primer_mask=bool(_g(d, "use_primer_mask", "UsePrimerMask", default=False)),
        hide_primer=bool(_g(d, "hide_primer", "HidePrimer", default=True)),
        primer_mask_size=int(_g(d, "primer_mask_size", "PrimerMaskSize", default=135)),
        last_training_date=_g(d, "last_training_date", "LastTrainingDate"),
        last_training_duration=int(_g(d, "last_training_duration", "LastTrainingDuration", default=0)),
        trained_image_count=int(_g(d, "trained_image_count", "TrainedImageCount", default=0)),
        training_confusion_table=_g(d, "training_confusion_table", "TrainingConfusionTable"),
        feedback_loop_enabled=fb_enabled,
        feedback_loop_confidence_floor=int(
            _g(d, "feedback_loop_confidence_floor", "FeedbackLoopConfidenceFloor", default=95)
        ),
        feedback_loop_upload_mode=normalize_upload_mode(
            _g(d, "feedback_loop_upload_mode", "FeedbackLoopUploadMode"), feedback_enabled=fb_enabled
        ),
        model_path=None,
        checkpoint_env=CheckpointEnv.from_dict(_g(d, "checkpoint_env", "CheckpointEnv")),
    )


def build_manifest(model: Model, cartridge_name: str, headstamps: list[str], mode: ExportMode) -> dict[str, Any]:
    return {
        "ModelName": model.name,
        "CartridgeName": cartridge_name,
        "Headstamps": list(headstamps),
        "ExportMode": mode.value,
        "ModelInfo": model_to_export_dict(model),
    }


def export_model(
    output_path: Path | str,
    model: Model,
    cartridge_name: str,
    headstamps: list[str],
    *,
    mode: ExportMode = ExportMode.MODEL_AND_IMAGES,
    model_file: Path | str | None = None,
    images_dir: Path | str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    manifest = build_manifest(model, cartridge_name, headstamps, mode)

    image_files: list[Path] = []
    if mode in (ExportMode.IMAGES_ONLY, ExportMode.MODEL_AND_IMAGES) and images_dir:
        d = Path(images_dir)
        if d.exists():
            image_files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _VALID_IMAGE_EXTS)

    include_model = (
        mode in (ExportMode.MODEL_ONLY, ExportMode.MODEL_AND_IMAGES)
        and model_file is not None
        and Path(model_file).exists()
    )
    total = 1 + (1 if include_model else 0) + len(image_files)
    step = 0

    def _bump() -> None:
        nonlocal step
        step += 1
        if progress is not None:
            progress(step, total)

    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            _bump()
            if include_model:
                zf.write(str(model_file), arcname=f"model/{Path(model_file).name}")
                _bump()
            for img in image_files:
                zf.write(str(img), arcname=f"images/{img.name}")
                _bump()
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return out


def read_manifest(zip_path: Path | str) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for entry in zf.infolist():
            if _normalize(entry.filename) == "manifest.json":
                with zf.open(entry) as f:
                    return json.loads(f.read().decode("utf-8-sig"))
    raise ValueError(f"{zip_path}: no manifest.json found")


def validate_archive(zip_path: Path | str) -> dict[str, Any]:
    """Pre-scan an archive; raise ValueError on anything dangerous. Returns a summary."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = list(zf.infolist())
    n_images = 0
    has_model = False
    for entry in entries:
        rel = _normalize(entry.filename)
        if _is_traversal(rel):
            raise ValueError(f"Refusing to import: traversal entry {entry.filename!r}")
        if (
            entry.file_size >= _RATIO_CHECK_MIN_BYTES
            and entry.compress_size > 0
            and entry.file_size / entry.compress_size > _MAX_COMPRESSION_RATIO
        ):
            raise ValueError(f"Refusing to import: entry {entry.filename!r} has an implausible compression ratio")
        posix = PurePosixPath(rel)
        if rel.endswith("/") or rel == "manifest.json":
            continue
        if posix.parts and posix.parts[0] == "images":
            if Path(posix.parts[-1]).suffix.lower() not in _VALID_IMAGE_EXTS:
                raise ValueError(f"Refusing to import: unexpected image entry {entry.filename!r}")
            n_images += 1
        elif posix.parts and posix.parts[0] == "model":
            basename = posix.parts[-1]
            suffix = Path(basename).suffix.lower()
            if suffix not in _VALID_MODEL_EXTS and basename.lower() != _LEGACY_ZIP_MODEL_NAME:
                raise ValueError(f"Refusing to import: unexpected model entry {entry.filename!r}")
            has_model = True
    if not any(_normalize(e.filename) == "manifest.json" for e in entries):
        raise ValueError("Archive has no manifest.json (is this a model export?)")
    return {"images": n_images, "has_model": has_model, "entries": len(entries)}


def find_update_target(zip_path: Path | str, repo: ModelRepo) -> Model | None:
    info = read_manifest(zip_path).get("ModelInfo") or {}
    uid = _g(info, "community_model_uid", "CommunityModelUID")
    return repo.find_by_community_uid(str(uid)) if uid else None


def _merge_onto_installed(incoming: Model, existing: Model, *, name: str) -> Model:
    incoming.id = existing.id
    incoming.name = name
    incoming.cartridge_name = existing.cartridge_name or incoming.cartridge_name
    incoming.ai_model_config = existing.ai_model_config
    if is_foreign_model(existing):
        incoming.model_type = existing.model_type
    incoming.model_path = existing.model_path
    if incoming.checkpoint_env.is_empty():
        incoming.checkpoint_env = existing.checkpoint_env
    incoming.feedback_loop_enabled = bool(existing.feedback_loop_enabled and incoming.feedback_loop_enabled)
    if incoming.feedback_loop_enabled:
        incoming.feedback_loop_upload_mode = existing.feedback_loop_upload_mode
    incoming.serve_enabled = existing.serve_enabled
    incoming.serve_alias = existing.serve_alias
    incoming.owner_client_id = existing.owner_client_id
    incoming.notes = existing.notes
    return incoming


def import_model(
    zip_path: Path | str,
    *,
    model_repo: ModelRepo,
    headstamp_repo: HeadstampRepo,
    cartridge_name_override: str | None = None,
    model_name_override: str | None = None,
    update_existing: bool = True,
    community_download: bool = False,
    owner_client_id: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Model:
    """Import an archive into the registry; returns the saved model row."""
    validate_archive(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = list(zf.infolist())
        manifest_entry = next(e for e in entries if _normalize(e.filename) == "manifest.json")
        manifest = json.loads(zf.read(manifest_entry).decode("utf-8-sig"))

        model = model_from_export_dict(manifest.get("ModelInfo") or {})
        if model.model_mode not in MODEL_MODES:
            model.model_mode = "convnext_tiny"
        if community_download:
            model.model_type = "CommunityManaged"

        existing = (
            model_repo.find_by_community_uid(model.community_model_uid)
            if update_existing and model.community_model_uid
            else None
        )
        if existing is not None and not cartridge_name_override:
            model.cartridge_name = existing.cartridge_name
        else:
            model.cartridge_name = cartridge_name_override or manifest.get("CartridgeName") or "Imported"

        if not model.community_model_uid:
            model.last_training_date = None
            model.last_training_duration = 0
            model.trained_image_count = 0
            model.training_confusion_table = None

        if existing is not None:
            saved = model_repo.update(_merge_onto_installed(model, existing, name=model_name_override or existing.name))
        else:
            desired = model_name_override or model.name or manifest.get("ModelName") or "Imported"
            model.name = model_repo.unique_name(desired)
            model.owner_client_id = owner_client_id
            saved = model_repo.create(model)

        img_dir = paths.model_images_dir(saved.id)
        mod_dir = paths.model_trained_dir(saved.id)
        img_dir.mkdir(parents=True, exist_ok=True)
        mod_dir.mkdir(parents=True, exist_ok=True)

        for hs_name in manifest.get("Headstamps") or []:
            if hs_name:
                try:
                    headstamp_repo.add(saved.id, str(hs_name))
                except Exception:  # noqa: BLE001
                    pass

        total = len(entries)
        for i, entry in enumerate(entries, 1):
            rel = _normalize(entry.filename)
            if rel.endswith("/") or rel == "manifest.json":
                if progress:
                    progress(i, total)
                continue
            posix = PurePosixPath(rel)
            if posix.parts and posix.parts[0] == "images":
                _extract_to(zf, entry, img_dir / posix.parts[-1])
            elif posix.parts and posix.parts[0] == "model":
                basename = posix.parts[-1]
                suffix = Path(basename).suffix.lower()
                if basename.lower() == _LEGACY_ZIP_MODEL_NAME or not suffix:
                    suffix = ".pth"
                dest = mod_dir / f"{saved.id}{suffix}"
                _extract_to(zf, entry, dest)
                saved.model_path = str(dest)
                saved = model_repo.update(saved)
            if progress:
                progress(i, total)

        # Any class present as an image but missing as a headstamp becomes one.
        from .image_store import class_counts

        for label in class_counts(img_dir):
            try:
                headstamp_repo.add(saved.id, label)
            except Exception:  # noqa: BLE001
                pass
        return model_repo.get(saved.id)  # type: ignore[return-value]


def _extract_to(zf: zipfile.ZipFile, entry: zipfile.ZipInfo, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with zf.open(entry) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out)
    os.replace(tmp, dest)
