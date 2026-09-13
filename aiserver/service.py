"""Application services: the one place that knows about the registry, the
job queue, the inference cache and the community client together.

Both the remote-client API and the web UI go through ``AppService``; neither
touches the repositories directly.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import shutil
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from . import evaluator, image_store, model_io, paths
from .db import Database
from .models import SUPPORTED_MODEL_MODES, AIModelConfig, CheckpointEnv, Model, TrainingConfig, is_openai_model, is_trainable
from .store import ClientRepo, HeadstampRepo, JobRepo, ModelRepo, SettingsRepo
from .training.manager import JobContext, JobManager, run_training

log = logging.getLogger(__name__)

SESSION_TTL_S = 12 * 3600


class ServiceError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class NotFound(ServiceError):
    def __init__(self, message: str = "Not found"):
        super().__init__(message, 404)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AppService:
    def __init__(self, config: Any, model_manager: Any, *, db: Database | None = None):
        self.config = config
        self.manager = model_manager
        paths.ensure_layout()
        self.db = db or Database()
        self.models = ModelRepo(self.db)
        self.headstamps = HeadstampRepo(self.db)
        self.clients = ClientRepo(self.db)
        self.settings = SettingsRepo(self.db)
        self.jobs = JobManager(JobRepo(self.db), on_event=self._on_job_event)
        self._sessions: dict[str, float] = {}
        self._auth = None
        self._auth_lock = threading.Lock()
        self._catalogue_cache: tuple[float, list[dict[str, Any]]] | None = None
        self.started_at = time.time()

    def start(self) -> None:
        self.jobs.start()
        self.sync_serving()

    def stop(self) -> None:
        self.jobs.stop()

    # ------------------------------------------------------------------
    # Admin auth (web UI)
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_password(password: str, salt: str | None = None) -> str:
        salt = salt or secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000).hex()
        return f"{salt}${digest}"

    def admin_password_set(self) -> bool:
        return bool(getattr(self.config, "ADMIN_PASSWORD", None)) or bool(self.settings.get("admin_password_hash"))

    def admin_required(self) -> bool:
        """Loopback-only servers work without a password; anything else needs one."""
        host = str(getattr(self.config, "HOST", "127.0.0.1") or "")
        if self.admin_password_set():
            return True
        return host not in ("127.0.0.1", "localhost", "::1")

    def set_admin_password(self, password: str, *, current: str | None = None) -> None:
        if self.admin_password_set() and not self.verify_admin_password(current or ""):
            raise ServiceError("Current password is incorrect", 403)
        if len(password or "") < 6:
            raise ServiceError("Password must be at least 6 characters")
        self.settings.set("admin_password_hash", self._hash_password(password))

    def verify_admin_password(self, password: str) -> bool:
        configured = getattr(self.config, "ADMIN_PASSWORD", None)
        if configured and hmac.compare_digest(str(configured), password or ""):
            return True
        stored = self.settings.get("admin_password_hash")
        if not stored or "$" not in stored:
            return False
        salt, _ = stored.split("$", 1)
        return hmac.compare_digest(self._hash_password(password or "", salt), stored)

    def create_session(self) -> str:
        token = secrets.token_urlsafe(32)
        self._sessions[token] = time.time() + SESSION_TTL_S
        return token

    def session_valid(self, token: str | None) -> bool:
        if not token:
            return False
        exp = self._sessions.get(token)
        if exp is None:
            return False
        if exp < time.time():
            self._sessions.pop(token, None)
            return False
        return True

    def drop_session(self, token: str | None) -> None:
        if token:
            self._sessions.pop(token, None)

    def api_key_valid(self, token: str | None) -> bool:
        expected = (getattr(self.config, "API_KEY", None) or "").strip()
        return bool(expected and token and hmac.compare_digest(expected, token))

    # ------------------------------------------------------------------
    # Server info
    # ------------------------------------------------------------------

    def server_info(self) -> dict[str, Any]:
        from . import __version__
        from model_manager import device_info

        return {
            "name": "CaseSorter AI Server",
            "version": __version__,
            "host": getattr(self.config, "HOST", None),
            "port": getattr(self.config, "PORT", None),
            "uptime_seconds": int(time.time() - self.started_at),
            "data_root": str(paths.data_root()),
            "device": device_info(),
            "api_key_required": bool((getattr(self.config, "API_KEY", None) or "").strip()),
            "remote_clients_enabled": bool(self.settings.get("allow_remote_clients", True)),
            "served_models": self.manager.describe(),
            "active_jobs": self.jobs.active(),
            "capabilities": ["serve", "train", "evaluate", "images", "export", "import", "community", "bind"],
            "supported_modes": list(SUPPORTED_MODEL_MODES),
        }

    # ------------------------------------------------------------------
    # Serving registry
    # ------------------------------------------------------------------

    def sync_serving(self) -> None:
        wanted: dict[str, Model] = {}
        for m in self.models.list():
            if m.serve_enabled and m.model_path and Path(m.model_path).exists():
                wanted[m.alias()] = m
        for alias in self.manager.aliases():
            if self.manager.source(alias) == "registry" and alias not in wanted:
                self.manager.unregister(alias)
        for alias, m in wanted.items():
            if self.manager.source(alias) == "config":
                log.warning("Model %r alias %r collides with config.MODELS; the config entry wins", m.name, alias)
                continue
            self.manager.register(alias, m.model_path, {"image_size": m.training_config.image_size}, trusted=False, source="registry")

    def set_serving(self, model_id: int, enabled: bool, alias: str | None = None) -> Model:
        m = self._model(model_id)
        if enabled and not (m.model_path and Path(m.model_path).exists()):
            raise ServiceError("This model has no trained checkpoint to serve. Train or import it first.")
        if enabled and is_openai_model(m):
            raise ServiceError("An OpenAI-mode model has no checkpoint to serve.")
        if alias is not None:
            alias = alias.strip() or None
            if alias:
                other = self.models.find_by_alias(alias)
                if other and other.id != m.id:
                    raise ServiceError(f"Alias {alias!r} is already used by model {other.name!r}")
                if self.manager.source(alias) == "config":
                    raise ServiceError(f"Alias {alias!r} is reserved by config.MODELS")
            m.serve_alias = alias
        m.serve_enabled = bool(enabled)
        m = self.models.update(m)
        self.sync_serving()
        return m

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def _model(self, model_id: int) -> Model:
        m = self.models.get(model_id)
        if m is None:
            raise NotFound(f"Model {model_id} not found")
        return m

    def model_view(self, m: Model) -> dict[str, Any]:
        img_dir = paths.model_images_dir(m.id)
        counts = image_store.class_counts(img_dir)
        d = m.to_dict()
        d["ai_model_config"].pop("api_key", None)
        d.update(
            {
                "alias": m.alias(),
                "trainable": is_trainable(m),
                "has_checkpoint": bool(m.model_path and Path(m.model_path).exists()),
                "checkpoint_size": (Path(m.model_path).stat().st_size if m.model_path and Path(m.model_path).exists() else None),
                "image_count": sum(counts.values()),
                "class_counts": counts,
                "headstamps": self.headstamps.names(m.id),
                "is_serving": m.serve_enabled and m.alias() in self.manager.aliases(),
                "active_job": self.jobs.running_for_model(m.id),
                "mode_label": {"convnext_tiny": "ConvNeXt-Tiny", "convnext_small": "ConvNeXt-Small", "convnext_base": "ConvNeXt-Base", "convnext_large": "ConvNeXt-Large", "openai": "OpenAI"}.get(m.model_mode, m.model_mode),
            }
        )
        return d

    def list_models(self) -> list[dict[str, Any]]:
        return [self.model_view(m) for m in self.models.list()]

    def get_model(self, model_id: int) -> dict[str, Any]:
        return self.model_view(self._model(model_id))

    def create_model(self, data: dict[str, Any], *, client_id: int | None = None) -> dict[str, Any]:
        name = str(data.get("name") or "").strip()
        if not name:
            raise ServiceError("Model name is required")
        mode = str(data.get("model_mode") or "convnext_tiny").lower().replace("-", "_")
        if mode not in SUPPORTED_MODEL_MODES and mode != "openai":
            raise ServiceError(f"Unsupported model_mode {mode!r}")
        tc = TrainingConfig.from_dict(data.get("training_config") or {})
        if mode != "openai":
            tc.model_name = mode
        m = Model(
            name=self.models.unique_name(name),
            cartridge_name=str(data.get("cartridge_name") or data.get("cartridge") or "").strip(),
            model_mode=mode,
            training_config=tc,
            ai_model_config=AIModelConfig.from_dict(data.get("ai_model_config") or {}),
            hide_primer=bool(data.get("hide_primer", True)),
            use_primer_mask=bool(data.get("use_primer_mask", False)),
            primer_mask_size=int(data.get("primer_mask_size", 135)),
            notes=str(data.get("notes") or ""),
            owner_client_id=client_id,
        )
        m = self.models.create(m)
        paths.model_images_dir(m.id).mkdir(parents=True, exist_ok=True)
        paths.model_trained_dir(m.id).mkdir(parents=True, exist_ok=True)
        for hs in data.get("headstamps") or []:
            if str(hs).strip():
                self.headstamps.add(m.id, str(hs))
        return self.model_view(m)

    _EDITABLE = {"name", "cartridge_name", "model_mode", "hide_primer", "use_primer_mask", "primer_mask_size", "notes",
                 "enable_image_processing", "feedback_loop_enabled", "feedback_loop_confidence_floor",
                 "feedback_loop_upload_mode", "serve_alias"}

    def update_model(self, model_id: int, data: dict[str, Any]) -> dict[str, Any]:
        m = self._model(model_id)
        for key in self._EDITABLE:
            if key in data and data[key] is not None:
                value = data[key]
                if key == "name":
                    value = str(value).strip()
                    if not value:
                        raise ServiceError("Model name is required")
                    other = self.models.get_by_name(value)
                    if other and other.id != m.id:
                        raise ServiceError(f"A model named {value!r} already exists")
                elif key == "model_mode":
                    value = str(value).lower().replace("-", "_")
                    if value not in SUPPORTED_MODEL_MODES and value != "openai":
                        raise ServiceError(f"Unsupported model_mode {value!r}")
                    if value != "openai":
                        m.training_config.model_name = value
                elif key in ("primer_mask_size", "feedback_loop_confidence_floor"):
                    value = int(value)
                elif key in ("hide_primer", "use_primer_mask", "enable_image_processing", "feedback_loop_enabled"):
                    value = bool(value)
                elif key == "serve_alias":
                    value = str(value).strip() or None
                setattr(m, key, value)
        if "training_config" in data and isinstance(data["training_config"], dict):
            merged = {**m.training_config.to_dict(), **data["training_config"]}
            m.training_config = TrainingConfig.from_dict(merged)
            if m.model_mode != "openai":
                m.training_config.model_name = m.model_mode
        if "ai_model_config" in data and isinstance(data["ai_model_config"], dict):
            merged = {**m.ai_model_config.to_dict(), **data["ai_model_config"]}
            m.ai_model_config = AIModelConfig.from_dict(merged)
        if "image_processing" in data and isinstance(data["image_processing"], dict):
            from .models import ImageProcessingConfig

            m.image_processing = ImageProcessingConfig.from_dict({**m.image_processing.to_dict(), **data["image_processing"]})
        m = self.models.update(m)
        self.sync_serving()
        return self.model_view(m)

    def delete_model(self, model_id: int) -> None:
        m = self._model(model_id)
        if self.jobs.running_for_model(m.id):
            raise ServiceError("A job is running for this model; cancel it first", 409)
        self.models.delete(m.id)
        self.sync_serving()
        self.manager.invalidate()
        shutil.rmtree(paths.model_dir(m.id), ignore_errors=True)

    # ------------------------------------------------------------------
    # Headstamps
    # ------------------------------------------------------------------

    def set_headstamps(self, model_id: int, names: list[str]) -> list[str]:
        self._model(model_id)
        return [h.name for h in self.headstamps.replace(model_id, names)]

    def add_headstamp(self, model_id: int, name: str) -> list[str]:
        self._model(model_id)
        self.headstamps.add(model_id, name)
        return self.headstamps.names(model_id)

    def rename_headstamp(self, model_id: int, old: str, new: str, *, rename_images: bool = True) -> dict[str, Any]:
        self._model(model_id)
        new = image_store.safe_label(new)
        if old not in self.headstamps.names(model_id):
            raise NotFound(f"Headstamp {old!r} not found")
        self.headstamps.rename(model_id, old, new)
        renamed = 0
        if rename_images:
            for p in image_store.list_images(paths.model_images_dir(model_id)):
                if image_store.parse_label(p.name) == old and image_store.reclassify(p, new):
                    renamed += 1
        return {"headstamps": self.headstamps.names(model_id), "images_renamed": renamed}

    def remove_headstamp(self, model_id: int, name: str) -> list[str]:
        self._model(model_id)
        self.headstamps.remove(model_id, name)
        return self.headstamps.names(model_id)

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def _image_path(self, model_id: int, filename: str) -> Path:
        try:
            name = image_store.safe_filename(filename)
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        p = paths.model_images_dir(model_id) / name
        if not p.is_file():
            raise NotFound(f"Image {filename!r} not found")
        return p

    def list_images(self, model_id: int, *, label: str | None = None, page: int = 1, page_size: int = 100, search: str | None = None) -> dict[str, Any]:
        m = self._model(model_id)
        files = image_store.list_images(paths.model_images_dir(m.id))
        files = image_store.filter_images(files, label, self.headstamps.names(m.id))
        if search:
            s = search.casefold()
            files = [p for p in files if s in p.name.casefold()]
        total = len(files)
        page_size = max(1, min(int(page_size), 500))
        page = max(1, int(page))
        start = (page - 1) * page_size
        items = [image_store.image_info(p) for p in files[start : start + page_size]]
        return {"total": total, "page": page, "page_size": page_size, "items": items, "labels": image_store.class_counts(paths.model_images_dir(m.id))}

    def add_images(self, model_id: int, uploads: list[tuple[str, bytes]], *, label: str | None = None) -> dict[str, Any]:
        m = self._model(model_id)
        if not is_trainable(m):
            raise ServiceError("Images cannot be added to a community or read-only model", 403)
        out_dir = paths.model_images_dir(m.id)
        saved, errors = [], []
        for original_name, data in uploads:
            use_label = (label or "").strip() or (image_store.parse_label(original_name) or "")
            if not use_label:
                errors.append({"file": original_name, "error": "No label: pass label= or name the file {label}__{ticks}.jpg"})
                continue
            try:
                p = image_store.save_image_bytes(data, out_dir, use_label, original_name=original_name)
                saved.append(image_store.image_info(p))
            except ValueError as exc:
                errors.append({"file": original_name, "error": str(exc)})
        labels = {s["label"] for s in saved if s["label"]}
        for lbl in labels:
            self.headstamps.add(m.id, lbl)
        return {"saved": saved, "errors": errors, "image_count": sum(image_store.class_counts(out_dir).values())}

    def image_bytes(self, model_id: int, filename: str, *, thumb: bool = False) -> tuple[bytes, str]:
        p = self._image_path(model_id, filename)
        if thumb:
            return image_store.thumbnail_jpeg(p), "image/jpeg"
        suffix = p.suffix.lower()
        mime = {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".bmp": "image/bmp", ".webp": "image/webp"}.get(suffix, "application/octet-stream")
        return p.read_bytes(), mime

    def reclassify_images(self, model_id: int, filenames: list[str], label: str) -> dict[str, Any]:
        m = self._model(model_id)
        if not is_trainable(m):
            raise ServiceError("Images of a community or read-only model cannot be changed", 403)
        label = image_store.safe_label(label)
        changed, failed = [], []
        for name in filenames:
            try:
                p = self._image_path(m.id, name)
            except ServiceError as exc:
                failed.append({"file": name, "error": str(exc)})
                continue
            dest = image_store.reclassify(p, label)
            if dest is None:
                failed.append({"file": name, "error": "rename failed"})
            else:
                changed.append({"from": name, "to": dest.name})
        if changed:
            self.headstamps.add(m.id, label)
        return {"changed": changed, "failed": failed}

    def delete_images(self, model_id: int, filenames: list[str]) -> dict[str, Any]:
        m = self._model(model_id)
        if not is_trainable(m):
            raise ServiceError("Images of a community or read-only model cannot be changed", 403)
        deleted, failed = [], []
        for name in filenames:
            try:
                p = self._image_path(m.id, name)
            except ServiceError:
                deleted.append(name)  # already gone
                continue
            (deleted if image_store.delete(p) else failed).append(name)
        return {"deleted": deleted, "failed": failed}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def start_training(self, model_id: int, overrides: dict[str, Any] | None = None, *, client_id: int | None = None) -> dict[str, Any]:
        m = self._model(model_id)
        if not is_trainable(m):
            raise ServiceError("Community, read-only and OpenAI-mode models cannot be trained", 403)
        if self.jobs.running_for_model(m.id):
            raise ServiceError("A job is already running for this model", 409)
        img_dir = paths.model_images_dir(m.id)
        counts = image_store.class_counts(img_dir)
        if not counts:
            raise ServiceError("Add at least one labelled training image first")
        if len(counts) < 2:
            raise ServiceError("Training needs images for at least two headstamps")
        if overrides:
            m.training_config = TrainingConfig.from_dict({**m.training_config.to_dict(), **overrides})
        m.training_config.model_name = m.model_mode
        m = self.models.update(m)
        output = paths.model_checkpoint_path(m.id)
        cfg = TrainingConfig.from_dict(m.training_config.to_dict())
        device = str(getattr(self.config, "TRAINING_DEVICE", "auto") or "auto")
        model_id_ = m.id

        def runner(ctx: JobContext) -> dict[str, Any]:
            started = time.time()
            result = run_training(ctx, image_dir=img_dir, output_model=output, config=cfg, device=device)
            self._adopt_training(model_id_, output, result, int(time.time() - started), sum(counts.values()))
            return {k: v for k, v in result.items() if k != "epochs"} | {"epochs": len(result.get("epochs") or [])}

        job = self.jobs.submit("train", runner, model_id=m.id, request={"training_config": cfg.to_dict(), "images": sum(counts.values()), "classes": len(counts)}, lane="heavy", client_id=client_id)
        return job

    def _adopt_training(self, model_id: int, output: Path, result: dict[str, Any], duration: int, image_count: int) -> None:
        m = self.models.get(model_id)
        if m is None or not output.exists():
            return
        m.model_path = str(output)
        m.last_training_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        m.last_training_duration = duration
        m.trained_image_count = image_count
        m.last_val_acc = result.get("best_val_acc")
        env = result.get("env")
        if isinstance(env, dict):
            m.checkpoint_env = CheckpointEnv.from_dict(env)
        self.models.update(m)
        classes = result.get("classes") or []
        if classes:
            self.headstamps.sync_from_classes(m.id, list(classes))
        self.manager.invalidate(m.alias())
        self.sync_serving()

    # ------------------------------------------------------------------
    # Inference / evaluation
    # ------------------------------------------------------------------

    def classify(self, model_id: int, image: Image.Image, *, topk: int = 5) -> dict[str, Any]:
        m = self._model(model_id)
        if not (m.model_path and Path(m.model_path).exists()):
            raise ServiceError("This model has no trained checkpoint yet", 409)
        return self.manager.predict_path(m.model_path, image, image_size=m.training_config.image_size, trusted=False, topk=topk)

    def checkpoint_path(self, model_id: int) -> Path:
        m = self._model(model_id)
        if not (m.model_path and Path(m.model_path).exists()):
            raise NotFound("This model has no trained checkpoint")
        return Path(m.model_path)

    def start_evaluation(self, model_id: int, *, folder: Path | None = None, mapping: dict[str, str] | None = None, source: str = "images", client_id: int | None = None) -> dict[str, Any]:
        m = self._model(model_id)
        if not (m.model_path and Path(m.model_path).exists()):
            raise ServiceError("This model has no trained checkpoint to evaluate", 409)
        target = folder or paths.model_images_dir(m.id)
        if not image_store.list_images(target):
            raise ServiceError("No images to evaluate")
        model_path = m.model_path
        image_size = m.training_config.image_size
        mid = m.id
        cleanup = folder is not None and folder != paths.model_images_dir(m.id)

        def runner(ctx: JobContext) -> dict[str, Any]:
            def classify(img: Image.Image) -> tuple[str, float]:
                pred = self.manager.predict_path(model_path, img, image_size=image_size, trusted=False, topk=1)
                return pred["label"], pred["score"] * 100.0

            def progress(i: int, total: int, name: str) -> None:
                ctx.progress(phase="evaluating", current=i, total=total, file=name)

            try:
                report = evaluator.evaluate_folder(target, classify=classify, mapping=mapping, progress=progress, should_stop=ctx.cancelled)
            finally:
                if cleanup:
                    shutil.rmtree(target, ignore_errors=True)
            report["source"] = source
            report["mapping"] = mapping or {}
            path = evaluator.save_report(mid, report)
            return {"report": path.name, "summary": report["summary"], "confusion": report["confusion"], "stopped": report["stopped"]}

        return self.jobs.submit("evaluate", runner, model_id=m.id, request={"source": source, "mapping": mapping or {}}, lane="light", client_id=client_id)

    def stage_upload_folder(self, model_id: int, uploads: list[tuple[str, bytes]]) -> Path:
        """Write uploaded evaluation images to a scratch folder (labels from filenames)."""
        self._model(model_id)
        folder = paths.model_uploads_dir(model_id) / uuid.uuid4().hex[:8]
        folder.mkdir(parents=True, exist_ok=True)
        for name, data in uploads:
            label = image_store.parse_label(name) or "unknown"
            image_store.save_image_bytes(data, folder, label, original_name=name)
        return folder

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def export_model(self, model_id: int, mode: str | None = None) -> Path:
        m = self._model(model_id)
        export_mode = model_io.ExportMode.parse(mode)
        out = paths.downloads_dir() / f"export_{m.id}_{uuid.uuid4().hex[:6]}.zip"
        model_io.export_model(
            out, m, m.cartridge_name or "Imported", self.headstamps.names(m.id),
            mode=export_mode,
            model_file=m.model_path if m.model_path and Path(m.model_path).exists() else None,
            images_dir=paths.model_images_dir(m.id),
        )
        return out

    def import_archive(self, zip_path: Path, *, name: str | None = None, cartridge: str | None = None, update_existing: bool = True, community_download: bool = False, client_id: int | None = None) -> dict[str, Any]:
        try:
            m = model_io.import_model(
                zip_path, model_repo=self.models, headstamp_repo=self.headstamps,
                cartridge_name_override=cartridge or None, model_name_override=name or None,
                update_existing=update_existing, community_download=community_download, owner_client_id=client_id,
            )
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
            raise ServiceError(f"Import failed: {exc}") from exc
        self.manager.invalidate()
        self.sync_serving()
        return self.model_view(m)

    # ------------------------------------------------------------------
    # Community
    # ------------------------------------------------------------------

    def auth(self):
        with self._auth_lock:
            if self._auth is None:
                from .community.auth import AuthManager

                self._auth = AuthManager()
            return self._auth

    def community_api(self):
        from .community.api import CommunityApi

        return CommunityApi(self.auth())

    def community_status(self) -> dict[str, Any]:
        try:
            status = self.auth().status()
        except Exception as exc:  # noqa: BLE001 - msal missing etc.
            return {"signed_in": False, "error": str(exc), "available": False}
        status["available"] = True
        status["profile"] = self.settings.get("community_profile")
        return status

    def community_login_begin(self) -> dict[str, Any]:
        from .community.auth import AuthError

        try:
            return self.auth().begin()
        except AuthError as exc:
            raise ServiceError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - network / msal failures
            raise ServiceError(f"Could not start sign-in: {exc}") from exc

    def community_login_complete(self, redirect_url: str) -> dict[str, Any]:
        from .community.auth import AuthError

        try:
            self.auth().complete(redirect_url)
        except AuthError as exc:
            raise ServiceError(str(exc)) from exc
        self._refresh_profile()
        return self.community_status()

    def community_logout(self) -> None:
        self.auth().logout()
        self.settings.delete("community_profile")
        self._catalogue_cache = None

    def _refresh_profile(self) -> None:
        try:
            meta = self.community_api().get_user_metadata()
            if meta is not None:
                self.settings.set("community_profile", meta.to_dict())
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not fetch community profile: %s", exc)

    def community_models(self, *, search: str = "", model_type: str = "", cartridge: str = "", refresh: bool = False) -> list[dict[str, Any]]:
        from .community.api import CommunityApiError

        cache_ok = self._catalogue_cache is not None and (time.time() - self._catalogue_cache[0]) < 120 and not (search or model_type or cartridge)
        if cache_ok and not refresh:
            entries = self._catalogue_cache[1]  # type: ignore[index]
        else:
            try:
                entries = [m.to_dict() for m in self.community_api().get_models(search=search, model_type=model_type, cartridge=cartridge)]
            except CommunityApiError as exc:
                raise ServiceError(str(exc), 502) from exc
            except Exception as exc:  # noqa: BLE001
                raise ServiceError(f"Community request failed: {exc}", 502) from exc
            if not (search or model_type or cartridge):
                self._catalogue_cache = (time.time(), entries)
        installed = {m.community_model_uid: m for m in self.models.list() if m.community_model_uid}
        for e in entries:
            local = installed.get(e["model_uid"])
            if local is None:
                e["state"] = "download"
                e["local_model_id"] = None
            else:
                e["local_model_id"] = local.id
                e["state"] = "update" if e["model_version"] > (local.model_version or 0) else "installed"
        return entries

    def community_cartridges(self) -> list[dict[str, Any]]:
        try:
            return [{"id": c.id, "name": c.name} for c in self.community_api().get_available_cartridges()]
        except Exception as exc:  # noqa: BLE001
            raise ServiceError(f"Community request failed: {exc}", 502) from exc

    def community_download(self, model_uid: str, *, update_existing: bool = True, serve: bool | None = None, client_id: int | None = None) -> dict[str, Any]:
        entries = self.community_models()
        info = next((e for e in entries if e["model_uid"] == model_uid), None)
        if info is None:
            raise NotFound("That community model was not found in the catalogue")
        api = self.community_api()
        auto_serve = bool(self.settings.get("auto_serve_downloads", True)) if serve is None else bool(serve)

        def runner(ctx: JobContext) -> dict[str, Any]:
            ctx.progress(phase="requesting", name=info["model_name"])
            payload = api.request_download(model_uid)
            url = payload.get("FullUrl") or payload.get("fullUrl")
            if not url:
                raise RuntimeError("Server did not return a download URL")
            dest = paths.downloads_dir() / f"{model_uid}.zip"

            def progress(done: int, total: int | None) -> None:
                ctx.check_cancelled()
                ctx.progress(phase="downloading", bytes=done, total=total or info.get("download_size") or None)

            api.download_to(url, dest, expected_total=info.get("download_size") or None, progress=progress)
            ctx.progress(phase="importing")
            view = self.import_archive(dest, update_existing=update_existing, community_download=True, client_id=client_id)
            try:
                dest.unlink()
            except OSError:
                pass
            m = self.models.get(view["id"])
            if m is not None:
                if info["model_version"] > (m.model_version or 0):
                    m.model_version = info["model_version"]
                if auto_serve and m.model_path and Path(m.model_path).exists() and not m.serve_enabled:
                    m.serve_enabled = True
                self.models.update(m)
                self.sync_serving()
                view = self.model_view(m)
            self._catalogue_cache = None
            return {"model": view}

        return self.jobs.submit("download", runner, request={"model_uid": model_uid, "name": info["model_name"], "version": info["model_version"]}, lane="light", client_id=client_id)

    def community_share(self, model_id: int, *, description: str, mode: str = "ModelAndImages", feedback_enabled: bool = False, feedback_floor: int = 95, cartridge_id: int = 0, client_id: int | None = None) -> dict[str, Any]:
        m = self._model(model_id)
        if not (m.model_path and Path(m.model_path).exists()) and model_io.ExportMode.parse(mode) != model_io.ExportMode.IMAGES_ONLY:
            raise ServiceError("This model has no checkpoint to share", 409)
        profile = self.settings.get("community_profile") or {}
        if not profile:
            self._refresh_profile()
            profile = self.settings.get("community_profile") or {}
        if not profile.get("can_contribute"):
            raise ServiceError("Your community account does not have the Contribute role", 403)
        api = self.community_api()
        uid = m.community_model_uid or str(uuid.uuid4())
        version = (m.model_version + 1) if m.community_model_uid else (m.model_version or 1)
        export_mode = model_io.ExportMode.parse(mode)
        name, email = self.auth().identity()
        author = profile.get("profile_name") or name or (email.split("@")[0] if email else "")
        headstamps = self.headstamps.names(m.id)
        mid = m.id

        def runner(ctx: JobContext) -> dict[str, Any]:
            import copy
            import json

            model = self.models.get(mid)
            assert model is not None
            share = copy.deepcopy(model)
            share.community_model_uid = uid
            share.model_version = version
            share.feedback_loop_enabled = bool(feedback_enabled)
            share.feedback_loop_confidence_floor = int(feedback_floor) if feedback_enabled else 0
            if feedback_enabled:
                share.feedback_loop_upload_mode = "Instant"
            ctx.progress(phase="packaging")
            zip_path = paths.downloads_dir() / f"{uid}.zip"
            model_io.export_model(
                zip_path, share, share.cartridge_name or "Imported", headstamps, mode=export_mode,
                model_file=share.model_path, images_dir=paths.model_images_dir(mid),
            )
            manifest_path = zip_path.with_suffix(".manifest.json")
            manifest_path.write_text(json.dumps(model_io.read_manifest(zip_path), indent=2), encoding="utf-8")
            info = {
                "ModelName": share.name,
                "PublishDate": datetime.now(timezone.utc).isoformat(),
                "CartridgeId": int(cartridge_id or 0),
                "CartridgeName": share.cartridge_name,
                "HeadstampCount": len(headstamps),
                "ImageCount": sum(image_store.class_counts(paths.model_images_dir(mid)).values()),
                "ModelDescription": description,
                "Author": author,
                "ModelExportMode": {"ModelOnly": 0, "ModelAndImages": 1, "ImagesOnly": 2}.get(export_mode.value, 1),
                "FeedbackLoopEnabled": bool(feedback_enabled),
                "FeedbackLoopConfidenceFloor": int(feedback_floor) if feedback_enabled else 0,
                "ModelVersion": version,
                "CommunityModelUID": uid,
            }

            def progress(sent: int, total: int) -> None:
                ctx.check_cancelled()
                ctx.progress(phase="uploading", bytes=sent, total=total)

            try:
                final_uid = api.share_model(zip_path=zip_path, manifest_path=manifest_path, model_info=info, progress=progress)
            finally:
                for p in (zip_path, manifest_path):
                    try:
                        p.unlink()
                    except OSError:
                        pass
            model = self.models.get(mid)
            if model is not None:
                model.community_model_uid = final_uid or uid
                model.model_version = version
                self.models.update(model)
            self._catalogue_cache = None
            return {"community_model_uid": final_uid or uid, "version": version}

        return self.jobs.submit("share", runner, model_id=m.id, request={"mode": export_mode.value, "description": description}, lane="light", client_id=client_id)

    # ------------------------------------------------------------------
    # Settings (admin)
    # ------------------------------------------------------------------

    SETTING_KEYS = {"allow_remote_clients": True, "auto_serve_downloads": True, "auto_serve_trained": False, "training_device": "auto"}

    def get_settings(self) -> dict[str, Any]:
        out = {k: self.settings.get(k, v) for k, v in self.SETTING_KEYS.items()}
        out["admin_password_set"] = self.admin_password_set()
        out["admin_required"] = self.admin_required()
        out["config"] = {
            "HOST": getattr(self.config, "HOST", None),
            "PORT": getattr(self.config, "PORT", None),
            "API_KEY_set": bool((getattr(self.config, "API_KEY", None) or "").strip()),
            "MODELS": dict(getattr(self.config, "MODELS", {}) or {}),
            "PRELOAD_MODELS": bool(getattr(self.config, "PRELOAD_MODELS", False)),
            "LOG_LEVEL": getattr(self.config, "LOG_LEVEL", "INFO"),
            "DATA_DIR": str(paths.data_root()),
        }
        return out

    def update_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.SETTING_KEYS:
            if key in data:
                value = data[key]
                if key == "training_device":
                    value = str(value) if str(value) in ("auto", "cpu", "cuda") else "auto"
                    self.config.TRAINING_DEVICE = value
                else:
                    value = bool(value)
                self.settings.set(key, value)
        return self.get_settings()

    def _on_job_event(self, event: str, job: dict[str, Any]) -> None:
        if event in ("done", "failed", "cancelled"):
            log.info("Job %s (%s) %s", job["id"], job["kind"], event)
        if event == "done" and job["kind"] == "train" and self.settings.get("auto_serve_trained", False) and job.get("model_id"):
            try:
                m = self.models.get(job["model_id"])
                if m and not m.serve_enabled:
                    self.set_serving(m.id, True)
            except Exception:  # noqa: BLE001
                log.exception("auto-serve after training failed")
