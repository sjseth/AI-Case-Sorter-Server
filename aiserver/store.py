"""Repositories over the SQLite registry: models, headstamps, clients, jobs, settings."""

from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from .db import Database
from .models import MODEL_MODES, MODEL_TYPES, BoundClient, Headstamp, Model, normalize_upload_mode


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _j(obj: Any) -> str | None:
    return json.dumps(obj) if obj is not None else None


class ModelRepo:
    def __init__(self, db: Database):
        self.db = db

    def list(self) -> list[Model]:
        return [Model.from_row(r) for r in self.db.query("SELECT * FROM models ORDER BY name COLLATE NOCASE")]

    def get(self, model_id: int) -> Model | None:
        row = self.db.one("SELECT * FROM models WHERE id = ?", (int(model_id),))
        return Model.from_row(row) if row else None

    def get_by_name(self, name: str) -> Model | None:
        row = self.db.one("SELECT * FROM models WHERE name = ? COLLATE NOCASE", (name,))
        return Model.from_row(row) if row else None

    def find_by_community_uid(self, uid: str | None) -> Model | None:
        if not uid:
            return None
        row = self.db.one("SELECT * FROM models WHERE community_model_uid = ?", (str(uid),))
        return Model.from_row(row) if row else None

    def find_by_alias(self, alias: str) -> Model | None:
        row = self.db.one(
            "SELECT * FROM models WHERE serve_enabled = 1 AND "
            "(serve_alias = ? COLLATE NOCASE OR (serve_alias IS NULL AND name = ? COLLATE NOCASE))",
            (alias, alias),
        )
        return Model.from_row(row) if row else None

    def unique_name(self, base: str) -> str:
        base = (base or "Model").strip() or "Model"
        existing = {m.name.lower() for m in self.list()}
        if base.lower() not in existing:
            return base
        n = 2
        while f"{base} ({n})".lower() in existing:
            n += 1
        return f"{base} ({n})"

    def _validate(self, m: Model) -> None:
        if not m.name or not m.name.strip():
            raise ValueError("Model name is required")
        if m.model_mode not in MODEL_MODES:
            raise ValueError(f"Unsupported model_mode {m.model_mode!r}")
        if m.model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model_type {m.model_type!r}")
        m.feedback_loop_upload_mode = normalize_upload_mode(
            m.feedback_loop_upload_mode, feedback_enabled=m.feedback_loop_enabled
        )

    def _params(self, m: Model) -> dict[str, Any]:
        return {
            "name": m.name.strip(),
            "cartridge_name": m.cartridge_name or "",
            "model_mode": m.model_mode,
            "model_type": m.model_type,
            "community_model_uid": m.community_model_uid,
            "model_version": int(m.model_version),
            "enable_image_processing": int(bool(m.enable_image_processing)),
            "image_processing_json": _j(m.image_processing.to_dict()),
            "training_config_json": _j(m.training_config.to_dict()),
            "ai_model_config_json": _j(m.ai_model_config.to_dict()),
            "use_primer_mask": int(bool(m.use_primer_mask)),
            "hide_primer": int(bool(m.hide_primer)),
            "primer_mask_size": int(m.primer_mask_size),
            "last_training_date": m.last_training_date,
            "last_training_duration": int(m.last_training_duration or 0),
            "trained_image_count": int(m.trained_image_count or 0),
            "training_confusion_table": m.training_confusion_table,
            "feedback_loop_enabled": int(bool(m.feedback_loop_enabled)),
            "feedback_loop_confidence_floor": int(m.feedback_loop_confidence_floor),
            "feedback_loop_upload_mode": m.feedback_loop_upload_mode,
            "model_path": m.model_path,
            "checkpoint_env_json": _j(m.checkpoint_env.to_dict()),
            "serve_enabled": int(bool(m.serve_enabled)),
            "serve_alias": (m.serve_alias or None),
            "owner_client_id": m.owner_client_id,
            "notes": m.notes or "",
            "last_val_acc": m.last_val_acc,
        }

    def create(self, m: Model) -> Model:
        self._validate(m)
        p = self._params(m)
        cols = ", ".join(p.keys())
        marks = ", ".join("?" for _ in p)
        with self.db.tx() as conn:
            cur = conn.execute(f"INSERT INTO models ({cols}) VALUES ({marks})", tuple(p.values()))
            m.id = int(cur.lastrowid)
        return self.get(m.id)  # type: ignore[return-value]

    def update(self, m: Model) -> Model:
        self._validate(m)
        p = self._params(m)
        sets = ", ".join(f"{k} = ?" for k in p)
        with self.db.tx() as conn:
            conn.execute(
                f"UPDATE models SET {sets}, updated_at = ? WHERE id = ?",
                (*p.values(), utcnow(), int(m.id)),
            )
        return self.get(m.id)  # type: ignore[return-value]

    def delete(self, model_id: int) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM models WHERE id = ?", (int(model_id),))


class HeadstampRepo:
    def __init__(self, db: Database):
        self.db = db

    def list(self, model_id: int) -> list[Headstamp]:
        rows = self.db.query(
            "SELECT * FROM headstamps WHERE model_id = ? ORDER BY name COLLATE NOCASE", (int(model_id),)
        )
        return [Headstamp.from_row(r) for r in rows]

    def names(self, model_id: int) -> list[str]:
        return [h.name for h in self.list(model_id)]

    def add(self, model_id: int, name: str, *, slot: int = 0, parent_name: str | None = None) -> Headstamp:
        name = (name or "").strip()
        if not name:
            raise ValueError("Headstamp name is required")
        with self.db.tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO headstamps(model_id, name, slot, parent_name) VALUES (?, ?, ?, ?)",
                (int(model_id), name, int(slot), parent_name),
            )
        row = self.db.one("SELECT * FROM headstamps WHERE model_id = ? AND name = ?", (int(model_id), name))
        return Headstamp.from_row(row)

    def rename(self, model_id: int, old: str, new: str) -> None:
        new = (new or "").strip()
        if not new:
            raise ValueError("Headstamp name is required")
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE headstamps SET name = ? WHERE model_id = ? AND name = ?", (new, int(model_id), old)
            )

    def set_slot(self, model_id: int, name: str, slot: int) -> None:
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE headstamps SET slot = ? WHERE model_id = ? AND name = ?", (int(slot), int(model_id), name)
            )

    def remove(self, model_id: int, name: str) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM headstamps WHERE model_id = ? AND name = ?", (int(model_id), name))

    def replace(self, model_id: int, names: list[str]) -> list[Headstamp]:
        wanted = []
        seen = set()
        for n in names:
            n = (n or "").strip()
            if n and n.lower() not in seen:
                seen.add(n.lower())
                wanted.append(n)
        with self.db.tx() as conn:
            existing = {r["name"]: r for r in conn.execute("SELECT * FROM headstamps WHERE model_id = ?", (int(model_id),))}
            for name in list(existing):
                if name not in wanted:
                    conn.execute("DELETE FROM headstamps WHERE id = ?", (existing[name]["id"],))
            for name in wanted:
                if name not in existing:
                    conn.execute(
                        "INSERT INTO headstamps(model_id, name, slot) VALUES (?, ?, 0)", (int(model_id), name)
                    )
        return self.list(model_id)

    def sync_from_classes(self, model_id: int, classes: list[str]) -> list[Headstamp]:
        """After training: make sure every class the checkpoint knows is a headstamp."""
        for c in classes:
            if c:
                self.add(model_id, c)
        return self.list(model_id)


class ClientRepo:
    """Light-weight clients bound to this server, authenticated by bearer token."""

    PAIRING_TTL_MINUTES = 15

    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def list(self) -> list[BoundClient]:
        return [BoundClient.from_row(r) for r in self.db.query("SELECT * FROM clients ORDER BY created_at")]

    def get(self, client_id: int) -> BoundClient | None:
        row = self.db.one("SELECT * FROM clients WHERE id = ?", (int(client_id),))
        return BoundClient.from_row(row) if row else None

    def authenticate(self, token: str) -> BoundClient | None:
        if not token:
            return None
        row = self.db.one("SELECT * FROM clients WHERE token_hash = ? AND revoked = 0", (self.hash_token(token),))
        if not row:
            return None
        client = BoundClient.from_row(row)
        with self.db.tx() as conn:
            conn.execute("UPDATE clients SET last_seen_at = ? WHERE id = ?", (utcnow(), client.id))
        return client

    def create(self, name: str, *, client_version: str | None = None) -> tuple[BoundClient, str]:
        token = "csk_" + secrets.token_urlsafe(32)
        with self.db.tx() as conn:
            cur = conn.execute(
                "INSERT INTO clients(name, token_hash, created_at, client_version) VALUES (?, ?, ?, ?)",
                ((name or "client").strip()[:120], self.hash_token(token), utcnow(), client_version),
            )
            cid = int(cur.lastrowid)
        return self.get(cid), token  # type: ignore[return-value]

    def revoke(self, client_id: int) -> None:
        with self.db.tx() as conn:
            conn.execute("UPDATE clients SET revoked = 1 WHERE id = ?", (int(client_id),))

    def delete(self, client_id: int) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM clients WHERE id = ?", (int(client_id),))

    def rename(self, client_id: int, name: str) -> None:
        with self.db.tx() as conn:
            conn.execute("UPDATE clients SET name = ? WHERE id = ?", ((name or "client").strip()[:120], int(client_id)))

    # -- pairing codes ------------------------------------------------------

    def new_pairing_code(self, label: str | None = None) -> dict[str, Any]:
        self.purge_expired_codes()
        code = "-".join(secrets.token_hex(2).upper() for _ in range(2))  # e.g. 3F9A-C21B
        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=self.PAIRING_TTL_MINUTES)
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO pairing_codes(code, created_at, expires_at, label) VALUES (?, ?, ?, ?)",
                (code, utcnow(), expires.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z", label),
            )
        return {"code": code, "expires_at": expires.isoformat(), "label": label}

    def list_pairing_codes(self) -> list[dict[str, Any]]:
        self.purge_expired_codes()
        return [dict(r) for r in self.db.query("SELECT * FROM pairing_codes WHERE used_by IS NULL ORDER BY created_at")]

    def purge_expired_codes(self) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM pairing_codes WHERE expires_at < ?", (utcnow(),))

    def redeem_pairing_code(self, code: str, client_name: str, *, client_version: str | None = None) -> tuple[BoundClient, str] | None:
        code = (code or "").strip().upper().replace(" ", "")
        self.purge_expired_codes()
        row = self.db.one("SELECT * FROM pairing_codes WHERE code = ? AND used_by IS NULL", (code,))
        if not row:
            return None
        client, token = self.create(client_name or row["label"] or "client", client_version=client_version)
        with self.db.tx() as conn:
            conn.execute("UPDATE pairing_codes SET used_by = ? WHERE code = ?", (client.id, code))
        return client, token


class JobRepo:
    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:16]

    def upsert(self, job: dict[str, Any]) -> None:
        with self.db.tx() as conn:
            conn.execute(
                """
                INSERT INTO jobs(id, kind, model_id, status, created_at, started_at, finished_at,
                                 request_json, progress_json, result_json, error, log_path, client_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status, started_at = excluded.started_at,
                    finished_at = excluded.finished_at, progress_json = excluded.progress_json,
                    result_json = excluded.result_json, error = excluded.error, log_path = excluded.log_path
                """,
                (
                    job["id"], job["kind"], job.get("model_id"), job["status"], job["created_at"],
                    job.get("started_at"), job.get("finished_at"), _j(job.get("request")),
                    _j(job.get("progress")), _j(job.get("result")), job.get("error"), job.get("log_path"),
                    job.get("client_id"),
                ),
            )

    @staticmethod
    def _row_to_job(r: Any) -> dict[str, Any]:
        def _p(s):
            try:
                return json.loads(s) if s else None
            except ValueError:
                return None

        return {
            "id": r["id"], "kind": r["kind"], "model_id": r["model_id"], "status": r["status"],
            "created_at": r["created_at"], "started_at": r["started_at"], "finished_at": r["finished_at"],
            "request": _p(r["request_json"]), "progress": _p(r["progress_json"]) or {},
            "result": _p(r["result_json"]), "error": r["error"], "log_path": r["log_path"],
            "client_id": r["client_id"],
        }

    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return self._row_to_job(row) if row else None

    def list(self, *, model_id: int | None = None, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT * FROM jobs"
        conds, params = [], []
        if model_id is not None:
            conds.append("model_id = ?")
            params.append(int(model_id))
        if kind:
            conds.append("kind = ?")
            params.append(kind)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        return [self._row_to_job(r) for r in self.db.query(sql, params)]

    def mark_interrupted(self) -> int:
        """At startup: anything still 'running'/'queued' from a previous process died with it."""
        with self.db.tx() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = 'failed', error = COALESCE(error, 'Server restarted before the job finished'), "
                "finished_at = ? WHERE status IN ('running', 'queued')",
                (utcnow(),),
            )
            return cur.rowcount


class SettingsRepo:
    def __init__(self, db: Database):
        self.db = db

    def get(self, key: str, default: Any = None) -> Any:
        row = self.db.one("SELECT value FROM settings WHERE key = ?", (key,))
        if not row or row["value"] is None:
            return default
        try:
            return json.loads(row["value"])
        except ValueError:
            return row["value"]

    def set(self, key: str, value: Any) -> None:
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )

    def delete(self, key: str) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM settings WHERE key = ?", (key,))

    def all(self) -> dict[str, Any]:
        return {r["key"]: self.get(r["key"]) for r in self.db.query("SELECT key FROM settings")}
