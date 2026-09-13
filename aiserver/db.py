"""SQLite connection + schema for the server registry.

One file (``config/server.db``), WAL mode, ``check_same_thread=False`` behind
a process-wide lock so FastAPI's threadpool and the training job thread can
share it. Migrations are a numbered list applied in order and recorded in
``schema_migrations``; add a new entry rather than editing an old one.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import paths

_MIGRATIONS: list[tuple[str, str]] = [
    (
        "0001_initial",
        """
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            cartridge_name TEXT NOT NULL DEFAULT '',
            model_mode TEXT NOT NULL DEFAULT 'convnext_tiny',
            model_type TEXT NOT NULL DEFAULT 'Standard',
            community_model_uid TEXT,
            model_version INTEGER NOT NULL DEFAULT 1,
            enable_image_processing INTEGER NOT NULL DEFAULT 1,
            image_processing_json TEXT,
            training_config_json TEXT,
            ai_model_config_json TEXT,
            use_primer_mask INTEGER NOT NULL DEFAULT 0,
            hide_primer INTEGER NOT NULL DEFAULT 1,
            primer_mask_size INTEGER NOT NULL DEFAULT 135,
            last_training_date TEXT,
            last_training_duration INTEGER NOT NULL DEFAULT 0,
            trained_image_count INTEGER NOT NULL DEFAULT 0,
            training_confusion_table TEXT,
            feedback_loop_enabled INTEGER NOT NULL DEFAULT 0,
            feedback_loop_confidence_floor INTEGER NOT NULL DEFAULT 95,
            feedback_loop_upload_mode TEXT NOT NULL DEFAULT 'Manual',
            model_path TEXT,
            checkpoint_env_json TEXT,
            serve_enabled INTEGER NOT NULL DEFAULT 0,
            serve_alias TEXT,
            owner_client_id INTEGER,
            notes TEXT NOT NULL DEFAULT '',
            last_val_acc REAL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ix_models_name ON models(name COLLATE NOCASE);
        CREATE INDEX IF NOT EXISTS ix_models_uid ON models(community_model_uid);

        CREATE TABLE IF NOT EXISTS headstamps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            slot INTEGER NOT NULL DEFAULT 0,
            parent_name TEXT,
            UNIQUE(model_id, name)
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            model_id INTEGER,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            request_json TEXT,
            progress_json TEXT,
            result_json TEXT,
            error TEXT,
            log_path TEXT,
            client_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS ix_jobs_model ON jobs(model_id, created_at);

        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            last_seen_at TEXT,
            revoked INTEGER NOT NULL DEFAULT 0,
            client_version TEXT
        );

        CREATE TABLE IF NOT EXISTS pairing_codes (
            code TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            label TEXT,
            used_by INTEGER
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """,
    ),
]


class Database:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else paths.db_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if str(self.path) != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self.migrate()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Serialized transaction; nested use re-enters the same lock."""
        with self._lock:
            conn = self._conn
            if conn.in_transaction:
                yield conn
                return
            conn.execute("BEGIN")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def query(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def one(self, sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def migrate(self) -> None:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            done = {r["name"] for r in self._conn.execute("SELECT name FROM schema_migrations")}
            for name, sql in _MIGRATIONS:
                if name in done:
                    continue
                with self.tx() as conn:
                    for stmt in _split_statements(sql):
                        conn.execute(stmt)
                    conn.execute(
                        "INSERT INTO schema_migrations(name, applied_at) VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                        (name,),
                    )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _split_statements(sql: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buf.append(line)
        if stripped.endswith(";"):
            out.append("\n".join(buf).rstrip().rstrip(";"))
            buf = []
    if buf:
        out.append("\n".join(buf))
    return out
