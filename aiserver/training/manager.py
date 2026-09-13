"""Job queue: training subprocesses and in-process evaluation/import work.

Two lanes, each a single worker thread, so a long training run never blocks
a quick evaluation, while two trainings never fight for the GPU:

* ``heavy``  -- training (one at a time)
* ``light``  -- evaluation, imports, downloads

Jobs are dicts (``JobRepo`` shape) mirrored to SQLite on every state change
so history survives a restart; a restart marks anything still running as
failed. Live jobs additionally keep an in-memory log ring buffer that the
API streams to the UI.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .. import paths
from ..models import TrainingConfig
from ..store import JobRepo, utcnow

log = logging.getLogger(__name__)

_PROGRESS_PREFIX = "[PROGRESS] "
_KILL_GRACE_S = 5.0
MAX_TRAINING_LOGS = 20
LOG_LINES_KEPT = 4000

JobRunner = Callable[["JobContext"], Any]


class JobCancelled(Exception):
    pass


class JobContext:
    """Handed to a runner: progress/log sinks and the cancel flag."""

    def __init__(self, manager: "JobManager", job: dict[str, Any]):
        self.manager = manager
        self.job = job
        self.cancel_event = threading.Event()
        self.proc: subprocess.Popen | None = None

    @property
    def id(self) -> str:
        return self.job["id"]

    @property
    def request(self) -> dict[str, Any]:
        return self.job.get("request") or {}

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled()

    def progress(self, **fields: Any) -> None:
        self.manager._update_progress(self.job, fields)

    def log(self, line: str) -> None:
        self.manager._append_log(self.job, line)


class JobManager:
    def __init__(self, repo: JobRepo, *, on_event: Callable[[str, dict[str, Any]], None] | None = None):
        self.repo = repo
        self.on_event = on_event
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._ctx: dict[str, JobContext] = {}
        self._logs: dict[str, collections.deque[str]] = {}
        self._queues: dict[str, collections.deque[str]] = {"heavy": collections.deque(), "light": collections.deque()}
        self._wake: dict[str, threading.Event] = {k: threading.Event() for k in self._queues}
        self._threads: list[threading.Thread] = []
        self._stopping = False
        interrupted = repo.mark_interrupted()
        if interrupted:
            log.warning("%d job(s) from a previous run were marked failed", interrupted)

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        for lane in self._queues:
            t = threading.Thread(target=self._worker, args=(lane,), name=f"jobs-{lane}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stopping = True
        for jid in list(self._ctx):
            self.cancel(jid)
        for ev in self._wake.values():
            ev.set()

    # -- submission -----------------------------------------------------------

    def submit(
        self,
        kind: str,
        runner: JobRunner,
        *,
        model_id: int | None = None,
        request: dict[str, Any] | None = None,
        lane: str = "light",
        client_id: int | None = None,
    ) -> dict[str, Any]:
        job = {
            "id": JobRepo.new_id(),
            "kind": kind,
            "model_id": model_id,
            "status": "queued",
            "created_at": utcnow(),
            "started_at": None,
            "finished_at": None,
            "request": request or {},
            "progress": {},
            "result": None,
            "error": None,
            "log_path": None,
            "client_id": client_id,
            "_runner": runner,
            "_lane": lane,
        }
        with self._lock:
            self._jobs[job["id"]] = job
            self._ctx[job["id"]] = JobContext(self, job)
            self._logs[job["id"]] = collections.deque(maxlen=LOG_LINES_KEPT)
            self._queues[lane].append(job["id"])
        self._persist(job)
        self._wake[lane].set()
        self._emit("queued", job)
        return self.public(job)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            ctx = self._ctx.get(job_id)
            job = self._jobs.get(job_id)
        if ctx is None or job is None:
            return False
        if job["status"] not in ("queued", "running"):
            return False
        ctx.cancel_event.set()
        if job["status"] == "queued":
            with self._lock:
                try:
                    self._queues[job["_lane"]].remove(job_id)
                except ValueError:
                    pass
            self._finish(job, "cancelled", error="Cancelled before it started")
            return True
        proc = ctx.proc
        if proc is not None and proc.poll() is None:
            _terminate(proc)
        return True

    # -- queries --------------------------------------------------------------

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return self.public(job)
        return self.repo.get(job_id)

    def list(self, *, model_id: int | None = None, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            live = [
                self.public(j)
                for j in self._jobs.values()
                if (model_id is None or j["model_id"] == model_id) and (kind is None or j["kind"] == kind)
            ]
        live_ids = {j["id"] for j in live}
        stored = [j for j in self.repo.list(model_id=model_id, kind=kind, limit=limit) if j["id"] not in live_ids]
        merged = sorted(live + stored, key=lambda j: j["created_at"], reverse=True)
        return merged[:limit]

    def active(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.public(j) for j in self._jobs.values() if j["status"] in ("queued", "running")]

    def running_for_model(self, model_id: int, kind: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            for j in self._jobs.values():
                if j["model_id"] == model_id and j["status"] in ("queued", "running") and (kind is None or j["kind"] == kind):
                    return self.public(j)
        return None

    def log_lines(self, job_id: str, *, after: int = 0, limit: int = 500) -> tuple[list[str], int]:
        """Lines after index ``after`` (absolute line numbers) and the new cursor."""
        with self._lock:
            ring = self._logs.get(job_id)
            job = self._jobs.get(job_id)
        if ring is not None:
            total = int((job or {}).get("_log_total", 0))
            lines = list(ring)
            offset = total - len(lines)
            start = max(after - offset, 0)
            picked = lines[start : start + limit]
            return picked, offset + start + len(picked)
        stored = self.repo.get(job_id)
        if stored and stored.get("log_path") and Path(stored["log_path"]).exists():
            try:
                all_lines = Path(stored["log_path"]).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                return [], after
            picked = all_lines[after : after + limit]
            return picked, after + len(picked)
        return [], after

    @staticmethod
    def public(job: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in job.items() if not k.startswith("_")}

    # -- internals ------------------------------------------------------------

    def _worker(self, lane: str) -> None:
        while not self._stopping:
            with self._lock:
                job_id = self._queues[lane].popleft() if self._queues[lane] else None
            if job_id is None:
                self._wake[lane].wait(timeout=1.0)
                self._wake[lane].clear()
                continue
            with self._lock:
                job = self._jobs.get(job_id)
                ctx = self._ctx.get(job_id)
            if job is None or ctx is None or job["status"] != "queued":
                continue
            job["status"] = "running"
            job["started_at"] = utcnow()
            self._persist(job)
            self._emit("started", job)
            try:
                result = job["_runner"](ctx)
                if ctx.cancelled():
                    self._finish(job, "cancelled", result=result)
                else:
                    self._finish(job, "done", result=result)
            except JobCancelled:
                self._finish(job, "cancelled")
            except Exception as exc:  # noqa: BLE001 - job failures are reported, not raised
                log.exception("Job %s (%s) failed", job_id, job["kind"])
                self._finish(job, "failed", error=str(exc) or exc.__class__.__name__)

    def _finish(self, job: dict[str, Any], status: str, *, result: Any = None, error: str | None = None) -> None:
        with self._lock:
            if job["status"] in ("done", "failed", "cancelled"):
                return
            job["status"] = status
            job["finished_at"] = utcnow()
            if result is not None:
                job["result"] = result
            if error:
                job["error"] = error
        self._persist(job)
        self._emit(status, job)
        # Drop live state after a grace period so late log polls still work.
        threading.Timer(120.0, self._forget, args=(job["id"],)).start()

    def _forget(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
            self._ctx.pop(job_id, None)
            self._logs.pop(job_id, None)

    def _persist(self, job: dict[str, Any]) -> None:
        try:
            self.repo.upsert(self.public(job))
        except Exception:  # noqa: BLE001
            log.exception("Could not persist job %s", job.get("id"))

    def _update_progress(self, job: dict[str, Any], fields: dict[str, Any]) -> None:
        with self._lock:
            job["progress"] = {**(job.get("progress") or {}), **fields}
            job["_dirty"] = True
        now = time.time()
        if now - job.get("_last_persist", 0) > 2.0:
            job["_last_persist"] = now
            self._persist(job)
        self._emit("progress", job)

    def _append_log(self, job: dict[str, Any], line: str) -> None:
        with self._lock:
            ring = self._logs.get(job["id"])
            if ring is not None:
                ring.append(line)
                job["_log_total"] = int(job.get("_log_total", 0)) + 1
            fh = job.get("_log_fh")
        if fh is not None:
            try:
                fh.write(line + "\n")
                fh.flush()
            except (OSError, ValueError):
                pass

    def _emit(self, event: str, job: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event, self.public(job))
        except Exception:  # noqa: BLE001
            log.exception("job event handler failed")


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def build_command(
    image_dir: Path,
    output_model: Path,
    config: TrainingConfig,
    *,
    python: str | None = None,
    script: Path | None = None,
    device: str = "auto",
    no_pretrained: bool = False,
) -> list[str]:
    script = script or Path(__file__).with_name("train_convnext.py")
    cmd = [
        python or sys.executable,
        "-u",
        str(script),
        "--image_dir", str(image_dir),
        "--output_model", str(output_model),
        "--model_name", config.model_name,
        "--epochs", str(config.epochs),
        "--batch_size", str(config.batch_size),
        "--lr", str(config.learning_rate),
        "--weight_decay", str(config.weight_decay),
        "--dropout", str(config.dropout_rate),
        "--val_split", str(config.val_split),
        "--imgsize", str(config.image_size),
        "--max_workers", str(config.max_workers),
        "--stochastic_depth_prob", str(config.stochastic_depth_prob),
        "--focal_gamma", str(config.focal_gamma),
        "--swa_start", str(config.swa_start),
        "--swa_mode", str(config.swa_mode),
        "--swa_acc_threshold", str(config.swa_acc_threshold),
        "--swa_patience", str(config.swa_patience),
        "--swa_min_epoch", str(config.swa_min_epoch),
        "--device", device if config.allow_gpu else "cpu",
    ]
    if config.train_all:
        cmd.append("--trainall")
    if config.freeze_backbone:
        cmd.append("--freeze_backbone")
    if config.use_focal_loss:
        cmd.append("--use_focal_loss")
    if config.use_swa:
        cmd.append("--use_swa")
    if no_pretrained:
        cmd.append("--no_pretrained")
    return cmd


def training_log_path(stamp: str | None = None) -> Path:
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    return paths.logs_dir() / f"training-{stamp}.log"


def _prune_logs(keep: int = MAX_TRAINING_LOGS) -> None:
    try:
        logs = sorted(paths.logs_dir().glob("training-*.log"))
    except OSError:
        return
    for old in logs[:-keep] if len(logs) > keep else []:
        try:
            old.unlink()
        except OSError:
            pass


def run_training(
    ctx: JobContext,
    *,
    image_dir: Path,
    output_model: Path,
    config: TrainingConfig,
    device: str = "auto",
    python: str | None = None,
    script: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Spawn the trainer, stream its output into the job, return the ``done`` payload."""
    cmd = build_command(image_dir, output_model, config, python=python, script=script, device=device)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    paths.logs_dir().mkdir(parents=True, exist_ok=True)
    _prune_logs()
    log_path = training_log_path()
    ctx.job["log_path"] = str(log_path)
    try:
        fh = open(log_path, "a", encoding="utf-8", errors="replace")
        ctx.job["_log_fh"] = fh
    except OSError:
        ctx.job["_log_fh"] = None
    ctx.log(f"# training log -- {datetime.now().isoformat(timespec='seconds')}")
    ctx.log(f"# command: {' '.join(cmd)}")
    ctx.progress(phase="starting", epoch=0, total=config.epochs)

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra_env or {})
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        creationflags=creationflags,
    )
    ctx.proc = proc
    if ctx.cancelled():
        _terminate(proc)

    final: dict[str, Any] | None = None
    epochs: list[dict[str, Any]] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\r\n")
        if line.startswith(_PROGRESS_PREFIX):
            try:
                payload = json.loads(line[len(_PROGRESS_PREFIX):])
            except json.JSONDecodeError:
                ctx.log(line)
                continue
            event = payload.pop("event", "log")
            if event == "start":
                ctx.progress(phase="training", **payload)
                ctx.log(f"[INFO] Training started: {payload.get('images')} images, {payload.get('classes')} classes")
            elif event == "batch":
                ctx.progress(phase="training", batch=payload)
            elif event == "epoch":
                epochs.append(payload)
                ctx.progress(
                    phase="training",
                    epoch=payload.get("epoch"),
                    total=payload.get("total"),
                    last_epoch=payload,
                    epochs=epochs,
                    batch=None,
                )
                ctx.log(
                    "[EPOCH] {epoch}/{total} train_loss={tl:.4f} train_acc={ta:.4f} val_loss={vl} val_acc={va} lr={lr:.2e} saved={saved}".format(
                        epoch=payload.get("epoch"), total=payload.get("total"),
                        tl=payload.get("train_loss") or 0.0, ta=payload.get("train_acc") or 0.0,
                        vl=(f"{payload['val_loss']:.4f}" if payload.get("val_loss") is not None else "-"),
                        va=(f"{payload['val_acc']:.4f}" if payload.get("val_acc") is not None else "-"),
                        lr=payload.get("lr") or 0.0, saved=payload.get("saved"),
                    )
                )
            elif event == "done":
                final = payload
                ctx.progress(phase="finishing", done=payload)
            elif event == "cancelled":
                ctx.progress(phase="cancelled")
            else:
                ctx.progress(**{event: payload})
        else:
            ctx.log(line)

    rc = proc.wait()
    ctx.log(f"# exit code: {rc}" + (" (cancelled)" if ctx.cancelled() else ""))
    fh = ctx.job.pop("_log_fh", None)
    if fh is not None:
        try:
            fh.close()
        except OSError:
            pass
    if ctx.cancelled():
        raise JobCancelled()
    if rc != 0:
        tail = "\n".join(list(ctx.manager._logs.get(ctx.id, []))[-15:])
        raise RuntimeError(f"Trainer exited with code {rc}.\n{tail}")
    if final is None:
        raise RuntimeError("Trainer exited without reporting completion")
    final["epochs"] = epochs
    return final


def _terminate(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            except (ValueError, OSError):
                proc.terminate()
        else:
            os.kill(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()

    def _reaper() -> None:
        try:
            proc.wait(timeout=_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()

    threading.Thread(target=_reaper, daemon=True).start()
