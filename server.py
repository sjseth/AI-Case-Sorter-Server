"""CaseSorter AI Server.

Two jobs in one process:

1. **Serve models** over an OpenAI-compatible API (``POST /v1/chat/completions``,
   ``GET /v1/models``, ``GET /getheadstamps``) -- what the CaseSorter desktop
   client's *AI Config* / OpenAI mode points at. Models come from
   ``config.MODELS`` and from the registry (anything trained, imported or
   downloaded here with serving switched on).

2. **Do the heavy lifting for light-weight clients** bound in *remote* mode:
   create models, receive training images, train, evaluate, moderate images,
   export/import and share, all through ``/api/v1`` (see ``aiserver/api``).
   A browser UI at ``/`` drives the same features for hands-on use.

Run it with:
    python server.py
"""

from __future__ import annotations

import base64
import io
import logging
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

# The embeddable Python distribution in python_e/ doesn't add the script's
# own directory to sys.path, so config/model_manager wouldn't be importable
# without this.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model_manager import ModelManager

from aiserver import __version__, paths
from aiserver.api import admin as admin_api
from aiserver.api import remote as remote_api
from aiserver.api.deps import SESSION_COOKIE
from aiserver.service import AppService


logging.basicConfig(
    level=getattr(logging, (config.LOG_LEVEL or "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aiserver")

if getattr(config, "DATA_DIR", None):
    paths.set_data_root(config.DATA_DIR)


# ---------------------------------------------------------------------------
# App + shared state
# ---------------------------------------------------------------------------

manager = ModelManager(
    aliases=config.MODELS,
    options=getattr(config, "MODEL_OPTIONS", {}),
)
service = AppService(config, manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    stored_device = service.settings.get("training_device")
    if stored_device:
        config.TRAINING_DEVICE = stored_device
    service.start()
    if getattr(config, "PRELOAD_MODELS", False):
        log.info("Preloading %d model(s)...", len(manager.aliases()))
        manager.preload()
    log.info("Data root: %s", paths.data_root())
    try:
        yield
    finally:
        service.stop()


app = FastAPI(
    title="CaseSorter AI Server",
    description=(
        "OpenAI-compatible inference server for trained ConvNeXt models, plus the "
        "remote-client API (/api/v1) that lets light-weight CaseSorter clients create, "
        "train, evaluate and manage models on this machine."
    ),
    version=__version__,
    lifespan=lifespan,
)
app.state.service = service
app.include_router(remote_api.router)
app.include_router(admin_api.router)


# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    client = request.client.host if request.client else "?"
    response = await call_next(request)
    path = request.url.path
    if not path.startswith("/ui/") and not (path.startswith("/api/v1/jobs/") and response.status_code == 200):
        log.info("%s %s %s -> %d", client, request.method, path, response.status_code)
    return response


# ---------------------------------------------------------------------------
# Authentication for the OpenAI endpoints
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> None:
    """``API_KEY`` from config.py, a bound client's token, or an admin session all pass."""
    svc = request.app.state.service
    expected = (getattr(svc.config, "API_KEY", None) or "").strip()
    token = creds.credentials if creds and creds.scheme.lower() == "bearer" else None
    if token:
        if expected and svc.api_key_valid(token):
            return
        if svc.settings.get("allow_remote_clients", True) and svc.clients.authenticate(token):
            return
    if not expected:
        return
    if svc.session_valid(request.cookies.get(SESSION_COOKIE)):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ImageURL(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage = Field(default_factory=Usage)
    # CaseSorter extension: top-1 softmax probability. Lives at the top level
    # so OpenAI-vision clients that don't know about it just ignore it.
    confidence: Optional[float] = None
    topk: Optional[List[dict[str, Any]]] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "casesorter"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[\w./+\-]+);base64,(?P<body>.+)$",
    re.DOTALL,
)


def _decode_data_url(url: str) -> Image.Image:
    text = (url or "").strip()
    match = _DATA_URL_RE.match(text)
    body = match.group("body") if match else text
    try:
        raw = base64.b64decode(body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not open image: {exc}")


def _extract_image(messages: List[ChatMessage]) -> Image.Image:
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if part.type == "image_url" and part.image_url is not None:
                    return _decode_data_url(part.image_url.url)
    raise HTTPException(status_code=400, detail="No image_url content found in messages")


def _manager():
    return app.state.service.manager


def _unknown_model(alias: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=(f"Model {alias!r} is not being served. Served models: {sorted(_manager().aliases())}"),
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/models", response_model=ModelList, dependencies=[Depends(require_api_key)])
def list_models() -> ModelList:
    now = int(time.time())
    return ModelList(data=[ModelInfo(id=name, created=now) for name in _manager().aliases()])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(require_api_key)])
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    mgr = _manager()
    if not mgr.has(req.model):
        raise _unknown_model(req.model)

    image = _extract_image(req.messages)
    try:
        pred = mgr.predict(req.model, image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    log.info("%s -> %s (score=%.3f)", req.model, pred["label"], pred["score"])

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[Choice(message=ResponseMessage(content=pred["label"]))],
        confidence=pred["score"],
        topk=pred.get("topk"),
    )


@app.get("/getheadstamps", response_model=List[str], dependencies=[Depends(require_api_key)])
def get_headstamps(model: Optional[str] = None) -> List[str]:
    mgr = _manager()
    alias = model
    if alias is None:
        configured = mgr.aliases()
        if len(configured) != 1:
            raise HTTPException(
                status_code=400,
                detail=(f"Multiple models served; specify ?model=<alias>. Known models: {sorted(configured)}"),
            )
        alias = configured[0]
    if not mgr.has(alias):
        raise _unknown_model(alias)
    try:
        return mgr.classes(alias)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "models": _manager().aliases(),
        "active_jobs": len(app.state.service.jobs.active()),
    }


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

_WEB_DIR = Path(__file__).resolve().parent / "aiserver" / "web"

if getattr(config, "ENABLE_WEB_UI", True) and _WEB_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_WEB_DIR)), name="ui")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(_WEB_DIR / "index.html"), headers={"Cache-Control": "no-cache"})

else:

    @app.get("/", include_in_schema=False)
    def index_redirect() -> RedirectResponse:
        return RedirectResponse("/docs")


# ---------------------------------------------------------------------------
# Raw-request logging
# ---------------------------------------------------------------------------

def _logging_http_protocol() -> type:
    """Wrap uvicorn's HTTP protocol to dump raw bytes at DEBUG level."""
    try:
        from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol as _Base
    except ImportError:
        from uvicorn.protocols.http.h11_impl import H11Protocol as _Base

    class LoggingHTTPProtocol(_Base):  # type: ignore[misc]
        def data_received(self, data: bytes) -> None:
            if log.isEnabledFor(logging.DEBUG):
                client = "%s:%d" % self.client if self.client else "unknown"
                log.debug("Raw data from %s (%d bytes): %r", client, len(data), data[:2048])
            super().data_received(data)

    return LoggingHTTPProtocol


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn

    log.info("Listening on http://%s:%d  (web UI at /, API docs at /docs)", config.HOST, config.PORT)
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        log_level=(config.LOG_LEVEL or "info").lower(),
        http=_logging_http_protocol(),
    )


if __name__ == "__main__":
    main()
