"""OpenAI-compatible HTTP server for trained ConvNeXt checkpoints.

The CaseSorter desktop client treats this process like an OpenAI endpoint:
it POSTs to /v1/chat/completions with the image embedded as a data: URL and
reads the predicted label out of choices[0].message.content. Configuration
(bind host, API key, model aliases) lives in config.py.

Run it with:
    python server.py
or:
    uvicorn server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import base64
import io
import logging
import re
import time
import uuid
from typing import Any, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

import config
from model_manager import ModelManager


logging.basicConfig(
    level=getattr(logging, (config.LOG_LEVEL or "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aiserver")


# ---------------------------------------------------------------------------
# App + shared state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CaseSorter AI Server",
    description=(
        "OpenAI-compatible inference server for trained ConvNeXt models. "
        "Implements POST /v1/chat/completions and GET /v1/models."
    ),
    version="0.1.0",
)

manager = ModelManager(
    aliases=config.MODELS,
    options=getattr(config, "MODEL_OPTIONS", {}),
)

if getattr(config, "PRELOAD_MODELS", False):
    log.info("Preloading %d model(s)...", len(config.MODELS))
    manager.preload()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


def require_api_key(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> None:
    expected = (config.API_KEY or "").strip()
    if not expected:
        return
    if creds is None or creds.scheme.lower() != "bearer" or creds.credentials != expected:
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
    # Accept (and ignore) any extra OpenAI fields the client sends so we stay
    # compatible with future client tweaks.
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

# data:image/png;base64,...   (mime type may include "+", "-", ".", "/")
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/v1/models",
    response_model=ModelList,
    dependencies=[Depends(require_api_key)],
)
def list_models() -> ModelList:
    now = int(time.time())
    return ModelList(data=[ModelInfo(id=name, created=now) for name in manager.aliases()])


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(require_api_key)],
)
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    if not manager.has(req.model):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model {req.model!r} is not configured. "
                f"Known models: {sorted(manager.aliases())}"
            ),
        )

    image = _extract_image(req.messages)
    pred = manager.predict(req.model, image)
    log.info("%s -> %s (score=%.3f)", req.model, pred["label"], pred["score"])

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[Choice(message=ResponseMessage(content=pred["label"]))],
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "models": manager.aliases()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn

    log.info("Listening on http://%s:%d", config.HOST, config.PORT)
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        log_level=(config.LOG_LEVEL or "info").lower(),
    )


if __name__ == "__main__":
    main()
