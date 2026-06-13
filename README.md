# CaseSorter AI Server

A small Python HTTP server that hosts trained **ConvNeXt** image-classification
checkpoints behind an **OpenAI-compatible API** (`POST /v1/chat/completions`,
`GET /v1/models`). It exists so the CaseSorter desktop client — which already
knows how to talk to OpenAI — can be pointed at a local server and run
inference against your own trained models with no client-side changes.

The server is **inference-only**. Training still happens with the standalone
ConvNeXt trainer (see [Checkpoint format](#checkpoint-format) for what it
expects to load).

---

## How it fits together

```
┌────────────────────┐    POST /v1/chat/completions     ┌──────────────────────┐
│ CaseSorter client  │ ─────────────────────────────▶ │  AI Server (this)    │
│ (or any OpenAI     │   { model, messages:[ ...      │                      │
│  SDK / curl / etc) │     image_url data:base64 ]}   │  FastAPI + PyTorch   │
│                    │ ◀───────────────────────────── │  + ConvNeXt ckpt     │
└────────────────────┘   { choices[0].message.content │                      │
                            = "Winchester_45ACP" }   └──────────────────────┘
```

The client sends a standard OpenAI multimodal chat completion request. The
server pulls the first `image_url` out of the messages, decodes the
`data:image/...;base64,...` payload, runs it through the configured model,
and returns the predicted class name as the assistant message content.

---

## Requirements

- **Python 3.10+** (uses `from __future__ import annotations` plus PEP 604 unions).
- **PyTorch 2.9.x** and **torchvision 0.24.x** (installed by `setup.py`).
- **Optional CUDA:** Ampere or newer (compute capability ≥ 8.0) with at
  least 4 GB of VRAM. Lower-spec GPUs and CPU-only machines automatically
  fall back to the CPU wheel.
- The HTTP stack (FastAPI, uvicorn, Pillow) — also installed by `setup.py`.

---

## Quick start

```bash
# 1. Install PyTorch + server deps (auto-detects GPU vs CPU)
python setup.py

# 2. Edit config.py:
#    - set HOST ("127.0.0.1" or "0.0.0.0")
#    - add at least one entry to MODELS
#    - optionally set API_KEY when binding to 0.0.0.0

# 3. Run the server
python server.py
# → INFO: Listening on http://127.0.0.1:8000

# 4. (optional) Sanity check
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models
```

Then in the CaseSorter client, configure the OpenAI connection with:

| Field          | Value                                          |
|----------------|------------------------------------------------|
| Endpoint URL   | `http://127.0.0.1:8000` (no trailing `/v1/...`) |
| Model          | the alias you put in `MODELS` (e.g. `headstamps-v3`) |
| API key        | match `API_KEY` from `config.py`, or anything if unset |

The client appends `/v1/chat/completions` itself.

---

## Configuration (`config.py`)

Every knob lives in `config.py`. There is no CLI or environment-variable
override — keep the file in source control if you want reproducible deploys.

```python
HOST: str = "127.0.0.1"          # "0.0.0.0" to expose on the LAN
PORT: int = 8000

API_KEY: Optional[str] = None    # None/"" = no auth, otherwise Bearer match

MODELS: Dict[str, str] = {
    "headstamps-v3":        "/abs/path/to/headstamps_v3.zip",
    "primers-experimental": "models/primers_swa.pth",  # relative to config.py
}

MODEL_OPTIONS: Dict[str, dict] = {
    "headstamps-v3": {"image_size": 232},   # override 224 default
}

PRELOAD_MODELS: bool = False     # True = warm all models at startup
LOG_LEVEL: str = "INFO"
```

### Notes

- **Bind host.** `127.0.0.1` is reachable only from the same machine — the
  safest default. `0.0.0.0` listens on every interface; pair this with a
  non-empty `API_KEY`.
- **API key.** When `API_KEY` is set, every request must send
  `Authorization: Bearer <key>`. When it's `None` or empty, the server
  accepts anything (including no `Authorization` header at all).
- **Model aliases.** The `model` field in the OpenAI request is looked up
  in `MODELS`. Unknown aliases return HTTP 404 with the list of known names.
  Paths can be absolute or relative to `config.py`.
- **Image size.** Defaults to 224 (the torchvision IMAGENET1K_V1 default
  for all four ConvNeXt sizes). Override via `MODEL_OPTIONS[alias]["image_size"]`
  if you trained at a different size (e.g. 232).
- **Preload.** Defaults to lazy loading — each model is loaded the first
  time it's used and cached afterwards. Flip `PRELOAD_MODELS = True` to pay
  the load cost up front.

---

## API reference

### `POST /v1/chat/completions`

Standard OpenAI chat-completions request. The server only looks at three things:

- `model` — must match a key in `config.MODELS`.
- `messages[*].content[*]` — the **first** content part with
  `type: "image_url"` is used as the image. Everything else (system
  prompts, text parts, additional images) is ignored.
- The image's `url` field — accepts either a `data:image/...;base64,...`
  URL or a bare base64 string.

Any other OpenAI fields (`temperature`, `top_p`, `max_tokens`, `stream`, …)
are accepted and silently ignored.

**Response** is a normal OpenAI completion shape. The predicted class name
is the entirety of `choices[0].message.content`:

```json
{
  "id": "chatcmpl-…",
  "object": "chat.completion",
  "created": 1718291312,
  "model": "headstamps-v3",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Winchester_45ACP"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

Confidence is **not** returned to the client — keeping the body to just the
label means existing OpenAI-vision clients can read the response without
any custom parsing. (`score` is logged server-side at `INFO` level.)

### `GET /v1/models`

Lists configured aliases in the standard OpenAI shape:

```json
{
  "object": "list",
  "data": [
    {"id": "headstamps-v3", "object": "model", "created": 1718291312, "owned_by": "casesorter"}
  ]
}
```

### `GET /healthz`

Unauthenticated quick check — returns `{"status": "ok", "models": [...]}`.

### Interactive docs

FastAPI's auto-generated Swagger UI is at `/docs` and the OpenAPI schema is
at `/openapi.json`.

---

## Checkpoint format

`model_manager.py` knows how to load checkpoint files written by the
`train_convnext.py` trainer (and its SWA-finalised `*_swa` companion). The
file extension doesn't matter -- `.pth`, `.zip`, or anything else all work,
since `torch.load()` reads the content, not the name. A checkpoint is a
`torch.save(...)` blob containing at minimum:

| Key                | Type                | Notes                                        |
|--------------------|---------------------|----------------------------------------------|
| `model_state_dict` | `dict[str, Tensor]` | Standard PyTorch state dict.                 |
| `classes`          | `list[str]`         | Ordered class labels (index → name).         |
| `base`             | `str`               | One of `convnext_tiny/small/base/large`.     |

Optional metadata (`val_acc`, `val_loss`, `dropout`, …) is tolerated and
ignored.

### Classifier-head layouts the loader handles

The trainer has produced three slightly different head shapes over its
history. The loader detects which one a checkpoint uses by looking at the
state-dict keys:

1. **`Sequential(Dropout, Linear)` at `classifier[2]`** — current
   trainer output. Detected via `classifier.2.1.weight`.
2. **Bare `Linear` at `classifier[2]`** — older style. Detected by the
   absence of the keys above.
3. **Extended head with `Linear` at `classifier[3]`** — variant with a
   separate `Dropout` layer. Detected via `classifier.3.weight`.

### SWA quirks

`torch.optim.swa_utils.AveragedModel` wraps the model so its state-dict
keys are prefixed with `module.` and contain an extra `n_averaged` buffer.
The loader strips the prefix and drops `n_averaged` before
`load_state_dict`. SWA-finalised checkpoints (commonly named `*_swa.pth` or
`*_swa.zip`) load via the same code path as regular ones.

### Preprocessing

Identical to the standalone inference path: resize to a square at the
configured `image_size`, `ToTensor()`, ImageNet normalisation
(`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). The image
is converted to RGB before resize, so PNGs with alpha and grayscale
images both work.

---

## Authentication model

The C# OpenAI client always sends an `Authorization: Bearer <ApiKey>`
header. The server's behaviour:

- `API_KEY = None` or empty → accept every request, header or no header.
- `API_KEY = "abc123"` → require `Authorization: Bearer abc123` on
  `/v1/chat/completions` and `/v1/models`. `/healthz` is always open.

There is intentionally no per-model ACL — anyone with the key can hit
every alias.

---

## File map

```
AIServer/
├── README.md            ← this file
├── Claude.md            ← short spec used during initial implementation
├── config.py            ← user-editable settings
├── server.py            ← FastAPI app + entry point
├── model_manager.py     ← checkpoint loader + inference + cache
└── setup.py             ← PyTorch / FastAPI installer (GPU/CPU autodetect)
```

Marker files written by `setup.py` (`.torch_setup_complete`,
`.server_deps_complete`) live in the same folder and are how the
installer decides whether to skip work on subsequent runs. Delete them
to force a fresh install.

---

## Troubleshooting

**`401 Invalid or missing API key`** — `config.API_KEY` is set but the
client isn't sending a matching Bearer token. Either clear `API_KEY` or
update the client's stored key.

**`404 Model 'foo' is not configured`** — the `model` field in the
request doesn't match any key in `config.MODELS`. The error body lists
the known aliases.

**`400 No image_url content found in messages`** — the request body has
no content part with `type: "image_url"`. The official OpenAI vision
shape is required.

**`400 Invalid base64 image` / `Could not open image`** — the data URL
body isn't decodable as a real image. Confirm the client is base64-encoding
raw PNG/JPEG bytes (not a URL or hex).

**Slow first request** — by default the model loads on first use. Set
`PRELOAD_MODELS = True` to load at startup instead.

**`CUDA reported available but test op failed`** in the logs — `torch`
sees a GPU but can't actually run on it (driver mismatch, etc.). The
server falls back to CPU automatically; rerun `setup.py` to
reinstall the right wheel.

---

## Adapting this to a new project

The folder is intentionally self-contained — there are no imports from
the parent CaseSorter codebase. To lift it into its own repo:

1. Copy the folder verbatim.
2. Drop `Claude.md` if you don't want the original brief in the history.
3. Decide whether to keep the GPU/CPU autodetect or simplify to a plain
   `requirements.txt`. The current `setup.py` pins
   `torch==2.9.1` / `torchvision==0.24.1` and uses the cu128 wheel index
   for GPU.
4. If you want a `/v1/embeddings` or `/v1/chat/completions` streaming
   surface later, it slots into `server.py` next to the existing
   endpoints; the manager already returns a `score` you can plumb into a
   richer response shape.
