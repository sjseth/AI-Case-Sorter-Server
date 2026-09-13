# CaseSorter AI Server

The server-side half of the [AI Case Sorter](https://github.com/sjseth/AI-Case-Sorter-Py).
It runs on the machine with the GPU (or the most CPU) and does two jobs:

1. **Serve models.** Trained **ConvNeXt** headstamp classifiers are exposed
   behind an **OpenAI-compatible API** (`POST /v1/chat/completions`,
   `GET /v1/models`, `GET /getheadstamps`). The CaseSorter desktop client
   already knows how to talk to OpenAI, so it can be pointed at this server
   with no client-side changes.
2. **Do the heavy lifting for light-weight clients.** A client running in
   *remote* mode **binds** to this server and creates its models here: it
   pushes training images, asks for a training run, watches progress,
   moderates images, evaluates, exports and shares. The client keeps the
   camera and the serial-connected sorting machine; the server keeps the
   models and the CPU/GPU work. Everything the desktop client can do with a
   model locally is available over `/api/v1` (see
   [Remote-client API](#remote-client-api)).

A **browser UI** at `http://<host>:<port>/` drives both: set the admin
password, pair clients, sign in to the community and download models, and
create / train / evaluate / serve models by hand.

---

## How it fits together

```
                       ┌───────────────────────────────────────────────┐
  Serving              │  AI Server                                    │
  ──────               │                                               │
  CaseSorter client ───┼─▶ POST /v1/chat/completions ──▶ ConvNeXt ckpt │
  (OpenAI mode / any   │   { model, messages:[image_url data:base64] } │
   OpenAI SDK / curl)  │   ◀── { choices[0].message.content = "WIN",   │
                       │         confidence: 0.97 }                    │
                       │                                               │
  Remote mode          │  /api/v1  ── models, images, train, evaluate, │
  ───────────          │             export/import, share, jobs        │
  light-weight client ─┼─▶ bind with pairing code → bearer token        │
  (camera + sorter)    │                                               │
                       │  Web UI  ── http://host:port/                 │
  Browser ─────────────┼─▶ dashboard, models, jobs, community,         │
                       │   clients, settings                           │
                       │                                               │
                       │  Registry: data/config/server.db              │
                       │  Models:   data/models/<id>/{images,          │
                       │            trainedmodel,reports}              │
                       └───────────────────────────────────────────────┘
```

Models come from three places and all serve the same way:

- **`config.MODELS`** -- checkpoint files you copied in by hand (the original
  way; still works exactly as before).
- **The registry** -- models created, trained, imported or downloaded
  through the UI / API. Toggle *Serve this model* and they appear on the
  OpenAI endpoints under their alias.
- **The community** -- sign in with the same reloadingrecipes.com account as
  the desktop client, browse the catalogue, download; downloads are served
  automatically (configurable).

---

## Requirements

- **Python 3.10+** (uses `from __future__ import annotations` plus PEP 604 unions).
- **PyTorch 2.9.x** and **torchvision 0.24.x** (installed by `setup.py`).
- **Optional CUDA:** Ampere or newer (compute capability ≥ 8.0) with at
  least 4 GB of VRAM. Lower-spec GPUs and CPU-only machines automatically
  fall back to the CPU wheel.
- The HTTP stack (FastAPI, uvicorn, Pillow, python-multipart) plus
  `requests` and `msal` for the community backend — all installed by
  `setup.py`. Existing installs pick up the new packages automatically on
  the next `startserver` run.

---

## Quick start

> **Disclaimer:** This server is intended to be used with **CaseSorter
> v1.3.2 or newer**. It may work with older client versions, but
> **importing headstamps from the model is not supported in earlier
> versions.**

The walkthrough below assumes you've never used Git before. If you're
already comfortable with Python and Git, skip to
[Quick start (advanced)](#quick-start-advanced).

### 1. Install Git

Download and install Git from the official site. The default options in
the installer are fine — just click through them if you're not sure.

- https://git-scm.com/downloads

### 2. Create a folder and clone the repo

1. Make a new folder on your computer where the server will live, for
   example `C:\AIServer` on Windows.
2. Open that folder, right-click inside it, and choose **Open in
   Terminal** (Windows 11) or **Git Bash Here** (installed by Git for
   Windows). On macOS or Linux, open Terminal and `cd` into the folder.
3. Clone this repository:

   ```
   git clone https://github.com/sjseth/AI-Case-Sorter-Server.git
   ```

4. Step into the folder that was just created:

   ```
   cd AI-Case-Sorter-Server
   ```

### 3. Copy your model file(s) into `models/` (optional)

You can skip steps 3–5 entirely and instead train, import or download
models from the browser UI once the server is running (step 6). If you
already have checkpoint files, this is the quickest way to serve them.

Any model you've trained yourself in the CaseSorter desktop client, or
downloaded from the community, lives under the client's training
directory. On a default Windows install that's:

```
C:\Program Files\SJSeth\AI Brass Sorter\training\models\{modelid}.zip
```

Copy the `.zip` for each model you want to host into the **`models/`**
folder inside the repo you just cloned.

### 4. Find your model id

The `{modelid}` in the path above matches the image folder name for that
model inside the client:

1. Open the CaseSorter client.
2. Go to **Models → Images → Open Folder**.
3. The image folder name shown there is your model id — write it down
   for each model you intend to host.

In the `Training/Models` folder there is a zip file named after each
model id (e.g. `67.zip`, `42.zip`).

### 5. Edit `config.py`

Open `config.py` in any text editor (Notepad is fine) and update two
dictionaries so they point at the model(s) you just copied:

```python
MODELS: Dict[str, str] = {
    "my-model": "models/<modelid>.zip",   # replace <modelid> with your filename
}

MODEL_OPTIONS: Dict[str, dict] = {
    "my-model": {"image_size": 480},      # 480 is the default for community models
}
```

The keys (`"my-model"` above) are aliases — whatever you type here is
what you'll set as the model name inside the client. Save the file when
you're done.

#### Optional: require an API key

If you're exposing the server beyond your own machine (e.g. `HOST` set
to `0.0.0.0` so other PCs on your network can reach it), set an API key
so random callers can't use it. In `config.py`, change:

```python
API_KEY: Optional[str] = None
```

to a quoted string of your choice — anything will do, just keep it
secret:

```python
API_KEY: Optional[str] = "my-secret-key-1234"
```

Then enter the **same value** in the CaseSorter client's API key field.
Leave `API_KEY` as `None` to skip authentication entirely.

### 6. Start the server

- **Windows:** double-click `startserver.bat`
- **macOS / Linux:** run `./startserver.sh`

The first launch installs PyTorch and the rest of the dependencies,
which can take several minutes. Once you see
`INFO: Listening on http://...:8000`, the server is up.

Open **http://localhost:8000/** in a browser. Because the server listens on
all interfaces by default, the first visit asks you to create an admin
password (on a `127.0.0.1`-only server no password is needed). From there
you can create a model, upload training images, train it and switch on
*Serve this model* -- or sign in on the **Community** page and download one.

Then in the CaseSorter client, configure the OpenAI connection with:

| Field        | Value                                                     |
|--------------|-----------------------------------------------------------|
| Endpoint URL | `http://localhost:8000` (or `http://<server-ip>:8000`)    |
| Model        | the alias you put in `MODELS`, or a served registry model's alias (shown on its Overview page) |
| API key      | match `API_KEY` from `config.py`, or anything if unset    |

The client must use `http://` — `https://` is **not** supported.

---

## Quick start (advanced)

If you already have Python and Git and just want the bare steps:

```bash
# 1. Clone and enter the repo
git clone https://github.com/sjseth/AI-Case-Sorter-Server.git
cd AI-Case-Sorter-Server

# 2. Drop your model .zip(s) into ./models/ and edit config.py:
#    - set HOST ("127.0.0.1" or "0.0.0.0")
#    - add at least one entry to MODELS (and MODEL_OPTIONS if not 224px)
#    - optionally set API_KEY when binding to 0.0.0.0

# 3. Launch -- the script installs PyTorch + deps on first run, then
#    runs the server.
./startserver.sh        # macOS / Linux
startserver.bat         # Windows

# 4. (optional) Sanity check
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models
```

The client appends `/v1/chat/completions` itself.

If you'd rather drive the install and launch yourself instead of using
the start script, run the two steps it wraps:

```bash
python setup.py    # one-time: PyTorch + server deps (GPU/CPU autodetect)
python server.py   # start the server
```

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

DATA_DIR: Optional[str] = None   # registry + images + checkpoints; None = "<repo>/data"
ENABLE_WEB_UI: bool = True       # browser UI at /
ADMIN_PASSWORD: Optional[str] = None  # None = set it from the UI on first visit
TRAINING_DEVICE: str = "auto"    # "auto" | "cpu" | "cuda" (also in the UI's Settings)
```

### Notes

- **Bind host.** `127.0.0.1` is reachable only from the same machine — the
  safest default. `0.0.0.0` listens on every interface; pair this with a
  non-empty `API_KEY`.
- **API key.** When `API_KEY` is set, every request must send
  `Authorization: Bearer <key>`. When it's `None` or empty, the server
  accepts anything (including no `Authorization` header at all). To
  enable auth, replace `None` with a quoted string and put the same
  value in the client's API key field:

  ```python
  API_KEY: Optional[str] = "my-secret-key-1234"
  ```
- **Model aliases.** The `model` field in the OpenAI request is looked up
  in `MODELS`. Unknown aliases return HTTP 404 with the list of known names.
  Paths can be absolute or relative to `config.py`.
- **Image size.** Defaults to 224 (the torchvision IMAGENET1K_V1 default
  for all four ConvNeXt sizes). Override via `MODEL_OPTIONS[alias]["image_size"]`
  if you trained at a different size (e.g. 232).
- **Preload.** Defaults to lazy loading — each model is loaded the first
  time it's used and cached afterwards. Flip `PRELOAD_MODELS = True` to pay
  the load cost up front.
- **Data root.** Everything the server creates lives under `DATA_DIR`
  (default `<repo>/data`, ignored by git; the `CASESORTER_SERVER_DATA_DIR`
  environment variable overrides both):

  ```
  data/
  ├── config/server.db        registry: models, headstamps, jobs, clients, settings
  ├── config/msal_cache.bin   community sign-in tokens (copy from a desktop install to reuse a login)
  ├── models/<id>/images/     training images  {label}__{ticks}.jpg
  ├── models/<id>/trainedmodel/<id>.pth
  ├── models/<id>/reports/    evaluation reports (JSON)
  ├── downloads/              community archives while they are being imported
  └── logs/                   training-<stamp>.log
  ```
- **Admin password.** The web UI and its `/api/admin` endpoints are open
  without a password only when `HOST` is `127.0.0.1`/`localhost`. On any
  other bind address the first visit creates one (stored hashed in the
  registry); `ADMIN_PASSWORD` in `config.py` overrides it. Sessions are a
  cookie; the `API_KEY`, when set, also works as an admin bearer token.

---

## Web UI

`http://<host>:<port>/` (disable with `ENABLE_WEB_UI = False`).

| Page | What it does |
|------|--------------|
| **Dashboard** | Device (GPU/CPU), served models, running jobs, model list. |
| **Models** | Create a model (name, cartridge, ConvNeXt size), import a ZIP, or open one. Per model: **Overview** (rename, headstamps, serving alias, export, checkpoint info, delete), **Images** (upload with a label, thumbnail grid, filter by headstamp, multi-select reclassify/delete, preview), **Training** (the desktop client's full training-settings dialog, start/cancel, live epoch + batch progress, log, history), **Evaluate** (score the training images or an uploaded held-out folder; accuracy, per-class table, confusion matrix, mismatches), **Share** (publish to the community). |
| **Jobs** | Every training / evaluation / download / share job with progress and cancel. |
| **Community** | Sign in (see below), browse the catalogue, download or update models. |
| **Clients** | Generate one-time pairing codes, see bound clients, rename / revoke them. |
| **Settings** | Allow remote clients, auto-serve downloads / freshly trained models, training device, admin password, and a read-only view of `config.py`. |

### Community sign-in from a server

The desktop client signs in with a browser and a loopback redirect on
`http://localhost:44300/`. The server uses the same account, app
registration and token cache, driving the same authorization-code flow
itself:

1. **Community → Sign in…** opens the reloadingrecipes.com sign-in page.
2. After signing in, the page redirects to `http://localhost:44300/…`.
   - If the browser runs **on the server machine**, the server's listener
     on port 44300 catches it and you are signed in.
   - If the browser runs **elsewhere**, that page fails to load. Copy its
     full address from the address bar and paste it into the *Complete
     sign-in* box on the Community page.
3. The refresh token is kept in `data/config/msal_cache.bin` (mode 0600).
   A `msal_cache.bin` copied from a desktop install works too.

The developer overrides the client supports are honoured:
`CASESORTER_API_BASE`, `CASESORTER_API_CA_BUNDLE`, `CASESORTER_API_INSECURE`.

---

## Remote-client API

Base path `/api/v1`. Interactive docs (with schemas) at `/docs`.

### Binding

An admin creates a pairing code (**Clients** page, or
`POST /api/admin/clients/pairing-code`). The client redeems it once:

```
POST /api/v1/bind
{"pairing_code": "3F9A-C21B", "client_name": "Shop PC", "client_version": "1.4.0"}

→ {"client_id": 1, "token": "csk_…", "server": {...}}
```

Every later request sends `Authorization: Bearer csk_…`. The same token is
accepted by the OpenAI endpoints, so a bound client needs no separate API
key. Admins can revoke tokens or disable remote clients altogether.
`config.API_KEY` (when set) and an admin session cookie also authenticate,
as admin.

### Endpoints

| Method & path | Purpose |
|---------------|---------|
| `GET /server` | Version, device, served models, active jobs, capabilities. |
| `GET /me` | Who the caller is (`admin` or a client). |
| `GET /models` · `POST /models` · `GET/PATCH/DELETE /models/{id}` | Registry CRUD. `POST` body: `name`, `cartridge_name`, `model_mode` (`convnext_tiny/small/base/large`), optional `headstamps`, `training_config`. The response carries `image_count`, `class_counts`, `headstamps`, `has_checkpoint`, `is_serving`, `active_job`. |
| `POST /models/{id}/serve` `{enabled, alias}` | Put a trained model on / take it off the OpenAI endpoints. |
| `GET/PUT/POST /models/{id}/headstamps`, `POST …/headstamps/rename`, `DELETE …/headstamps/{name}` | Headstamp list; rename also renames the training images. |
| `POST /models/{id}/images` (multipart `files[]`, optional `label`) | Add training images. Files named `{label}__{ticks}.jpg` keep their label and ticks; otherwise `label` is required. Images are re-encoded as JPEG. |
| `GET /models/{id}/images?label=&page=&page_size=&search=` | Paged listing plus per-label counts. `label` may be `All`, a headstamp, or `UNKNOWN HEADSTAMP`. |
| `GET /models/{id}/images/{file}[?thumb=true]` · `DELETE …` | Fetch (or thumbnail) / delete one image. |
| `POST /models/{id}/images/bulk` `{action: reclassify|delete, filenames, label}` | Moderation in bulk. |
| `POST /models/{id}/train` `{training_config: {...}}` | Queue a training run (one at a time). Returns the job. Overrides are saved on the model. |
| `GET /jobs`, `GET /models/{id}/jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/log?after=`, `POST /jobs/{id}/cancel` | Job status, progress (`progress.epoch/total`, `progress.batch`, `progress.epochs[]`), log tail, cancel. |
| `POST /models/{id}/evaluate` `{mapping}` · `POST /models/{id}/evaluate/upload` (multipart) | Evaluate against the training images or an uploaded labelled folder. Result: `summary` (accuracy, per-class), `confusion`, `report` name. |
| `GET /models/{id}/evaluations`, `GET …/evaluations/{name}` | Stored reports. |
| `POST /models/{id}/classify` (multipart `file` or JSON `{image: base64}`) | One-off classification with top-k, without serving the model. |
| `GET /models/{id}/checkpoint` | Download the `.pth` (e.g. for a client that wants to run it locally). |
| `GET /models/{id}/export?mode=ModelAndImages|ModelOnly|ImagesOnly` | ZIP in the desktop client's format (`manifest.json` + `model/` + `images/`). |
| `POST /models/import` (multipart `file`, optional `name`, `cartridge`, `update_existing`) | Import such a ZIP (also legacy Windows-app archives). |
| `GET /community/status`, `GET /community/models?search=&model_type=`, `GET /community/cartridges` | Catalogue, with `state` = `download` / `update` / `installed` per entry. |
| `POST /community/models/{uid}/download` `{update_existing, serve}` | Download + import as a job. |
| `POST /models/{id}/share` `{description, mode, feedback_enabled, feedback_floor}` | Publish to the community (needs the Contribute role). |

Jobs: `status` is `queued → running → done | failed | cancelled`. Training
`progress` mirrors the trainer's markers (`start`, `batch`, `epoch`,
`done`); `result` of a finished training carries `best_val_acc`,
`classes`, `duration_seconds`, `env`.

### What training does

The training worker (`aiserver/training/train_convnext.py`) is the desktop
client's trainer: same flat-folder `{label}__{ticks}` dataset, augmentations,
`Sequential(Dropout, Linear)` head, AdamW + cosine schedule, label
smoothing 0.1, optional focal loss and SWA, and the same checkpoint payload
(`model_state_dict`, `classes`, `base`, `image_size`, `*_version`). So a
checkpoint trained here loads in the desktop client, and vice versa. Server
additions: clean cancel via SIGTERM, `--device`, atomic checkpoint writes,
batch-level progress, and a fallback to random init when the ImageNet
weight download fails on an offline box.

After a run: the checkpoint becomes the model's `model_path`, every class
becomes a headstamp, `last_training_date` / duration / image count /
`checkpoint_env` are recorded, and the served copy (if any) is reloaded.

---

## API reference

### `POST /v1/chat/completions`

Standard OpenAI chat-completions request. The server only looks at three things:

- `model` — a key in `config.MODELS`, or the alias of a registry model with
  serving enabled.
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
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "confidence": 0.973
}
```

`confidence` is a CaseSorter extension: the top-1 softmax probability
of the predicted class, in the range `0.0`–`1.0`. It lives at the top
level of the response — OpenAI-vision clients that don't know about it
ignore it, and clients that do can read `response.confidence` directly.
`choices[0].message.content` is still just the bare label, so the
"AI response" the SDKs expose is unchanged. (The same value is also
logged server-side at `INFO` level.)

`topk` (also a CaseSorter extension) lists the top predictions with
their probabilities.

### `GET /v1/models`

Lists every served alias — `config.MODELS` entries plus registry models
with serving enabled — in the standard OpenAI shape:

```json
{
  "object": "list",
  "data": [
    {"id": "headstamps-v3", "object": "model", "created": 1718291312, "owned_by": "casesorter"}
  ]
}
```

### `GET /getheadstamps`

Returns the ordered class labels (headstamps) baked into a loaded model as a
JSON array of strings, so a client can seed its headstamp list from the
server's source of truth instead of maintaining its own copy.

Query parameter:

- `model` — alias from `config.MODELS`. Optional when only one model is
  configured; required otherwise.

```json
["Federal", "Winchester", "CCI"]
```

The model is loaded on demand (same cache as `/v1/chat/completions`), so the
first call against a lazy model pays the load cost. Auth follows the same
`API_KEY` rule as the `/v1/*` endpoints.

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
  `/v1/chat/completions`, `/v1/models` and `/getheadstamps`. `/healthz` is
  always open.
- A bound client's token (`csk_…`) is accepted everywhere the API key is,
  so a light-weight client needs only the one credential.

There is intentionally no per-model ACL — anyone with the key can hit
every alias.

---

## Development

```
python -m pytest tests        # registry, image store, ZIP import/export, evaluator, API
```

The tests use a temporary data root and a stub inference manager, so they
run without a GPU and without any checkpoint files.

---

## File map

```
AI-Case-Sorter-Server/
├── README.md                 ← this file
├── config.py                 ← user-editable settings
├── server.py                 ← FastAPI app: OpenAI endpoints, UI mount, entry point
├── model_manager.py          ← checkpoint loader + inference cache (config + registry aliases)
├── setup.py                  ← PyTorch / server-deps installer (GPU/CPU autodetect)
├── startserver.bat / .sh     ← setup.py + server.py
├── models/                   ← hand-copied checkpoints referenced by config.MODELS
├── data/                     ← created at runtime: registry, images, checkpoints, logs
├── aiserver/
│   ├── paths.py              ← data-root layout
│   ├── db.py, store.py       ← SQLite schema + repositories (models, headstamps, jobs, clients)
│   ├── models.py             ← dataclasses shared with the desktop client's manifest format
│   ├── image_store.py        ← {label}__{ticks}.jpg storage, thumbnails, reclassify/delete
│   ├── model_io.py           ← ZIP export/import (desktop client + legacy Windows format)
│   ├── evaluator.py          ← folder evaluation, confusion matrix, stored reports
│   ├── service.py            ← everything the API and UI call
│   ├── training/
│   │   ├── train_convnext.py ← the trainer (port of the desktop client's)
│   │   └── manager.py        ← job queue, subprocess runner, progress parsing
│   ├── community/
│   │   ├── auth.py           ← Azure AD B2C sign-in (MSAL auth-code flow)
│   │   └── api.py            ← reloadingrecipes.com catalogue / download / share client
│   ├── api/
│   │   ├── deps.py           ← admin / client authentication
│   │   ├── remote.py         ← /api/v1 (bound clients and the UI)
│   │   └── admin.py          ← /api/admin (UI only)
│   └── web/                  ← the browser UI (index.html, app.js, app.css; no build step)
└── tests/                    ← pytest suite
```

Marker files written by `setup.py` (`.torch_setup_complete`,
`.server_deps_complete`) live in the same folder and are how the
installer decides whether to skip work on subsequent runs. Delete them
to force a fresh install.

---

## Troubleshooting

**Client can't connect / TLS or certificate error** — the CaseSorter
client must point at `http://localhost:<port>` or
`http://<ip-address>:<port>`. `https://` is **not** supported and will
fail to connect.

**`401 Invalid or missing API key`** — `config.API_KEY` is set but the
client isn't sending a matching Bearer token. Either clear `API_KEY` or
update the client's stored key.

**`404 Model 'foo' is not being served`** — the `model` field in the
request doesn't match a key in `config.MODELS` or a served registry
model's alias. The error body lists the served aliases; check *Serve this
model* on the model's Overview page.

**Web UI asks for a password I never set** — the server is bound to a
non-loopback address and no admin password exists yet: the first visit
creates one. To reset a forgotten password, stop the server and delete the
`admin_password_hash` row from `data/config/server.db` (or set
`ADMIN_PASSWORD` in `config.py`).

**Community sign-in redirects to a page that won't load** — expected when
the browser isn't on the server machine. Copy that page's full address
(`http://localhost:44300/?code=…`) into the *Complete sign-in* box.

**Training fails immediately with a weight-download error** — the box is
offline and torchvision can't fetch ImageNet weights. The trainer logs a
warning and continues from random init; accuracy will be lower. Run one
training on a connected machine once (the weights are cached in
`~/.cache/torch`).

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
the parent CaseSorter codebase. The community integration is the only
CaseSorter-specific piece (`aiserver/community/`): drop it, or point it at
your own backend via `CASESORTER_API_BASE`, and the rest (registry, image
store, trainer, evaluator, serving, remote-client API, UI) is generic
image-classification infrastructure.
