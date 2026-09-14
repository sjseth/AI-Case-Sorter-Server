# Remote-mode client specification

**Audience:** whoever implements *remote mode* in the desktop client,
[AI-Case-Sorter-Py](https://github.com/sjseth/AI-Case-Sorter-Py).
**Server:** CaseSorter AI Server 0.2.x (this repository).
**Status:** the API below is implemented and exercised by `tests/test_api.py`;
the client side does not exist yet. This document is the contract between the two.

---

## 1. What remote mode is

Today the desktop client does everything on the machine it runs on: it
captures images, stores them, trains ConvNeXt models with PyTorch, evaluates
them and classifies during a sort run. Remote mode moves the heavy part onto
the AI Server:

| Responsibility | Local mode (today) | Remote mode |
|---|---|---|
| Camera, crop, primer mask | client | client (unchanged) |
| Serial board, slots, run loop | client | client (unchanged) |
| Model registry (name, cartridge, mode, headstamps) | client SQLite | **server** (client keeps a mirror row for slots) |
| Training images | `<data>/models/<id>/images` | **server** `data/models/<id>/images` |
| Training | client subprocess (`train_convnext.py`) | **server** job queue |
| Evaluation | client | **server** job |
| Classification during a run | client PyTorch or HTTP | **server** (HTTP) |
| Export / import ZIP | client | **server**, ZIP streamed to/from the client |
| Community sign-in, download, share | client | **server** (the server holds the community login) |
| PyTorch installed | required for local models | **not required** |

A client is **bound** to exactly one server at a time. Binding is a one-time
pairing that yields a bearer token; the token is the only credential the
client needs afterwards, for both the remote API and the OpenAI endpoints.

The server is the source of truth for models and images. The client should
never assume a model still exists, still has a checkpoint, or still has the
headstamps it saw last time; re-read on every page entry.

---

## 2. Transport

- Base URL: what the operator enters, e.g. `http://192.168.1.50:8000`
  (plain HTTP only; the server does not terminate TLS).
- Remote API prefix: `{base}/api/v1`. Interactive schema: `{base}/docs`.
- JSON in and out, UTF-8. Uploads are `multipart/form-data`.
- Every request after binding: `Authorization: Bearer csk_…`.
- Timeouts: 15 s for ordinary calls, 120 s for image batch uploads, no
  timeout (streaming) for ZIP export/import and checkpoint download.
- Errors: non-2xx with body `{"detail": "<human readable>"}`. Map:

| Status | Meaning | Client behaviour |
|---|---|---|
| 400 | Bad request / validation (e.g. "Training needs images for at least two headstamps") | Show `detail` to the operator |
| 401 | Token missing, revoked, or remote clients disabled on the server | Mark the binding broken; offer to pair again |
| 403 | Allowed for admins only, or a community/read-only model can't be changed | Show `detail`; hide the action next time |
| 404 | Model / job / image gone | Refresh the list; drop stale mirror rows |
| 409 | Conflict: a job is already running, or no checkpoint yet | Show `detail`; poll jobs |
| 502 | The server could not reach the community backend | Show `detail`; retry later |
| 5xx | Server bug | Show `detail`, log the response body |

The request-logging middleware on the server logs every call at INFO except
job polls, so polling every 1–2 s is expected and fine.

Before using a server, call `GET /api/v1/server` and check
`version` (≥ `0.2.0`) and that `capabilities` contains `train`, `images`,
`evaluate`. Store `device.device` (`cuda` / `cpu`) to warn the operator
that CPU training is slow.

---

## 3. Binding (pairing)

Operator flow:

1. On the server's web UI (**Clients → Generate pairing code**) the operator
   gets a code like `3F9A-C21B`, valid 15 minutes, single use.
2. In the client: **Settings → AI Server**: enter server URL and the code,
   click **Bind**.
3. Client calls:

```http
POST /api/v1/bind
Content-Type: application/json

{"pairing_code": "3F9A-C21B", "client_name": "Shop PC", "client_version": "1.4.0"}
```

```json
{
  "client_id": 2,
  "client_name": "Shop PC",
  "token": "csk_K_2iMCq0I-FUrWAtDMTpxWrfCy32OUYlB-x85MZdL9g",
  "server": { "...": "same object as GET /api/v1/server" }
}
```

4. Persist `server_url`, `token`, `client_id`, `client_name` (settings
   table; the token is a secret, treat it like the OpenAI API key today).
5. Verify with `GET /api/v1/me` → `{"kind": "client", "client_id": 2, "client_name": "Shop PC"}`.

Errors: `400 Invalid or expired pairing code`, `403 Remote clients are
disabled on this server`. Codes are case-insensitive; strip spaces.

**Unbind** is purely client-side (forget the token). The server keeps the
client row until an admin revokes or deletes it; a revoked token gets 401.

The token also works on `POST /v1/chat/completions`, `GET /v1/models` and
`GET /getheadstamps`, so a client that already has an OpenAI-mode code path
can reuse it with `api_key = token`.

---

## 4. Concepts and data shapes

### 4.1 Model

`GET /api/v1/models/{id}` (and each element of `GET /api/v1/models`):

```json
{
  "id": 1,
  "name": "E2E",
  "cartridge_name": "9mm",
  "model_mode": "convnext_tiny",
  "model_type": "Standard",
  "community_model_uid": null,
  "model_version": 1,
  "enable_image_processing": true,
  "image_processing": {"strategy": "hough", "primer_mode": "hide", "primer_radius": 135,
                       "hough": {"dp": 2.0, "min_dist": 500, "param1": 100, "param2": 60, "min_radius": 150, "max_radius": 250}},
  "training_config": { "...": "see 4.3" },
  "ai_model_config": {"endpoint_url": "", "model": "", "prompt": "", "image_quality": 100, "image_scale": 100},
  "use_primer_mask": false,
  "hide_primer": true,
  "primer_mask_size": 135,
  "last_training_date": "2026-09-13 22:21",
  "last_training_duration": 8,
  "trained_image_count": 33,
  "training_confusion_table": null,
  "feedback_loop_enabled": false,
  "feedback_loop_confidence_floor": 95,
  "feedback_loop_upload_mode": "Manual",
  "model_path": "/…/data/models/1/trainedmodel/1.pth",
  "checkpoint_env": {"torch": "2.14.0+cpu", "torchvision": "0.29.0+cpu", "numpy": "2.4.6"},
  "serve_enabled": true,
  "serve_alias": "e2e-model",
  "owner_client_id": null,
  "notes": "",
  "created_at": "2026-09-13T22:13:48.275Z",
  "updated_at": "2026-09-13T22:21:35.415Z",
  "last_val_acc": 1.0,

  "alias": "e2e-model",
  "trainable": true,
  "has_checkpoint": true,
  "checkpoint_size": 111339291,
  "image_count": 33,
  "class_counts": {"FC": 16, "WIN": 17},
  "headstamps": ["FC", "RP", "WIN"],
  "is_serving": true,
  "active_job": null,
  "mode_label": "ConvNeXt-Tiny"
}
```

The first block is the client's own `Model` dataclass (`sorter/data/models.py`)
field for field, so `Model.from_dict`-style reuse is intended. Notes:

- `model_path` is a **server** path; never use it on the client. Use
  `has_checkpoint` instead.
- `trainable` is false for `CommunityManaged` / `ReadOnly` models and for
  `model_mode == "openai"`. The server refuses image changes and training
  on those with 403; hide those actions.
- `active_job` is a job object (section 4.4) when a training/evaluation is
  queued or running for this model, else `null`.
- `alias` is the name to use on the OpenAI endpoints (`serve_alias` or the
  model name). `is_serving` says whether it is currently on `/v1`.
- `ai_model_config.api_key` is never returned.

### 4.2 Headstamps

The server stores a headstamp **name list** per model. It does **not** store
slot assignments, parents, or slot templates. Those stay in the client's
SQLite exactly as today, keyed off a local mirror row (section 6.2).

Headstamps are created automatically from image labels on upload, from the
classes in a finished checkpoint, and from a ZIP's manifest. Renaming a
headstamp on the server also renames its training images.

### 4.3 Training config

`training_config` is the client's `TrainingConfig` verbatim (all 25 fields,
same defaults: `epochs 10`, `learning_rate 1e-4`, `batch_size 32`,
`image_size 232`, …). `image_directory` and `output_model_path` are ignored
by the server. `allow_gpu: false` forces CPU training. `model_name` is
always overwritten with the model's `model_mode` before a run, as the
client does.

### 4.4 Job

Training, evaluation, community download and share are asynchronous jobs:

```json
{
  "id": "fab2500d0d9342b9",
  "kind": "train",                          // train | evaluate | download | share
  "model_id": 1,
  "status": "running",                      // queued | running | done | failed | cancelled
  "created_at": "2026-09-13T22:21:26.451Z",
  "started_at": "2026-09-13T22:21:26.452Z",
  "finished_at": null,
  "request": {"training_config": {"...": "..."}, "images": 33, "classes": 2},
  "progress": {"...": "kind-specific, see 5.6"},
  "result": null,                           // set when status == done
  "error": null,                            // string when failed / cancelled early
  "log_path": "/…/logs/training-20260913-222126.log",
  "client_id": 2
}
```

Only one training job runs at a time on a server (others queue). Evaluation
and downloads run on a separate lane, so they don't wait behind a training.
Jobs survive a server restart as `failed` with an explanatory `error`.

### 4.5 Image filenames

Same convention as the client: `{label}__{ticks}.jpg`, `ticks` = .NET
`DateTime.Ticks`. The server sanitises labels with the same rules as
`sorter/training/dataset.safe_label`. Uploaded files that already follow the
convention keep their label and ticks, so an image round-trips
client → server → export → client with the same name.

---

## 5. Endpoint reference

All paths below are relative to `{base}/api/v1` and require the bearer token
unless noted. Bodies are JSON unless noted.

### 5.1 Server and identity

| Method | Path | Notes |
|---|---|---|
| `POST` | `/bind` | No auth. Section 3. |
| `GET` | `/me` | `{"kind": "client", "client_id": 2, "client_name": "Shop PC"}` |
| `GET` | `/server` | Version, device, `served_models[]`, `active_jobs[]`, `capabilities[]`, `supported_modes[]`. |

### 5.2 Models

| Method | Path | Body / query | Returns |
|---|---|---|---|
| `GET` | `/models` | | `[Model]` (section 4.1), sorted by name |
| `POST` | `/models` | `{"name", "cartridge_name", "model_mode", "headstamps": [..], "training_config": {..}, "hide_primer", "use_primer_mask", "primer_mask_size", "notes"}` — only `name` required | `201` + Model. A duplicate name gets ` (2)` appended: check the returned `name`. |
| `GET` | `/models/{id}` | | Model |
| `PATCH` | `/models/{id}` | Any of `name`, `cartridge_name`, `model_mode`, `hide_primer`, `use_primer_mask`, `primer_mask_size`, `notes`, `enable_image_processing`, `feedback_loop_*`, `serve_alias`, plus partial `training_config`, `ai_model_config`, `image_processing` objects (merged) | Model |
| `DELETE` | `/models/{id}` | | `204`. `409` if a job is running. Deletes images, checkpoint, reports. |
| `POST` | `/models/{id}/serve` | `{"enabled": true, "alias": "nine"}` (`alias` optional) | Model. `400` without a checkpoint or if the alias collides. |

`model_mode` ∈ `convnext_tiny | convnext_small | convnext_base | convnext_large`
(`openai` is accepted but such a model can't be trained or served).

### 5.3 Headstamps

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/models/{id}/headstamps` | | `["FC", "WIN"]` |
| `PUT` | `/models/{id}/headstamps` | `{"headstamps": ["FC", "WIN"]}` — replaces the list | `["FC", "WIN"]` |
| `POST` | `/models/{id}/headstamps` | `{"name": "RP"}` | full list |
| `POST` | `/models/{id}/headstamps/rename` | `{"old": "WIN", "new": "Winchester", "rename_images": true}` | `{"headstamps": [...], "images_renamed": 17}` |
| `DELETE` | `/models/{id}/headstamps/{name}` | | full list. Images with that label are **not** deleted. |

### 5.4 Images

| Method | Path | Body / query | Returns |
|---|---|---|---|
| `POST` | `/models/{id}/images` | multipart: one or more `files` parts; optional `label` form field | `201` `{"saved": [ImageInfo], "errors": [{"file", "error"}], "image_count": 34}` |
| `GET` | `/models/{id}/images` | `label` (`All` \| a headstamp \| `UNKNOWN HEADSTAMP`), `page` (1-based), `page_size` (≤500), `search` | `{"total", "page", "page_size", "items": [ImageInfo], "labels": {"FC": 16, "WIN": 17}}` |
| `GET` | `/models/{id}/images/{filename}` | `?thumb=true` for a ≤160 px JPEG | image bytes |
| `DELETE` | `/models/{id}/images/{filename}` | | `{"deleted": [..], "failed": [..]}` |
| `POST` | `/models/{id}/images/bulk` | `{"action": "reclassify", "filenames": [..], "label": "FC"}` or `{"action": "delete", "filenames": [..]}` | reclassify: `{"changed": [{"from","to"}], "failed": [{"file","error"}]}`; delete: `{"deleted": [..], "failed": [..]}` |

`ImageInfo` = `{"filename": "FC__200.jpg", "label": "FC", "size": 3386, "modified": "2026-09-13T22:13:48+00:00"}`.

Upload rules:
- A part named `{label}__{ticks}.jpg` needs no `label` field. Any other
  filename needs `label`, or it lands in `errors`.
- Send the **480×480 cropped, primer-masked** frame the client would have
  saved locally (`image_proc.crop_headstamp` + `apply_primer_mask`), JPEG
  quality 95. The server re-encodes as JPEG anyway.
- Batch about 20–50 files per request. Bytes are held in memory server-side
  per request.
- Reclassify after a rename: `filenames` refers to the *current* names; use
  the `changed[].to` names afterwards.

### 5.5 Training

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/models/{id}/train` | `{"training_config": {partial overrides}}` or `{}` | `202` + Job (status `queued`). Overrides are saved on the model first. |
| `GET` | `/models/{id}/jobs` | `?kind=train&limit=20` | `[Job]`, newest first |
| `GET` | `/jobs/{id}` | | Job |
| `GET` | `/jobs/{id}/log` | `?after=<cursor>&limit=500` | `{"lines": [..], "cursor": 123}` — pass `cursor` back as `after` |
| `POST` | `/jobs/{id}/cancel` | | Job. Cancel is graceful: the trainer stops after the current batch; the best checkpoint written so far stays but is **not** adopted. |

Preconditions checked by the server (all `400`/`409` with `detail`): the
model is trainable; no job already running for it; at least one image; at
least two distinct labels.

When a training job reaches `done` the server has already: set
`model_path`, `last_training_date`, `last_training_duration`,
`trained_image_count`, `last_val_acc`, `checkpoint_env`; added every class as
a headstamp; reloaded the served copy. Re-fetch the model.

### 5.6 Job progress shapes

`kind == "train"`, while running:

```json
"progress": {
  "phase": "training",              // starting | training | finishing | cancelled
  "epoch": 3, "total": 10,           // epochs completed / target
  "classes": 7, "class_names": ["FC", "..."], "images": 320, "device": "cuda", "model": "convnext_tiny",
  "batch": {"epoch": 4, "total": 10, "phase": "train", "batch": 12, "batches": 40,
            "loss": 0.42, "acc": 0.87, "elapsed": 31.2},        // null between epochs
  "last_epoch": {"epoch": 3, "total": 10, "train_loss": 0.37, "train_acc": 0.96,
                 "val_loss": 0.29, "val_acc": 1.0, "lr": 5e-5, "swa_active": false,
                 "saved": "new-best", "epoch_seconds": 3.5, "elapsed_seconds": 11.0},
  "epochs": [ ...one entry per completed epoch, same shape as last_epoch... ]
}
```

These are the trainer's `[PROGRESS]` markers, i.e. what
`dialog_training_progress.py` already consumes: `epoch`/`total` drive the
bar; `saved` ∈ `new-best | skipped | no-validation`; `val_*` are `null`
without a validation split. `batch` is new (intra-epoch feedback).

`kind == "train"`, `result` when done:

```json
"result": {"best_val_acc": 1.0, "best_val_loss": 0.29, "classes": ["FC", "WIN"], "images": 33,
           "duration_seconds": 3.5,
           "env": {"torch_version": "2.14.0+cpu", "torchvision_version": "0.29.0+cpu", "numpy_version": "2.4.6"},
           "epochs": 1}
```

`kind == "evaluate"`: `progress = {"phase": "evaluating", "current": 14, "total": 33, "file": "WIN__…jpg"}`;
`result = {"report": "evaluation_20260913_221402.json", "summary": {...}, "confusion": {...}, "stopped": false}`
(shapes in 5.7).

`kind == "download"`: `progress = {"phase": "requesting|downloading|importing", "bytes": n, "total": n}`;
`result = {"model": Model}`.

`kind == "share"`: `progress = {"phase": "packaging|uploading", "bytes": n, "total": n}`;
`result = {"community_model_uid": "…", "version": 2}`.

### 5.7 Evaluation

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/models/{id}/evaluate` | `{"mapping": {"folder label": "model class"}}` (optional) — scores the model's own training images | `202` + Job |
| `POST` | `/models/{id}/evaluate/upload` | multipart `files` (labelled `{label}__x.jpg`), optional `mapping` form field (JSON string) — scores a held-out set; uploads are deleted afterwards | `202` + Job |
| `GET` | `/models/{id}/evaluations` | | `[{"name", "created_at", "total", "total_accuracy", "avg_confidence", "source"}]` newest first |
| `GET` | `/models/{id}/evaluations/{name}` | | full report |

Report:

```json
{
  "results": [{"filename": "FC__200.jpg", "predicted": "FC", "confidence": 96.03,
               "original": "FC", "raw_original": "FC", "has_mapping": false, "match": "match"}],
  "summary": {"total": 33, "with_original": 33, "with_mapping": 0, "total_matches": 33, "known_matches": 0,
              "total_accuracy": 100.0, "known_accuracy": null, "avg_confidence": 92.0, "errors": 0,
              "per_class": [{"cls": "FC", "count": 16, "avg": 96.5, "high": 97.0, "low": 96.0, "mismatches": 0}]},
  "confusion": {"labels": ["FC", "WIN"], "matrix": [[16, 0], [0, 17]]},
  "stopped": false, "source": "training-images", "mapping": {},
  "created_at": "2026-09-13T22:14:02+00:00", "model_id": 1
}
```

`results[]` and `summary` are the client's `evaluator.evaluate_folder` /
`summarize` shapes (confidence 0–100, `match` ∈ `match|mismatch|unknown`),
minus `filepath`. `confusion` is new (rows = actual, columns = predicted).
For `source == "training-images"` thumbnails come from
`GET /models/{id}/images/{filename}?thumb=true`; for uploads there are none.

### 5.8 Classification

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/models/{id}/classify` | multipart `file` (+ optional `topk` form field), **or** JSON `{"image": "<base64 or data URL>", "topk": 5}` | `{"label": "WIN", "confidence": 0.86, "topk": [{"label": "WIN", "score": 0.86}, ...], "model_base": "convnext_tiny", "image_size": 232}` |

`confidence` is 0–1 (multiply by 100 for the client's percent convention).
Works whether or not the model is served. `409` when the model has no
checkpoint. The server resizes to the model's `training_config.image_size`,
so send the 480×480 crop as-is.

Alternative for the run loop: `POST {base}/v1/chat/completions` with
`model = <alias>` and `api_key = <token>` (the OpenAI shape the client already
implements). That requires `is_serving == true`; the remote endpoint above
does not. Both return the same prediction.

### 5.9 Checkpoint, export, import

| Method | Path | Body / query | Returns |
|---|---|---|---|
| `GET` | `/models/{id}/checkpoint` | | the `.pth` file (`application/octet-stream`), 100 MB+ |
| `GET` | `/models/{id}/export` | `?mode=ModelAndImages|ModelOnly|ImagesOnly` | ZIP (`application/zip`, `Content-Disposition: attachment; filename=…zip`) in the client's `model_io` format |
| `POST` | `/models/import` | multipart `file` (ZIP), optional form fields `name`, `cartridge`, `update_existing` (`true`/`false`, default true) | `201` + Model |

Import accepts client exports, this server's exports and legacy Windows-app
archives (same rules as `sorter/data/model_io.import_model`). A ZIP whose
`community_model_uid` is already installed updates that model in place
unless `update_existing=false`.

The checkpoint endpoint exists so a client with PyTorch installed can pull a
server-trained model down and run it locally (offline fallback). The
checkpoint payload is identical to what the client's own trainer writes.

### 5.10 Community (through the server's login)

| Method | Path | Body / query | Returns |
|---|---|---|---|
| `GET` | `/community/status` | | `{"signed_in": true, "name": "…", "email": "…", "profile": {"profile_name", "country", "roles", "can_contribute"}, "available": true}` |
| `GET` | `/community/models` | `?search=&model_type=(ModelAndImages\|ModelOnly\|ImagesOnly)&cartridge=&refresh=true` | `[CatalogueEntry]` |
| `GET` | `/community/cartridges` | | `[{"id", "name"}]` |
| `POST` | `/community/models/{model_uid}/download` | `{"update_existing": true, "serve": null}` | `202` + Job (`kind: download`) |
| `POST` | `/models/{id}/share` | `{"description", "mode": "ModelAndImages", "feedback_enabled": false, "feedback_floor": 95, "cartridge_id": 0}` | `202` + Job (`kind: share`) |

`CatalogueEntry` is the client's `community_api.ModelInfo` in snake_case
(`model_uid, model_name, cartridge_name, cartridge_id, model_version,
model_description, author, publish_date, image_count, headstamp_count,
download_size, export_mode, feedback_loop_enabled,
feedback_loop_confidence_floor`) plus `state` ∈ `download | update | installed`
and `local_model_id`.

Signing in/out is **admin-only** (the server's web UI). A client gets
`{"signed_in": false}` and should tell the operator to sign in on the
server's page. The community feedback loop (below-threshold uploads) is
not proxied by the server in 0.2; keep that client-side as today or leave it
disabled for remote models.

---

## 6. How the client should use it

### 6.1 Settings and mode switch

Add an **AI Server** settings page: server URL, pairing code entry, Bind /
Unbind, connection status (`GET /server` result: version, device, served
models), and a **Mode** choice: *Local* (today) / *Remote*. Remote mode is
only selectable once bound. Store under settings keys, e.g.
`remote.server_url`, `remote.token`, `remote.client_id`, `remote.enabled`.

Sanity check on every app start and on entering the Models page:
`GET /me`. A 401 flips the binding to "needs re-pairing" and shows it in
the status bar; the app keeps working for anything local.

### 6.2 Models page in remote mode

The list comes from `GET /models`. Keep a **mirror row** in the local
`models` table for every remote model the operator has used, with a new
column `remote_model_id` (nullable, unique). The mirror row exists so that
everything keyed off a local model id keeps working untouched: headstamp
slot assignments, parent classifications, slot templates,
`default_model_id`, run images and feedback images folders.

Sync rules:
- On page entry, upsert mirror rows from the server list (name, cartridge,
  mode, type, community uid, version, `training_config`, `checkpoint_env`).
- Reconcile the local `headstamps` table with the server's list: add missing
  names (slot 0), keep existing rows and their slots, remove rows whose
  names are gone server-side (as `dialog_headstamps` already does on rename).
- A remote model has no local `model_path`; `has_checkpoint` from the server
  is the "Trained" column.
- Actions map 1:1: New model → `POST /models`; Edit → `PATCH`; Delete →
  `DELETE` then delete the mirror row; Export → stream `GET /export` to a
  file the operator picks; Import → `POST /models/import`; Evaluate → 5.7;
  Headstamps dialog → 5.3 for names, local DB for slots/parents.
- `trainable == false` hides Images/Train/Delete-images exactly as
  `is_trainable` does today.

### 6.3 Train page

- **Capture and save**: crop as today, then `POST /models/{id}/images` with
  `label` instead of writing to disk. Queue captures and upload in batches
  in a worker so the feed loop isn't blocked; show a pending-upload count.
  If the upload fails, keep the JPEG in a local spool folder
  (`<data>/models/<local id>/upload_spool/`) and retry.
- Per-class counts: `labels` from `GET /models/{id}/images?page_size=1`.
- **Training settings dialog**: unchanged UI, but save to the server with
  `PATCH /models/{id} {"training_config": {...}}`.
- **Start training** → `POST /models/{id}/train`. Open the existing
  progress dialog fed by polling `GET /jobs/{id}` every 1 s and
  `GET /jobs/{id}/log?after=cursor` for the console; map `progress` onto the
  dialog's `training/start`, `training/epoch`, `training/log` handlers.
  Cancel → `POST /jobs/{id}/cancel`. On `done`, re-fetch the model and sync
  headstamps (6.2).
- If the app starts while a job is running on the server (`active_job` on
  the model), offer to reopen the progress dialog.
- Live classify-while-training preview: `POST /models/{id}/classify`.

### 6.4 Images dialog

Replace filesystem calls with 5.4: `GET /images` with `label`, `page`,
`page_size`; thumbnails via `?thumb=true` (cache them by filename +
`modified`); reclassify/delete via `/images/bulk`; preview via the full
image URL. Filter values (`All`, headstamp names, `UNKNOWN HEADSTAMP`) are
the same strings.

### 6.5 Sort run

`classifier.classify_active` gains a branch: active model has
`remote_model_id` → `POST /models/{id}/classify` (multipart JPEG, quality
from the AI config page, default 95) and return `(label, confidence * 100)`.
Reuse the `requests.Session` pooling from `api_client.py`. Confidence floor
logic, feedback captures and slot routing are unchanged. Treat network
errors like an `ApiError` today (stop the run with a message).

Latency: the server's inference path is the same as the standalone server
(single-digit ms on a GPU plus network); budget one round trip per case.

### 6.6 Community page

If remote mode is on, source the catalogue from `GET /community/models`
and start downloads with `POST /community/models/{uid}/download`, polling
the job. Show `GET /community/status`; when `signed_in` is false show
"Sign in on the server at `{base}/`" instead of the MSAL dialog. Share →
`POST /models/{id}/share`. Local-mode community code stays as is.

### 6.7 Offline and failure behaviour

- Server unreachable: every remote action fails fast with the connection
  error; the run page refuses to start a run with a remote model; local
  models keep working.
- A model deleted on the server (404): drop the mirror row and its
  headstamps after confirming with the operator.
- Token revoked (401): status bar warning + re-pair from Settings.
- Training failed: `job.error` holds the last lines of the trainer log; show
  it in the progress dialog as `training/failed` does today.
- Two clients editing the same model: last write wins on the server; the
  client re-reads on page entry, so stale views are short-lived.

---

## 7. Minimal reference client (Python)

```python
import requests

class RemoteServer:
    def __init__(self, base_url: str, token: str, timeout: float = 15.0):
        self.base = base_url.rstrip("/") + "/api/v1"
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"
        self.timeout = timeout

    def _r(self, method, path, **kw):
        kw.setdefault("timeout", self.timeout)
        resp = self.s.request(method, self.base + path, **kw)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail")
            except ValueError:
                detail = resp.text[:200]
            raise RemoteError(resp.status_code, detail)
        return resp

    @classmethod
    def bind(cls, base_url, pairing_code, client_name, client_version):
        r = requests.post(base_url.rstrip("/") + "/api/v1/bind", timeout=15,
                          json={"pairing_code": pairing_code, "client_name": client_name,
                                "client_version": client_version})
        r.raise_for_status()
        data = r.json()
        return cls(base_url, data["token"]), data

    def me(self):                 return self._r("GET", "/me").json()
    def server(self):             return self._r("GET", "/server").json()
    def models(self):             return self._r("GET", "/models").json()
    def model(self, mid):         return self._r("GET", f"/models/{mid}").json()
    def create_model(self, **b):  return self._r("POST", "/models", json=b).json()
    def update_model(self, mid, **b): return self._r("PATCH", f"/models/{mid}", json=b).json()
    def delete_model(self, mid):  self._r("DELETE", f"/models/{mid}")

    def upload_images(self, mid, files, label=None):
        # files: list of (filename, jpeg_bytes)
        parts = [("files", (name, data, "image/jpeg")) for name, data in files]
        form = {"label": label} if label else None
        return self._r("POST", f"/models/{mid}/images", files=parts, data=form, timeout=120).json()

    def list_images(self, mid, label="All", page=1, page_size=100):
        return self._r("GET", f"/models/{mid}/images",
                       params={"label": label, "page": page, "page_size": page_size}).json()

    def thumbnail(self, mid, filename):
        return self._r("GET", f"/models/{mid}/images/{filename}", params={"thumb": "true"}).content

    def bulk_images(self, mid, action, filenames, label=None):
        return self._r("POST", f"/models/{mid}/images/bulk",
                       json={"action": action, "filenames": filenames, "label": label}).json()

    def train(self, mid, **training_config):
        return self._r("POST", f"/models/{mid}/train", json={"training_config": training_config}).json()

    def job(self, jid):           return self._r("GET", f"/jobs/{jid}").json()
    def job_log(self, jid, after=0): return self._r("GET", f"/jobs/{jid}/log", params={"after": after}).json()
    def cancel(self, jid):        return self._r("POST", f"/jobs/{jid}/cancel").json()

    def classify(self, mid, jpeg_bytes):
        r = self._r("POST", f"/models/{mid}/classify",
                    files={"file": ("frame.jpg", jpeg_bytes, "image/jpeg")}, timeout=60).json()
        return r["label"], r["confidence"] * 100.0

    def evaluate(self, mid, mapping=None):
        return self._r("POST", f"/models/{mid}/evaluate", json={"mapping": mapping or {}}).json()

    def export_to(self, mid, dest_path, mode="ModelAndImages"):
        with self._r("GET", f"/models/{mid}/export", params={"mode": mode}, stream=True, timeout=None) as r:
            with open(dest_path, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)

    def import_zip(self, zip_path, name=None):
        with open(zip_path, "rb") as fh:
            return self._r("POST", "/models/import", files={"file": (zip_path.name, fh, "application/zip")},
                           data={"name": name} if name else None, timeout=None).json()


class RemoteError(Exception):
    def __init__(self, status, detail):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail
```

Polling loop for the progress dialog:

```python
cursor = 0
while True:
    job = api.job(job_id)
    lines = api.job_log(job_id, after=cursor)
    cursor = lines["cursor"]
    for line in lines["lines"]:
        bus.post("training/log", line)
    p = job["progress"]
    if p.get("last_epoch"):
        bus.post("training/epoch", p["last_epoch"])
    if job["status"] in ("done", "failed", "cancelled"):
        break
    time.sleep(1.0)
```

---

## 8. Acceptance checklist for the client work

1. Bind with a pairing code; `GET /me` shows the client; unbind forgets the token; a revoked token is reported as "needs re-pairing".
2. Create a model in remote mode; it appears on the server's web UI; the mirror row exists locally with `remote_model_id`.
3. Capture 20 images on the Train page; they appear under the model on the server with the `{label}__{ticks}.jpg` names; per-class counts update.
4. Images dialog lists, filters, reclassifies and deletes server images; thumbnails load.
5. Start training; the progress dialog shows epochs and console lines from the server; cancel works; after completion the model shows Trained and the headstamps include every class.
6. Run a sort with the remote model active: each case is classified by the server; slot routing and confidence floor behave as with a local model.
7. Evaluate from the Models page; summary and mismatch list render from the report JSON.
8. Export a remote model to a ZIP; import that ZIP into a local-mode client; it trains and runs there.
9. Community page in remote mode lists the catalogue when the server is signed in and downloads a model as a job.
10. With the server stopped, local-mode features keep working and remote actions fail with a clear message.
