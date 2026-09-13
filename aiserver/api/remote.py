"""``/api/v1`` -- the API a light-weight CaseSorter client binds to.

Everything the desktop client does locally for a model (create it, add and
moderate training images, train, evaluate, export/import, share) is exposed
here so a client in *remote* mode can hand the heavy lifting to this server.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from ..service import AppService, ServiceError
from .deps import Principal, get_service, require_principal

router = APIRouter(prefix="/api/v1", tags=["remote"])


def _err(exc: ServiceError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=str(exc))


# ----- schemas -----------------------------------------------------------------

class BindRequest(BaseModel):
    pairing_code: str
    client_name: str = "CaseSorter client"
    client_version: Optional[str] = None


class ModelCreate(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    name: str
    cartridge_name: str = ""
    model_mode: str = "convnext_tiny"
    headstamps: List[str] = Field(default_factory=list)
    training_config: dict[str, Any] = Field(default_factory=dict)
    ai_model_config: dict[str, Any] = Field(default_factory=dict)
    hide_primer: bool = True
    use_primer_mask: bool = False
    primer_mask_size: int = 135
    notes: str = ""


class ModelUpdate(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())


class HeadstampList(BaseModel):
    headstamps: List[str]


class HeadstampRename(BaseModel):
    old: str
    new: str
    rename_images: bool = True


class ImageBulk(BaseModel):
    action: str  # reclassify | delete
    filenames: List[str]
    label: Optional[str] = None


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    training_config: dict[str, Any] = Field(default_factory=dict)


class EvaluateRequest(BaseModel):
    mapping: dict[str, str] = Field(default_factory=dict)


class ClassifyJSON(BaseModel):
    image: str  # data URL or bare base64
    topk: int = 5


class ServeRequest(BaseModel):
    enabled: bool
    alias: Optional[str] = None


class ShareRequest(BaseModel):
    description: str = ""
    mode: str = "ModelAndImages"
    feedback_enabled: bool = False
    feedback_floor: int = 95
    cartridge_id: int = 0


class DownloadRequest(BaseModel):
    update_existing: bool = True
    serve: Optional[bool] = None


# ----- binding ---------------------------------------------------------------

@router.post("/bind")
def bind_client(req: BindRequest, svc: AppService = Depends(get_service)) -> dict[str, Any]:
    if not svc.settings.get("allow_remote_clients", True):
        raise HTTPException(status_code=403, detail="Remote clients are disabled on this server")
    result = svc.clients.redeem_pairing_code(req.pairing_code, req.client_name, client_version=req.client_version)
    if result is None:
        raise HTTPException(status_code=400, detail="Invalid or expired pairing code")
    client, token = result
    return {"client_id": client.id, "client_name": client.name, "token": token, "server": svc.server_info()}


@router.get("/me")
def whoami(principal: Principal = Depends(require_principal)) -> dict[str, Any]:
    return {"kind": principal.kind, "client_id": principal.client_id, "client_name": principal.client_name}


@router.get("/server")
def server_info(svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> dict[str, Any]:
    return svc.server_info()


# ----- models ----------------------------------------------------------------

@router.get("/models")
def list_models(svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> list[dict[str, Any]]:
    return svc.list_models()


@router.post("/models", status_code=201)
def create_model(req: ModelCreate, svc: AppService = Depends(get_service), p: Principal = Depends(require_principal)) -> dict[str, Any]:
    try:
        return svc.create_model(req.model_dump(), client_id=p.client_id)
    except ServiceError as exc:
        raise _err(exc)


@router.get("/models/{model_id}")
def get_model(model_id: int, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> dict[str, Any]:
    try:
        return svc.get_model(model_id)
    except ServiceError as exc:
        raise _err(exc)


@router.patch("/models/{model_id}")
def update_model(model_id: int, req: ModelUpdate, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> dict[str, Any]:
    try:
        return svc.update_model(model_id, req.model_dump())
    except ServiceError as exc:
        raise _err(exc)


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: int, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> Response:
    try:
        svc.delete_model(model_id)
    except ServiceError as exc:
        raise _err(exc)
    return Response(status_code=204)


@router.post("/models/{model_id}/serve")
def set_serving(model_id: int, req: ServeRequest, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> dict[str, Any]:
    try:
        return svc.model_view(svc.set_serving(model_id, req.enabled, req.alias))
    except ServiceError as exc:
        raise _err(exc)


# ----- headstamps ------------------------------------------------------------

@router.get("/models/{model_id}/headstamps")
def get_headstamps(model_id: int, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> list[str]:
    try:
        return svc.get_model(model_id)["headstamps"]
    except ServiceError as exc:
        raise _err(exc)


@router.put("/models/{model_id}/headstamps")
def put_headstamps(model_id: int, req: HeadstampList, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> list[str]:
    try:
        return svc.set_headstamps(model_id, req.headstamps)
    except ServiceError as exc:
        raise _err(exc)


@router.post("/models/{model_id}/headstamps")
def add_headstamp(model_id: int, req: dict[str, str], svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> list[str]:
    try:
        return svc.add_headstamp(model_id, req.get("name", ""))
    except (ServiceError, ValueError) as exc:
        raise HTTPException(status_code=getattr(exc, "status", 400), detail=str(exc))


@router.post("/models/{model_id}/headstamps/rename")
def rename_headstamp(model_id: int, req: HeadstampRename, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> dict[str, Any]:
    try:
        return svc.rename_headstamp(model_id, req.old, req.new, rename_images=req.rename_images)
    except ServiceError as exc:
        raise _err(exc)


@router.delete("/models/{model_id}/headstamps/{name}")
def remove_headstamp(model_id: int, name: str, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> list[str]:
    try:
        return svc.remove_headstamp(model_id, name)
    except ServiceError as exc:
        raise _err(exc)


# ----- images ----------------------------------------------------------------

@router.get("/models/{model_id}/images")
def list_images(
    model_id: int,
    label: Optional[str] = None,
    page: int = 1,
    page_size: int = Query(100, le=500),
    search: Optional[str] = None,
    svc: AppService = Depends(get_service),
    _: Principal = Depends(require_principal),
) -> dict[str, Any]:
    try:
        return svc.list_images(model_id, label=label, page=page, page_size=page_size, search=search)
    except ServiceError as exc:
        raise _err(exc)


@router.post("/models/{model_id}/images", status_code=201)
async def upload_images(
    model_id: int,
    files: List[UploadFile] = File(...),
    label: Optional[str] = Form(None),
    svc: AppService = Depends(get_service),
    _: Principal = Depends(require_principal),
) -> dict[str, Any]:
    uploads = [(f.filename or "upload.jpg", await f.read()) for f in files]
    try:
        return svc.add_images(model_id, uploads, label=label)
    except ServiceError as exc:
        raise _err(exc)


@router.get("/models/{model_id}/images/{filename}")
def get_image(model_id: int, filename: str, thumb: bool = False, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> Response:
    try:
        data, mime = svc.image_bytes(model_id, filename, thumb=thumb)
    except ServiceError as exc:
        raise _err(exc)
    return Response(content=data, media_type=mime, headers={"Cache-Control": "private, max-age=3600"})


@router.delete("/models/{model_id}/images/{filename}")
def delete_image(model_id: int, filename: str, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> dict[str, Any]:
    try:
        return svc.delete_images(model_id, [filename])
    except ServiceError as exc:
        raise _err(exc)


@router.post("/models/{model_id}/images/bulk")
def bulk_images(model_id: int, req: ImageBulk, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> dict[str, Any]:
    try:
        if req.action == "reclassify":
            if not req.label:
                raise HTTPException(status_code=400, detail="label is required for reclassify")
            return svc.reclassify_images(model_id, req.filenames, req.label)
        if req.action == "delete":
            return svc.delete_images(model_id, req.filenames)
    except ServiceError as exc:
        raise _err(exc)
    raise HTTPException(status_code=400, detail="action must be 'reclassify' or 'delete'")


# ----- training / jobs ---------------------------------------------------------

@router.post("/models/{model_id}/train", status_code=202)
def train(model_id: int, req: TrainRequest | None = None, svc: AppService = Depends(get_service), p: Principal = Depends(require_principal)) -> dict[str, Any]:
    overrides = (req.training_config if req else None) or (req.model_dump(exclude={"training_config"}) if req else None)
    try:
        return svc.start_training(model_id, overrides, client_id=p.client_id)
    except ServiceError as exc:
        raise _err(exc)


@router.get("/models/{model_id}/jobs")
def model_jobs(model_id: int, kind: Optional[str] = None, limit: int = 20, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> list[dict[str, Any]]:
    return svc.jobs.list(model_id=model_id, kind=kind, limit=limit)


@router.get("/jobs")
def list_jobs(kind: Optional[str] = None, limit: int = 50, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> list[dict[str, Any]]:
    return svc.jobs.list(kind=kind, limit=limit)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> dict[str, Any]:
    job = svc.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/log")
def job_log(job_id: str, after: int = 0, limit: int = 500, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> dict[str, Any]:
    if svc.jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    lines, cursor = svc.jobs.log_lines(job_id, after=after, limit=limit)
    return {"lines": lines, "cursor": cursor}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> dict[str, Any]:
    if not svc.jobs.cancel(job_id):
        job = svc.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    return svc.jobs.get(job_id) or {"id": job_id, "status": "cancelled"}


# ----- evaluation --------------------------------------------------------------

@router.post("/models/{model_id}/evaluate", status_code=202)
def evaluate_own_images(model_id: int, req: EvaluateRequest | None = None, svc: AppService = Depends(get_service), p: Principal = Depends(require_principal)) -> dict[str, Any]:
    try:
        return svc.start_evaluation(model_id, mapping=(req.mapping if req else None), source="training-images", client_id=p.client_id)
    except ServiceError as exc:
        raise _err(exc)


@router.post("/models/{model_id}/evaluate/upload", status_code=202)
async def evaluate_uploaded(
    model_id: int,
    files: List[UploadFile] = File(...),
    mapping: Optional[str] = Form(None),
    svc: AppService = Depends(get_service),
    p: Principal = Depends(require_principal),
) -> dict[str, Any]:
    import json

    uploads = [(f.filename or "upload.jpg", await f.read()) for f in files]
    try:
        folder = svc.stage_upload_folder(model_id, uploads)
        map_dict = json.loads(mapping) if mapping else None
        return svc.start_evaluation(model_id, folder=folder, mapping=map_dict, source="upload", client_id=p.client_id)
    except ServiceError as exc:
        raise _err(exc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Bad request: {exc}")


@router.get("/models/{model_id}/evaluations")
def list_evaluations(model_id: int, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> list[dict[str, Any]]:
    from .. import evaluator

    try:
        svc.get_model(model_id)
    except ServiceError as exc:
        raise _err(exc)
    return evaluator.list_reports(model_id)


@router.get("/models/{model_id}/evaluations/{name}")
def get_evaluation(model_id: int, name: str, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> dict[str, Any]:
    from .. import evaluator

    report = evaluator.load_report(model_id, name)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.get("/models/{model_id}/evaluations/{name}/image/{filename}")
def evaluation_image(model_id: int, name: str, filename: str, thumb: bool = True, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> Response:
    """Images referenced by a training-image evaluation (uploads are deleted after scoring)."""
    try:
        data, mime = svc.image_bytes(model_id, filename, thumb=thumb)
    except ServiceError as exc:
        raise _err(exc)
    return Response(content=data, media_type=mime)


# ----- classify / checkpoint ---------------------------------------------------

def _decode_image(payload: str) -> Image.Image:
    text = (payload or "").strip()
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    try:
        raw = base64.b64decode(text)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")


@router.post("/models/{model_id}/classify")
async def classify(
    model_id: int,
    request: Request,
    file: Optional[UploadFile] = File(None),
    topk: int = Form(5),
    svc: AppService = Depends(get_service),
    _: Principal = Depends(require_principal),
) -> dict[str, Any]:
    image: Image.Image
    if file is not None:
        try:
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not open image: {exc}")
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Send a multipart 'file' or JSON {image: <base64>}")
        parsed = ClassifyJSON(**body)
        image = _decode_image(parsed.image)
        topk = parsed.topk
    try:
        pred = svc.classify(model_id, image, topk=topk)
    except ServiceError as exc:
        raise _err(exc)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"label": pred["label"], "confidence": pred["score"], "topk": pred["topk"], "model_base": pred["model_base"], "image_size": pred["image_size"]}


@router.get("/models/{model_id}/checkpoint")
def download_checkpoint(model_id: int, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> FileResponse:
    try:
        path = svc.checkpoint_path(model_id)
    except ServiceError as exc:
        raise _err(exc)
    return FileResponse(str(path), media_type="application/octet-stream", filename=f"{model_id}.pth")


# ----- export / import ---------------------------------------------------------

@router.get("/models/{model_id}/export")
def export_model(model_id: int, mode: str = "ModelAndImages", svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> FileResponse:
    try:
        path = svc.export_model(model_id, mode)
        name = svc.get_model(model_id)["name"]
    except (ServiceError, ValueError) as exc:
        raise HTTPException(status_code=getattr(exc, "status", 400), detail=str(exc))
    from starlette.background import BackgroundTask

    def _cleanup() -> None:
        try:
            Path(path).unlink()
        except OSError:
            pass

    return FileResponse(str(path), media_type="application/zip", filename=f"{name}.zip".replace(" ", "_"), background=BackgroundTask(_cleanup))


@router.post("/models/import", status_code=201)
async def import_model(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    cartridge: Optional[str] = Form(None),
    update_existing: bool = Form(True),
    svc: AppService = Depends(get_service),
    p: Principal = Depends(require_principal),
) -> dict[str, Any]:
    from .. import paths

    tmp = paths.downloads_dir() / f"import_{Path(file.filename or 'model.zip').stem[:40]}_{id(file)}.zip"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as fh:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    try:
        return svc.import_archive(tmp, name=name, cartridge=cartridge, update_existing=update_existing, client_id=p.client_id)
    except ServiceError as exc:
        raise _err(exc)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


# ----- community (read + download; sign-in is admin-only) ----------------------

@router.get("/community/status")
def community_status(svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> dict[str, Any]:
    status = svc.community_status()
    status.pop("auth_url", None)
    return status


@router.get("/community/models")
def community_models(search: str = "", model_type: str = "", cartridge: str = "", refresh: bool = False, svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> list[dict[str, Any]]:
    try:
        return svc.community_models(search=search, model_type=model_type, cartridge=cartridge, refresh=refresh)
    except ServiceError as exc:
        raise _err(exc)


@router.get("/community/cartridges")
def community_cartridges(svc: AppService = Depends(get_service), _: Principal = Depends(require_principal)) -> list[dict[str, Any]]:
    try:
        return svc.community_cartridges()
    except ServiceError as exc:
        raise _err(exc)


@router.post("/community/models/{model_uid}/download", status_code=202)
def community_download(model_uid: str, req: DownloadRequest | None = None, svc: AppService = Depends(get_service), p: Principal = Depends(require_principal)) -> dict[str, Any]:
    req = req or DownloadRequest()
    try:
        return svc.community_download(model_uid, update_existing=req.update_existing, serve=req.serve, client_id=p.client_id)
    except ServiceError as exc:
        raise _err(exc)


@router.post("/models/{model_id}/share", status_code=202)
def community_share(model_id: int, req: ShareRequest, svc: AppService = Depends(get_service), p: Principal = Depends(require_principal)) -> dict[str, Any]:
    try:
        return svc.community_share(
            model_id, description=req.description, mode=req.mode, feedback_enabled=req.feedback_enabled,
            feedback_floor=req.feedback_floor, cartridge_id=req.cartridge_id, client_id=p.client_id,
        )
    except (ServiceError, ValueError) as exc:
        raise HTTPException(status_code=getattr(exc, "status", 400), detail=str(exc))
