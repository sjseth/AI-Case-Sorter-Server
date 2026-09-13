"""``/api/admin`` -- web-UI only: sign-in, settings, bound clients, community login."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from ..service import AppService, ServiceError
from .deps import SESSION_COOKIE, Principal, get_service, require_admin, resolve_principal

router = APIRouter(prefix="/api/admin", tags=["admin"])


class PasswordBody(BaseModel):
    password: str
    current: Optional[str] = None


class LoginBody(BaseModel):
    password: str


class SettingsBody(BaseModel):
    allow_remote_clients: Optional[bool] = None
    auto_serve_downloads: Optional[bool] = None
    auto_serve_trained: Optional[bool] = None
    training_device: Optional[str] = None


class PairingBody(BaseModel):
    label: Optional[str] = None


class RenameBody(BaseModel):
    name: str


class CompleteLoginBody(BaseModel):
    redirect_url: str


def _set_cookie(response: Response, token: str, request: Request) -> None:
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax", secure=request.url.scheme == "https", max_age=12 * 3600, path="/")


@router.get("/auth")
def auth_state(request: Request, svc: AppService = Depends(get_service), principal: Principal | None = Depends(resolve_principal)) -> dict[str, Any]:
    return {
        "authenticated": principal is not None and principal.is_admin,
        "password_set": svc.admin_password_set(),
        "setup_required": not svc.admin_password_set() and svc.admin_required(),
        "password_required": svc.admin_required(),
        "host": getattr(svc.config, "HOST", None),
    }


@router.post("/setup")
def setup(body: PasswordBody, request: Request, response: Response, svc: AppService = Depends(get_service)) -> dict[str, Any]:
    if svc.admin_password_set():
        raise HTTPException(status_code=409, detail="A password is already set; sign in and change it from Settings")
    try:
        svc.set_admin_password(body.password)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    _set_cookie(response, svc.create_session(), request)
    return {"ok": True}


@router.post("/login")
def login(body: LoginBody, request: Request, response: Response, svc: AppService = Depends(get_service)) -> dict[str, Any]:
    if not svc.admin_password_set():
        if svc.admin_required():
            raise HTTPException(status_code=409, detail="Set a password first")
        _set_cookie(response, svc.create_session(), request)
        return {"ok": True}
    if not svc.verify_admin_password(body.password):
        raise HTTPException(status_code=401, detail="Wrong password")
    _set_cookie(response, svc.create_session(), request)
    return {"ok": True}


@router.post("/logout")
def logout(request: Request, response: Response, svc: AppService = Depends(get_service)) -> dict[str, Any]:
    svc.drop_session(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.post("/password")
def change_password(body: PasswordBody, svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> dict[str, Any]:
    try:
        svc.set_admin_password(body.password, current=body.current)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    return {"ok": True}


@router.get("/settings")
def get_settings(svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> dict[str, Any]:
    return svc.get_settings()


@router.patch("/settings")
def update_settings(body: SettingsBody, svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> dict[str, Any]:
    return svc.update_settings(body.model_dump(exclude_none=True))


# ----- bound clients -----------------------------------------------------------

@router.get("/clients")
def list_clients(svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> dict[str, Any]:
    return {"clients": [c.to_public_dict() for c in svc.clients.list()], "pairing_codes": svc.clients.list_pairing_codes()}


@router.post("/clients/pairing-code", status_code=201)
def new_pairing_code(body: PairingBody | None = None, svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> dict[str, Any]:
    return svc.clients.new_pairing_code((body.label if body else None) or None)


@router.post("/clients/{client_id}/revoke")
def revoke_client(client_id: int, svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> dict[str, Any]:
    if svc.clients.get(client_id) is None:
        raise HTTPException(status_code=404, detail="Client not found")
    svc.clients.revoke(client_id)
    return svc.clients.get(client_id).to_public_dict()  # type: ignore[union-attr]


@router.post("/clients/{client_id}/rename")
def rename_client(client_id: int, body: RenameBody, svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> dict[str, Any]:
    if svc.clients.get(client_id) is None:
        raise HTTPException(status_code=404, detail="Client not found")
    svc.clients.rename(client_id, body.name)
    return svc.clients.get(client_id).to_public_dict()  # type: ignore[union-attr]


@router.delete("/clients/{client_id}", status_code=204)
def delete_client(client_id: int, svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> Response:
    svc.clients.delete(client_id)
    return Response(status_code=204)


@router.post("/clients/token", status_code=201)
def issue_token(body: RenameBody, svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> dict[str, Any]:
    """Issue a client token directly (no pairing code) for scripted setups."""
    client, token = svc.clients.create(body.name)
    return {"client": client.to_public_dict(), "token": token}


# ----- community sign-in -------------------------------------------------------

@router.get("/community/status")
def community_status(svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> dict[str, Any]:
    return svc.community_status()


@router.post("/community/login/start")
def community_login_start(svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> dict[str, Any]:
    try:
        return svc.community_login_begin()
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


@router.post("/community/login/complete")
def community_login_complete(body: CompleteLoginBody, svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> dict[str, Any]:
    try:
        return svc.community_login_complete(body.redirect_url)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


@router.post("/community/login/cancel")
def community_login_cancel(svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> dict[str, Any]:
    svc.auth().cancel()
    return svc.community_status()


@router.post("/community/logout")
def community_logout(svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> dict[str, Any]:
    svc.community_logout()
    return svc.community_status()


@router.get("/served")
def served(svc: AppService = Depends(get_service), _: Principal = Depends(require_admin)) -> list[dict[str, Any]]:
    return svc.manager.describe()
