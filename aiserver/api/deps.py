"""Authentication for the remote-client and admin APIs.

A request is one of:

* **admin** -- a valid web-UI session cookie, the configured ``API_KEY`` as a
  bearer token, or (on a loopback-only server with no password set) anyone.
* **client** -- a bearer token issued to a bound light-weight client.

Admin can do everything; a client can do everything model-related but not
change server settings, manage other clients or sign in to the community.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..service import AppService

SESSION_COOKIE = "cs_admin_session"
_bearer = HTTPBearer(auto_error=False)


@dataclass
class Principal:
    kind: str  # "admin" | "client"
    client_id: int | None = None
    client_name: str | None = None

    @property
    def is_admin(self) -> bool:
        return self.kind == "admin"


def get_service(request: Request) -> AppService:
    return request.app.state.service


def resolve_principal(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    svc: AppService = Depends(get_service),
) -> Principal | None:
    token = creds.credentials if creds and creds.scheme.lower() == "bearer" else None
    if token:
        if svc.api_key_valid(token):
            return Principal("admin")
        if svc.settings.get("allow_remote_clients", True):
            client = svc.clients.authenticate(token)
            if client is not None:
                return Principal("client", client.id, client.name)
    if svc.session_valid(request.cookies.get(SESSION_COOKIE)):
        return Principal("admin")
    if not svc.admin_required():
        return Principal("admin")
    return None


def require_principal(principal: Principal | None = Depends(resolve_principal)) -> Principal:
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in, or send a client token / API key as a Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def require_admin(principal: Principal = Depends(require_principal)) -> Principal:
    if not principal.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required")
    return principal
