"""Community sign-in for a headless server.

Same identity as the desktop client -- same MSAL public-client id, the same
Azure AD B2C policy, the same ``ApiRead`` scope and the same on-disk token
cache format (``config/msal_cache.bin``), so a cache copied from a desktop
install works here unchanged.

B2C does not offer the device-code flow, so the server drives the
**authorization-code flow with PKCE** itself:

1. ``begin()`` builds the sign-in URL (redirect ``http://localhost:44300/``,
   the URI registered for the client app) and opens a one-shot loopback
   listener on that port.
2. The operator opens the URL in a browser. If that browser runs on the
   server machine the redirect lands on the listener and sign-in completes
   by itself. Otherwise the browser shows a "can't connect" page for
   ``localhost:44300`` -- the operator copies that page's address and pastes
   it into the UI, which calls ``complete(redirect_url)``.
3. Either path hands the auth response to MSAL, which stores the refresh
   token in the cache; subsequent API calls use ``acquire_token_silent``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import stat
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import paths

log = logging.getLogger(__name__)

CLIENT_ID = "9b0e9de2-e652-47f6-ad7d-c93a90be8a2e"
TENANT_ID = "704a9dfb-f600-47db-b95f-28ea72de1ab3"
B2C_AUTHORITY = "https://sjsreloadingrecipes.b2clogin.com/tfp/sjsreloadingrecipes.onmicrosoft.com/B2C_1_SignInUp"
REDIRECT_PORT = 44300
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/"
API_SCOPES: list[str] = ["https://sjsreloadingrecipes.onmicrosoft.com/7756be51-5446-43ca-9742-4693f53ad48a/ApiRead"]

FLOW_TTL_S = 15 * 60


class AuthError(Exception):
    pass


@dataclass
class AuthResult:
    access_token: str
    name: str | None
    email: str | None
    expires_in: int | None
    raw: dict[str, Any]


def _msal():
    try:
        import msal
    except ImportError as exc:  # pragma: no cover
        raise AuthError("The 'msal' package is not installed. Run setup.py again.") from exc
    return msal


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    """Display-only decode of a JWT payload (no signature check)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _extract_name(claims: dict[str, Any]) -> str | None:
    name = claims.get("name")
    if name:
        return str(name)
    given = f"{claims.get('given_name', '')} {claims.get('family_name', '')}".strip()
    return given or None


def _extract_email(claims: dict[str, Any]) -> str | None:
    emails = claims.get("emails")
    if isinstance(emails, list) and emails:
        return str(emails[0])
    if isinstance(emails, str) and emails:
        return emails
    email = claims.get("email")
    return str(email) if email else None


class _FileTokenCache:
    """MSAL SerializableTokenCache persisted to a 0600 file."""

    def __init__(self, path: Path):
        msal = _msal()
        self.path = path
        self.cache = msal.SerializableTokenCache()
        try:
            if path.exists():
                self.cache.deserialize(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read token cache %s: %s", path, exc)

    def flush(self) -> None:
        if not self.cache.has_state_changed:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(self.cache.serialize(), encoding="utf-8")
            os.replace(tmp, self.path)
            try:
                os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
        except OSError as exc:
            log.warning("Could not write token cache %s: %s", self.path, exc)


class _RedirectCatcher:
    """One-shot HTTP listener on the loopback redirect port."""

    def __init__(self, port: int, on_url):
        self.port = port
        self.on_url = on_url
        self.server: HTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> bool:
        catcher = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                url = f"http://localhost:{catcher.port}{self.path}"
                ok, message = catcher.on_url(url)
                body = (
                    "<html><body style='font-family:sans-serif;padding:2em'>"
                    f"<h2>{'Signed in' if ok else 'Sign-in failed'}</h2><p>{message}</p>"
                    "<p>You can close this tab and return to the CaseSorter AI Server.</p></body></html>"
                ).encode("utf-8")
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                threading.Thread(target=catcher.stop, daemon=True).start()

            def log_message(self, *_args):  # silence
                return

        try:
            self.server = HTTPServer(("127.0.0.1", self.port), Handler)
        except OSError as exc:
            log.warning("Redirect listener on port %d unavailable: %s", self.port, exc)
            return False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="auth-redirect")
        self.thread.start()
        return True

    def stop(self) -> None:
        srv = self.server
        self.server = None
        if srv is not None:
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:  # noqa: BLE001
                pass


class AuthManager:
    def __init__(self, cache_path: Path | None = None, *, client_id: str = CLIENT_ID, authority: str = B2C_AUTHORITY):
        msal = _msal()
        self._cache = _FileTokenCache(cache_path or paths.msal_cache_path())
        self._client = msal.PublicClientApplication(client_id, authority=authority, token_cache=self._cache.cache)
        self._last_claims: dict[str, Any] | None = None
        self._flow: dict[str, Any] | None = None
        self._flow_started = 0.0
        self._catcher: _RedirectCatcher | None = None
        self._lock = threading.RLock()
        self.last_error: str | None = None

    # -- introspection --------------------------------------------------------

    def accounts(self) -> list[dict[str, Any]]:
        return list(self._client.get_accounts() or [])

    def current_account(self) -> dict[str, Any] | None:
        accts = self.accounts()
        return accts[0] if accts else None

    def is_authenticated(self) -> bool:
        return self.current_account() is not None

    def identity(self) -> tuple[str | None, str | None]:
        claims = self._last_claims
        if not claims:
            acct = self.current_account()
            if acct is None:
                return None, None
            claims = self._cached_id_claims(acct) or acct.get("id_token_claims") or {}
        return _extract_name(claims), _extract_email(claims)

    def _cached_id_claims(self, acct: dict[str, Any]) -> dict[str, Any]:
        cache = self._cache.cache
        try:
            kind = cache.CredentialType.ID_TOKEN
            query = {"home_account_id": acct.get("home_account_id")}
            entries = cache.search(kind, query=query) if hasattr(cache, "search") else cache.find(kind, query=query)
            for entry in entries:
                secret = entry.get("secret")
                if secret:
                    claims = _decode_jwt_claims(secret)
                    if claims:
                        return claims
        except Exception:  # noqa: BLE001
            pass
        return {}

    def status(self) -> dict[str, Any]:
        name, email = self.identity()
        flow_active = self._flow is not None and (time.time() - self._flow_started) < FLOW_TTL_S
        return {
            "signed_in": self.is_authenticated(),
            "name": name,
            "email": email,
            "login_pending": flow_active,
            "auth_url": (self._flow or {}).get("auth_uri") if flow_active else None,
            "redirect_uri": REDIRECT_URI,
            "listener_active": bool(self._catcher and self._catcher.server is not None),
            "last_error": self.last_error,
        }

    # -- tokens ---------------------------------------------------------------

    def _remember(self, result: dict[str, Any]) -> None:
        claims = result.get("id_token_claims")
        if isinstance(claims, dict):
            self._last_claims = claims

    def acquire_token_silent(self, scopes: list[str] | None = None) -> AuthResult | None:
        acct = self.current_account()
        if acct is None:
            return None
        result = self._client.acquire_token_silent(scopes or API_SCOPES, account=acct)
        if not result or "access_token" not in result:
            return None
        self._cache.flush()
        self._remember(result)
        return AuthResult(
            access_token=result["access_token"],
            name=_extract_name(result.get("id_token_claims") or {}) or self.identity()[0],
            email=_extract_email(result.get("id_token_claims") or {}) or self.identity()[1],
            expires_in=result.get("expires_in"),
            raw=result,
        )

    # -- interactive flow -----------------------------------------------------

    def begin(self, *, scopes: list[str] | None = None, start_listener: bool = True) -> dict[str, Any]:
        """Start an auth-code flow; returns the URL the operator must open."""
        with self._lock:
            self.last_error = None
            self._stop_catcher()
            self._flow = self._client.initiate_auth_code_flow(
                scopes or API_SCOPES,
                redirect_uri=REDIRECT_URI,
                prompt="select_account",
            )
            if "auth_uri" not in self._flow:
                err = self._flow.get("error_description") or self._flow.get("error") or "unknown error"
                self._flow = None
                raise AuthError(f"Could not start sign-in: {err}")
            self._flow_started = time.time()
            listening = False
            if start_listener:
                self._catcher = _RedirectCatcher(REDIRECT_PORT, self._on_redirect)
                listening = self._catcher.start()
                if not listening:
                    self._catcher = None
            return {"auth_url": self._flow["auth_uri"], "redirect_uri": REDIRECT_URI, "listener_active": listening}

    def _on_redirect(self, url: str) -> tuple[bool, str]:
        try:
            self.complete(url)
            return True, "Sign-in completed."
        except AuthError as exc:
            return False, str(exc)

    def complete(self, redirect_url: str) -> AuthResult:
        """Finish the flow from the full redirect URL (``http://localhost:44300/?code=...``)."""
        with self._lock:
            flow = self._flow
            if flow is None or (time.time() - self._flow_started) > FLOW_TTL_S:
                self._flow = None
                raise AuthError("No sign-in in progress (or it expired). Start again.")
            text = (redirect_url or "").strip()
            if text.startswith("?"):
                text = REDIRECT_URI + text
            elif not text.startswith("http"):
                text = REDIRECT_URI + "?" + text.lstrip("?")
            query = parse_qs(urlparse(text).query)
            response = {k: v[0] for k, v in query.items() if v}
            if "code" not in response and "error" not in response:
                raise AuthError("That URL has no authorization code. Paste the full address of the page you were redirected to.")
            try:
                result = self._client.acquire_token_by_auth_code_flow(flow, response)
            except ValueError as exc:
                raise AuthError(f"Sign-in could not be completed: {exc}") from exc
            if not result or "access_token" not in result:
                err = (result or {}).get("error_description") or (result or {}).get("error") or "no access_token returned"
                self.last_error = str(err)
                raise AuthError(f"Login failed: {err}")
            self._flow = None
            self._stop_catcher()
            self._cache.flush()
            self._remember(result)
            log.info("Community sign-in completed for %s", self.identity()[1] or self.identity()[0])
            return AuthResult(
                access_token=result["access_token"],
                name=_extract_name(result.get("id_token_claims") or {}),
                email=_extract_email(result.get("id_token_claims") or {}),
                expires_in=result.get("expires_in"),
                raw=result,
            )

    def cancel(self) -> None:
        with self._lock:
            self._flow = None
            self._stop_catcher()

    def _stop_catcher(self) -> None:
        if self._catcher is not None:
            self._catcher.stop()
            self._catcher = None

    def logout(self) -> None:
        with self._lock:
            for acct in self.accounts():
                self._client.remove_account(acct)
            self._cache.flush()
            self._last_claims = None
            self.cancel()
