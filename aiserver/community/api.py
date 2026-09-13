"""Client for the reloadingrecipes.com community backend (port of the desktop
client's ``community_api.py``: same endpoints, same JSON shapes, same casing
tolerance)."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote_plus

import requests

from .auth import API_SCOPES, AuthManager

log = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://www.reloadingrecipes.com/api"
ENV_API_BASE = "CASESORTER_API_BASE"
ENV_CA_BUNDLE = "CASESORTER_API_CA_BUNDLE"
ENV_INSECURE = "CASESORTER_API_INSECURE"
_TRUTHY = {"1", "true", "yes", "on"}

_session = requests.Session()


def api_base() -> str:
    return (os.environ.get(ENV_API_BASE, "").strip() or DEFAULT_API_BASE).rstrip("/")


def _is_loopback(url: str) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.endswith(".localhost") or host.startswith("127.")


def tls_verify() -> bool | str:
    bundle = os.environ.get(ENV_CA_BUNDLE, "").strip()
    if bundle and Path(bundle).exists():
        return bundle
    if os.environ.get(ENV_INSECURE, "").strip().lower() in _TRUTHY and _is_loopback(api_base()):
        try:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:  # noqa: BLE001
            pass
        return False
    return True


def _pick(d: dict[str, Any], key: str, default: Any = "") -> Any:
    camel = key[:1].lower() + key[1:]
    if key in d and d[key] is not None:
        return d[key]
    if camel in d and d[camel] is not None:
        return d[camel]
    return default


def _export_mode_name(raw: Any) -> str:
    names = {0: "ModelOnly", 1: "ModelAndImages", 2: "ImagesOnly", 3: "ManifestOnly"}
    if isinstance(raw, bool):
        return "ModelAndImages"
    if isinstance(raw, int):
        return names.get(raw, "ModelAndImages")
    if isinstance(raw, str) and raw.strip():
        text = raw.strip()
        if text.isdigit():
            return names.get(int(text), "ModelAndImages")
        return text
    return "ModelAndImages"


@dataclass
class CartridgeInfo:
    id: int = 0
    name: str = ""

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "CartridgeInfo":
        return cls(id=int(_pick(d, "Id", 0) or 0), name=str(_pick(d, "Name", "")))


@dataclass
class ModelInfo:
    model_uid: str = ""
    model_name: str = ""
    cartridge_name: str = ""
    cartridge_id: int = 0
    model_version: int = 1
    model_description: str = ""
    author: str = ""
    publish_date: str = ""
    image_count: int = 0
    headstamp_count: int = 0
    download_size: int = 0
    export_mode: str = "ModelAndImages"
    feedback_loop_enabled: bool = False
    feedback_loop_confidence_floor: int = 0

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ModelInfo":
        def as_int(key: str, default: int = 0) -> int:
            try:
                return int(_pick(d, key, default) or 0)
            except (TypeError, ValueError):
                return default

        return cls(
            model_uid=str(_pick(d, "ModelUID", "") or _pick(d, "ModelUid", "")),
            model_name=str(_pick(d, "ModelName", "")),
            cartridge_name=str(_pick(d, "CartridgeName", "")),
            cartridge_id=as_int("CartridgeId"),
            model_version=as_int("ModelVersion", 1),
            model_description=str(_pick(d, "ModelDescription", "") or ""),
            author=str(_pick(d, "Author", "") or ""),
            publish_date=str(_pick(d, "PublishDate", "") or ""),
            image_count=as_int("ImageCount"),
            headstamp_count=as_int("HeadstampCount"),
            download_size=as_int("DownloadSize"),
            export_mode=_export_mode_name(_pick(d, "ModelExportMode", None)),
            feedback_loop_enabled=bool(_pick(d, "FeedbackLoopEnabled", False)),
            feedback_loop_confidence_floor=as_int("FeedbackLoopConfidenceFloor"),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class SasResponse:
    sas_token: str = ""
    blob_path: str = ""
    container_name: str = ""
    container_uri: str = ""
    account_uri: str = ""
    full_url: str = ""
    model_info: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "SasResponse":
        mi = _pick(d, "ModelInfo", {}) or {}
        return cls(
            sas_token=str(_pick(d, "SasToken", "") or ""),
            blob_path=str(_pick(d, "BlobPath", "") or ""),
            container_name=str(_pick(d, "ContainerName", "") or ""),
            container_uri=str(_pick(d, "ContainerURI", "") or _pick(d, "ContainerUri", "") or ""),
            account_uri=str(_pick(d, "AccountURI", "") or _pick(d, "AccountUri", "") or ""),
            full_url=str(_pick(d, "FullUrl", "") or ""),
            model_info=dict(mi) if isinstance(mi, dict) else {},
            raw=dict(d),
        )

    def blob_put_url(self) -> str:
        return f"{self.container_uri.rstrip('/')}/{self.blob_path.lstrip('/')}?{self.sas_token.lstrip('?')}"


@dataclass
class ModelSettings:
    wish_list: list[str] = field(default_factory=list)
    confidence_floor: int = 0
    feedback_enabled: bool = True
    blocked: bool = False
    version: int = 0
    notes: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ModelSettings":
        low = {str(k).lower(): v for k, v in d.items()}

        def as_int(key: str, default: int) -> int:
            try:
                return int(low.get(key, default) or 0)
            except (TypeError, ValueError):
                return default

        wish = low.get("wishlist") or []
        notes_raw = low.get("notes") or []
        notes = []
        for entry in notes_raw if isinstance(notes_raw, list) else []:
            if not isinstance(entry, dict):
                continue
            e = {str(k).lower(): v for k, v in entry.items()}
            text = str(e.get("note") or "").strip()
            if text:
                notes.append({"id": e.get("id"), "note": text, "created": e.get("created")})
        return cls(
            wish_list=[str(w).strip() for w in wish if str(w).strip()] if isinstance(wish, list) else [],
            confidence_floor=as_int("confidencefloor", 0),
            feedback_enabled=bool(low.get("feedbackenabled", True)),
            blocked=bool(low.get("blocked", False)),
            version=as_int("version", 0),
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class UserMetaData:
    profile_name: str = ""
    country: str = ""
    roles: int = 0

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "UserMetaData":
        try:
            roles = int(_pick(d, "Roles", 0) or 0)
        except (TypeError, ValueError):
            roles = 0
        return cls(profile_name=str(_pick(d, "ProfileName", "") or ""), country=str(_pick(d, "Country", "") or ""), roles=roles)

    def can_contribute(self) -> bool:
        return bool(self.roles & 2)

    def to_dict(self) -> dict[str, Any]:
        return {"profile_name": self.profile_name, "country": self.country, "roles": self.roles, "can_contribute": self.can_contribute()}


class CommunityApiError(Exception):
    pass


class _ProgressReader:
    def __init__(self, path: Path | str, total: int, callback: Callable[[int, int], None] | None):
        self._fh = open(path, "rb")
        self._total = total
        self._sent = 0
        self._cb = callback

    def __len__(self) -> int:
        return self._total

    def read(self, size: int = -1) -> bytes:
        chunk = self._fh.read(size)
        self._sent += len(chunk)
        if self._cb is not None:
            self._cb(self._sent, self._total)
        return chunk

    def close(self) -> None:
        self._fh.close()


class CommunityApi:
    def __init__(
        self,
        auth: AuthManager,
        *,
        base_url: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        verify: bool | str | None = None,
    ):
        self.auth = auth
        self.base_url = (base_url or api_base()).rstrip("/")
        self.session = session or _session
        self.timeout = timeout
        self.verify = tls_verify() if verify is None else verify

    def _headers(self) -> dict[str, str]:
        result = self.auth.acquire_token_silent(API_SCOPES)
        if result is None:
            raise CommunityApiError("Not signed in to the community. Sign in first.")
        return {"Authorization": f"Bearer {result.access_token}"}

    def _get(self, path: str, **kwargs: Any) -> requests.Response:
        return self.session.get(f"{self.base_url}{path}", headers=self._headers(), timeout=self.timeout, verify=self.verify, **kwargs)

    def _post(self, path: str, json: Any = None) -> requests.Response:
        return self.session.post(
            f"{self.base_url}{path}",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=json,
            timeout=self.timeout,
            verify=self.verify,
        )

    # -- users ----------------------------------------------------------------

    def get_user_metadata(self) -> UserMetaData | None:
        resp = self._get("/Users/GetUserMetaData")
        if resp.status_code != 200:
            return None
        try:
            return UserMetaData.from_json(resp.json())
        except ValueError:
            return None

    # -- catalogue ------------------------------------------------------------

    def get_available_cartridges(self) -> list[CartridgeInfo]:
        resp = self._get("/Models/GetAvailableCartridges")
        if resp.status_code != 200:
            raise CommunityApiError(f"GetAvailableCartridges -> {resp.status_code}")
        return [CartridgeInfo.from_json(d) for d in resp.json() or []]

    def get_models(self, *, search: str = "", model_type: str = "", cartridge: str = "") -> list[ModelInfo]:
        query = f"searchQuery={quote_plus(search or '')}&ModelType={quote_plus(model_type or '')}&Cartridge={quote_plus(cartridge or '')}"
        resp = self._get(f"/Models/GetModels?{query}")
        if resp.status_code != 200:
            raise CommunityApiError(f"GetModels -> {resp.status_code}")
        return [ModelInfo.from_json(d) for d in resp.json() or []]

    def request_download(self, model_uid: str) -> dict[str, Any]:
        resp = self._get(f"/Models/RequestDownloadModel/{quote_plus(model_uid)}")
        if resp.status_code != 200:
            raise CommunityApiError(f"RequestDownloadModel -> {resp.status_code}")
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def fetch_model_settings(self, model_uid: str) -> ModelSettings | None:
        if not model_uid:
            return None
        try:
            resp = self._get(f"/Models/FetchModelSettings?communityModelId={quote_plus(model_uid)}")
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception:  # noqa: BLE001
            return None
        return ModelSettings.from_json(data) if isinstance(data, dict) else None

    def download_to(
        self,
        url: str,
        dest: Path | str,
        *,
        progress: Callable[[int, int | None], None] | None = None,
        chunk_size: int = 64 * 1024,
        expected_total: int | None = None,
    ) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with self.session.get(url, stream=True, timeout=self.timeout, verify=self.verify) as resp:
            resp.raise_for_status()
            total = expected_total
            if not total:
                try:
                    total = int(resp.headers.get("Content-Length") or 0) or None
                except ValueError:
                    total = None
            done = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
        os.replace(tmp, dest)
        return dest

    # -- sharing --------------------------------------------------------------

    def request_file_upload(self, *, filename: str, model_info: dict[str, Any]) -> SasResponse:
        resp = self._post("/Models/FileUploadRequest", json={"filename": filename, "ModelInfo": model_info})
        text = (resp.text or "").strip()
        if resp.status_code != 200:
            raise CommunityApiError(f"FileUploadRequest -> HTTP {resp.status_code}: {text[:300] or '(empty body)'}")
        if text.startswith("<!DOCTYPE html>") or text.lower().startswith("<html"):
            raise CommunityApiError("The upload endpoint is not available for this account (it requires the Contribute community role).")
        try:
            data = resp.json()
        except ValueError as exc:
            raise CommunityApiError(f"FileUploadRequest returned non-JSON: {text[:200]!r}") from exc
        if not isinstance(data, dict):
            raise CommunityApiError("FileUploadRequest returned an unexpected payload.")
        ticket = SasResponse.from_json(data)
        if not ticket.sas_token and not ticket.full_url:
            raise CommunityApiError("FileUploadRequest returned no SAS token.")
        return ticket

    def upload_blob(
        self,
        file_path: Path | str,
        ticket: SasResponse,
        *,
        content_type: str = "application/octet-stream",
        progress: Callable[[int, int], None] | None = None,
        retries: int = 3,
    ) -> None:
        total = os.path.getsize(file_path)
        url = ticket.blob_put_url()
        headers = {"x-ms-blob-type": "BlockBlob", "Content-Type": content_type}
        last_exc: Exception | None = None
        for attempt in range(max(1, retries)):
            reader = _ProgressReader(file_path, total, progress)
            try:
                resp = self.session.put(url, data=reader, headers=headers, timeout=None, verify=self.verify)
                resp.raise_for_status()
                return
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(2**attempt)
            finally:
                reader.close()
        if last_exc is not None:
            raise last_exc

    def complete_upload(self, ticket: SasResponse) -> bool:
        resp = self._post("/Models/CompleteUpload", json=ticket.raw)
        return resp.status_code < 400

    def request_manifest_upload(self, blob_path: str) -> SasResponse | None:
        resp = self._post("/Models/ManifestUploadRequest", json={"BlobPath": blob_path})
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        return SasResponse.from_json(data) if isinstance(data, dict) else None

    def share_model(
        self,
        *,
        zip_path: Path | str,
        manifest_path: Path | str,
        model_info: dict[str, Any],
        progress: Callable[[int, int], None] | None = None,
    ) -> str | None:
        zip_path = Path(zip_path)
        manifest_path = Path(manifest_path)
        ticket = self.request_file_upload(filename=zip_path.name, model_info=model_info)
        self.upload_blob(zip_path, ticket, content_type="application/zip", progress=progress)
        self.complete_upload(ticket)
        final_uid = ticket.model_info.get("ModelUID") or ticket.model_info.get("modelUID") or model_info.get("CommunityModelUID")
        if manifest_path.exists() and ticket.blob_path:
            manifest_sas = self.request_manifest_upload(ticket.blob_path)
            if manifest_sas is not None:
                self.upload_blob(manifest_path, manifest_sas, content_type="application/json")
        return final_uid
