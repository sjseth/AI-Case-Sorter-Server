"""Training-image storage for a model.

Same filename convention as the desktop client and the legacy Windows app,
so images move between all three without renaming:

    {label}__{ticks}.jpg      ticks = .NET DateTime.Ticks (100 ns since 0001-01-01)

Everything here is pathlib + Pillow; no OpenCV dependency on the server.
"""

from __future__ import annotations

import io
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image

IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ALL_FILTER = "All"
UNKNOWN_FILTER = "UNKNOWN HEADSTAMP"

_UNSAFE_LABEL_CHARS = re.compile(r'[\x00-\x1f\x7f/\\:*?"<>|]')
_RESERVED_WIN_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_MAX_LABEL_LEN = 100
_DOTNET_EPOCH_OFFSET_TICKS = 621_355_968_000_000_000
_SAFE_FILENAME = re.compile(r"^[^/\\\x00]+$")


def safe_label(label: str) -> str:
    cleaned = _UNSAFE_LABEL_CHARS.sub("_", str(label))
    cleaned = cleaned.replace("..", "_").replace("__", "_")
    cleaned = cleaned.strip().strip(". ")
    if cleaned.upper() in _RESERVED_WIN_NAMES:
        cleaned = f"_{cleaned}"
    cleaned = cleaned[:_MAX_LABEL_LEN].strip()
    return cleaned or "unknown"


def dotnet_ticks(when: datetime | None = None) -> int:
    if when is None:
        when = datetime.now(timezone.utc)
    elif when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return int(when.timestamp() * 10_000_000) + _DOTNET_EPOCH_OFFSET_TICKS


def training_filename(label: str, *, ext: str = ".jpg", when: datetime | None = None) -> str:
    if not label or not str(label).strip():
        raise ValueError("label cannot be empty")
    return f"{safe_label(label)}__{dotnet_ticks(when)}{ext}"


def parse_label(filename: str) -> str | None:
    base = Path(filename).name
    stem = Path(base).stem
    if "__" not in stem:
        return None
    return stem.split("__", 1)[0]


def safe_filename(name: str) -> str:
    """Reject anything that is not a bare basename (API inputs)."""
    base = Path(name).name
    if not base or base != name or not _SAFE_FILENAME.match(base) or base in (".", ".."):
        raise ValueError(f"Invalid filename {name!r}")
    return base


def list_images(folder: Path | str, extensions: Iterable[str] = IMAGE_EXTS) -> list[Path]:
    d = Path(folder)
    if not d.is_dir():
        return []
    exts = {e.lower() for e in extensions}
    return sorted(
        (p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts),
        key=lambda p: p.name.casefold(),
    )


def filter_images(files: list[Path], selected: str | None, known_headstamps: Iterable[str]) -> list[Path]:
    if not selected or selected == ALL_FILTER:
        return list(files)
    if selected == UNKNOWN_FILTER:
        known = {k.casefold() for k in known_headstamps}
        return [p for p in files if (parse_label(p.name) or "").casefold() not in known]
    prefix = f"{selected}__".casefold()
    return [p for p in files if p.name.casefold().startswith(prefix)]


def class_counts(folder: Path | str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for p in list_images(folder):
        label = parse_label(p.name)
        if label:
            counter[label] += 1
    return dict(sorted(counter.items(), key=lambda kv: kv[0].casefold()))


def _assert_within(out_dir: Path, dest: Path) -> None:
    base = out_dir.resolve()
    if not dest.resolve().is_relative_to(base):
        raise ValueError(f"refusing to write outside {base}: {dest}")


def save_image_bytes(
    data: bytes,
    out_dir: Path | str,
    label: str,
    *,
    original_name: str | None = None,
    quality: int = 95,
    when: datetime | None = None,
) -> Path:
    """Decode + re-encode an uploaded image as JPEG under the training convention.

    Re-encoding (rather than writing the bytes verbatim) normalises format and
    strips anything that isn't pixel data. If ``original_name`` already follows
    ``{label}__{ticks}`` the ticks are kept so an image round-trips through
    export/import without changing identity.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise ValueError(f"Not a decodable image: {exc}") from exc
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    fname: str | None = None
    if original_name:
        stem = Path(original_name).stem
        if "__" in stem:
            _, _, ticks = stem.partition("__")
            if ticks.isdigit():
                fname = f"{safe_label(label)}__{ticks}.jpg"
    if fname is None:
        fname = training_filename(label, ext=".jpg", when=when)
    dest = out_dir / fname
    _assert_within(out_dir, dest)
    # Never overwrite: bump the ticks until unique.
    while dest.exists():
        dest = out_dir / training_filename(label, ext=".jpg")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    img.save(tmp, format="JPEG", quality=int(quality))
    os.replace(tmp, dest)
    return dest


def reclassify(path: Path | str, new_label: str) -> Path | None:
    p = Path(path)
    name = p.name
    new_label = safe_label(new_label)
    if "__" not in name or not new_label:
        return None
    idx = name.index("__")
    if name[:idx] == new_label:
        return p
    dest = p.with_name(new_label + name[idx:])
    if dest.exists():
        dest = p.with_name(training_filename(new_label, ext=p.suffix))
    try:
        p.rename(dest)
        return dest
    except OSError:
        return None


def delete(path: Path | str, *, attempts: int = 5, retry_delay_s: float = 0.05) -> bool:
    target = Path(path)
    for attempt in range(attempts):
        try:
            target.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if attempt == attempts - 1:
                return False
            time.sleep(retry_delay_s)
    return False


def thumbnail_jpeg(path: Path | str, size: int = 160) -> bytes:
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def image_info(path: Path) -> dict:
    st = path.stat()
    return {
        "filename": path.name,
        "label": parse_label(path.name),
        "size": st.st_size,
        "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }
