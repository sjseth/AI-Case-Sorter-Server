"""Offline evaluation: classify a folder of labelled images and score it.

Ground truth comes from the ``{label}__{ticks}.ext`` filename convention. The
result shape matches the desktop client's evaluator (``results`` rows with
``predicted`` / ``confidence`` (0-100) / ``original`` / ``match``; ``summary``
with total/known accuracy and a per-predicted-class table) plus a confusion
matrix, and is stored as JSON in the model's ``reports/`` folder.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image

from . import paths
from .image_store import IMAGE_EXTS, list_images, parse_label

ClassifyFn = Callable[[Image.Image], tuple[str, float]]
ProgressFn = Callable[[int, int, str], None]
StopFn = Callable[[], bool]


def folder_classes(folder: Path | str) -> list[str]:
    seen: dict[str, None] = {}
    for p in list_images(folder):
        label = parse_label(p.name)
        if label:
            seen.setdefault(label, None)
    return sorted(seen, key=str.casefold)


def map_classification(original: str, mapping: dict[str, str]) -> tuple[str, bool]:
    if not original or not original.strip():
        return "", False
    if original in mapping:
        return mapping[original], True
    upper = original.upper()
    for key, value in mapping.items():
        if key.upper() == upper:
            return value, True
    return original, False


def match_status(predicted: str, original: str) -> str:
    if not original:
        return "unknown"
    return "match" if predicted == original else "mismatch"


# ----- auto-map suggestion (verbatim port) ----------------------------------

def _tokenize(headstamp: str) -> list[str]:
    return [t for t in re.split(r"[ \-_+]+", headstamp) if t]


def _norm(token: str) -> str:
    return "".join(ch for ch in token if ch.isalnum()).upper()


def _tokens_equivalent(a: str, b: str) -> bool:
    return a.casefold() == b.casefold() or _norm(a) == _norm(b)


def _score_token_match(target: list[str], model: list[str]) -> float:
    matches = [(t, m) for t in range(len(target)) for m in range(len(model)) if _tokens_equivalent(target[t], model[m])]
    if not matches:
        return 0.0
    match_count = min(len({t for t, _ in matches}), len({m for _, m in matches}))
    if match_count == 1:
        t0, m0 = matches[0]
        if t0 != 0 or m0 != 0:
            return 0.0
    score = match_count * 100.0
    positional = sum(1 for i in range(min(len(target), len(model))) if _tokens_equivalent(target[i], model[i]))
    score += positional * 50.0
    in_order, last = True, -1
    for _, m in sorted(matches, key=lambda x: x[0]):
        if m < last:
            in_order = False
            break
        last = m
    if in_order:
        score += 25.0
    score -= abs(len(target) - len(model)) * 10.0
    if match_count == len(target) == len(model):
        score += 75.0
    return score


def suggest_mapping(target_class: str, model_classes: list[str]) -> str | None:
    target = (target_class or "").strip()
    if not target:
        return None
    for mc in model_classes:
        if mc.casefold() == target.casefold():
            return mc
    target_tokens = _tokenize(target)
    best, best_score = None, 0.0
    for mc in model_classes:
        score = _score_token_match(target_tokens, _tokenize(mc))
        if score > best_score:
            best, best_score = mc, score
    if best is not None:
        return best
    for mc in model_classes:
        if target.casefold().startswith(mc.casefold()):
            return mc
    for mc in model_classes:
        if mc.casefold() in target.casefold():
            return mc
    return None


def auto_map(target_classes: list[str], model_classes: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for target in target_classes:
        suggestion = suggest_mapping(target, model_classes)
        if suggestion:
            out[target] = suggestion
    return out


# ----- evaluation ------------------------------------------------------------

def evaluate_folder(
    folder: Path | str,
    *,
    classify: ClassifyFn,
    mapping: dict[str, str] | None = None,
    extensions: Iterable[str] = IMAGE_EXTS,
    progress: ProgressFn | None = None,
    should_stop: StopFn | None = None,
) -> dict[str, Any]:
    mapping = mapping or {}
    images = list_images(folder, extensions)
    total = len(images)
    results: list[dict[str, Any]] = []
    stopped = False
    for i, path in enumerate(images, 1):
        if should_stop is not None and should_stop():
            stopped = True
            break
        if progress is not None:
            progress(i, total, path.name)
        predicted, confidence = "ERROR", 0.0
        try:
            with Image.open(path) as img:
                predicted, confidence = classify(img.convert("RGB"))
        except Exception:  # noqa: BLE001 - one bad file never aborts the run
            predicted, confidence = "ERROR", 0.0
        raw = parse_label(path.name) or ""
        original, was_mapped = map_classification(raw, mapping)
        results.append(
            {
                "filename": path.name,
                "predicted": predicted,
                "confidence": float(confidence),
                "original": original,
                "raw_original": raw,
                "has_mapping": was_mapped,
                "match": match_status(predicted, original),
            }
        )
    return {"results": results, "summary": summarize(results), "confusion": confusion(results), "stopped": stopped}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    with_original = [r for r in results if r["original"]]
    with_mapping = [r for r in results if r["has_mapping"]]
    total_matches = sum(1 for r in with_original if r["predicted"] == r["original"])
    known_matches = sum(1 for r in with_mapping if r["predicted"] == r["original"])
    avg_conf = (sum(r["confidence"] for r in results) / total) if total else 0.0

    by_class: dict[str, dict[str, Any]] = {}
    for r in results:
        bucket = by_class.setdefault(r["predicted"], {"count": 0, "confs": [], "mismatches": 0})
        bucket["count"] += 1
        bucket["confs"].append(r["confidence"])
        if r["original"] and r["predicted"] != r["original"]:
            bucket["mismatches"] += 1
    per_class = []
    for cls, b in sorted(by_class.items(), key=lambda kv: kv[0].casefold()):
        confs = b["confs"]
        per_class.append(
            {
                "cls": cls,
                "count": b["count"],
                "avg": sum(confs) / len(confs) if confs else 0.0,
                "high": max(confs) if confs else 0.0,
                "low": min(confs) if confs else 0.0,
                "mismatches": b["mismatches"],
            }
        )
    return {
        "total": total,
        "with_original": len(with_original),
        "with_mapping": len(with_mapping),
        "total_matches": total_matches,
        "known_matches": known_matches,
        "total_accuracy": (total_matches / len(with_original) * 100.0) if with_original else None,
        "known_accuracy": (known_matches / len(with_mapping) * 100.0) if with_mapping else None,
        "avg_confidence": avg_conf,
        "errors": sum(1 for r in results if r["predicted"] == "ERROR"),
        "per_class": per_class,
    }


def confusion(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Truth x predicted counts, only for rows with ground truth."""
    labels: dict[str, None] = {}
    for r in results:
        if r["original"]:
            labels.setdefault(r["original"], None)
            labels.setdefault(r["predicted"], None)
    ordered = sorted(labels, key=str.casefold)
    idx = {name: i for i, name in enumerate(ordered)}
    matrix = [[0] * len(ordered) for _ in ordered]
    for r in results:
        if r["original"]:
            matrix[idx[r["original"]]][idx[r["predicted"]]] += 1
    return {"labels": ordered, "matrix": matrix}


def save_report(model_id: int, report: dict[str, Any]) -> Path:
    reports = paths.model_reports_dir(model_id)
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = reports / f"evaluation_{stamp}.json"
    report = {**report, "created_at": datetime.now(timezone.utc).isoformat(), "model_id": int(model_id)}
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def list_reports(model_id: int) -> list[dict[str, Any]]:
    reports = paths.model_reports_dir(model_id)
    if not reports.is_dir():
        return []
    out = []
    for p in sorted(reports.glob("evaluation_*.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        summary = data.get("summary") or {}
        out.append(
            {
                "name": p.name,
                "created_at": data.get("created_at"),
                "total": summary.get("total"),
                "total_accuracy": summary.get("total_accuracy"),
                "avg_confidence": summary.get("avg_confidence"),
                "source": data.get("source"),
            }
        )
    return out


def load_report(model_id: int, name: str) -> dict[str, Any] | None:
    if "/" in name or "\\" in name or not name.endswith(".json"):
        return None
    p = paths.model_reports_dir(model_id) / name
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
