"""Test fixtures: an isolated data root and a FastAPI test client per test."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("CASESORTER_SERVER_DATA_DIR", str(root))
    from aiserver import paths

    paths.set_data_root(None)
    yield root
    paths.set_data_root(None)


@pytest.fixture()
def service(data_root, monkeypatch):
    """An AppService on a throwaway registry with a stub inference manager."""
    import types

    from aiserver.service import AppService

    cfg = types.SimpleNamespace(HOST="127.0.0.1", PORT=8000, API_KEY=None, MODELS={}, MODEL_OPTIONS={},
                                PRELOAD_MODELS=False, LOG_LEVEL="INFO", ADMIN_PASSWORD=None, TRAINING_DEVICE="cpu")

    class StubManager:
        def __init__(self):
            self.registry = {}

        def register(self, alias, path, options=None, *, trusted=False, source="registry"):
            self.registry[alias] = (str(path), source)

        def unregister(self, alias):
            self.registry.pop(alias, None)

        def invalidate(self, alias=None):
            pass

        def aliases(self):
            return list(self.registry)

        def has(self, alias):
            return alias in self.registry

        def source(self, alias):
            return self.registry.get(alias, (None, None))[1]

        def describe(self):
            return [{"alias": a, "source": s, "loaded": False, "classes": None} for a, (_, s) in self.registry.items()]

        def predict_path(self, path, image, *, image_size=None, trusted=False, topk=5):
            return {"label": "WIN", "score": 0.9, "model_base": "convnext_tiny", "image_size": image_size or 224,
                    "topk": [{"label": "WIN", "score": 0.9}, {"label": "FC", "score": 0.1}]}

    svc = AppService(cfg, StubManager())
    svc.start()
    yield svc
    svc.stop()


@pytest.fixture()
def client(service, monkeypatch):
    """TestClient over the real FastAPI app wired to the fixture service."""
    from fastapi.testclient import TestClient

    import server

    monkeypatch.setattr(server, "service", service)
    server.app.state.service = service
    with TestClient(server.app) as c:
        yield c


def make_jpeg(color=(200, 30, 30), size=64, seed=0) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (size, size), color)
    ImageDraw.Draw(im).ellipse((4 + seed % 5, 4, size - 6, size - 8 + seed % 5), outline=(255, 255, 255), width=2)
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    return buf.getvalue()
