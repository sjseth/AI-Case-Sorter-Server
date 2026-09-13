import json
import zipfile
from pathlib import Path

import pytest

from aiserver import model_io, paths
from aiserver.db import Database
from aiserver.models import Model
from aiserver.store import HeadstampRepo, ModelRepo
from tests.conftest import make_jpeg


def _repos():
    db = Database(":memory:")
    return ModelRepo(db), HeadstampRepo(db)


def test_export_import_roundtrip(data_root, tmp_path):
    models, hs = _repos()
    m = models.create(Model(name="Nine", cartridge_name="9mm"))
    img_dir = paths.model_images_dir(m.id)
    img_dir.mkdir(parents=True)
    (img_dir / "WIN__1.jpg").write_bytes(make_jpeg())
    (img_dir / "FC__2.jpg").write_bytes(make_jpeg())
    ckpt = paths.model_checkpoint_path(m.id)
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"PK\x03\x04fake")
    out = model_io.export_model(tmp_path / "x.zip", m, "9mm", ["WIN", "FC"], model_file=ckpt, images_dir=img_dir)
    with zipfile.ZipFile(out) as zf:
        names = sorted(zf.namelist())
    assert names == ["FC__2.jpg".join(["images/", ""]), "images/WIN__1.jpg", "manifest.json", f"model/{m.id}.pth"]
    manifest = model_io.read_manifest(out)
    assert manifest["ModelName"] == "Nine" and manifest["ExportMode"] == "ModelAndImages"
    assert "model_path" not in manifest["ModelInfo"] and "api_key" not in manifest["ModelInfo"]["ai_model_config"]

    imported = model_io.import_model(out, model_repo=models, headstamp_repo=hs)
    assert imported.name == "Nine (2)" and imported.model_type == "Standard"
    assert Path(imported.model_path).name == f"{imported.id}.pth"
    assert sorted(p.name for p in paths.model_images_dir(imported.id).iterdir()) == ["FC__2.jpg", "WIN__1.jpg"]
    assert hs.names(imported.id) == ["FC", "WIN"]


def test_legacy_manifest_and_community_update(data_root, tmp_path):
    models, hs = _repos()
    legacy = {
        "ModelName": "Legacy", "CartridgeName": "45", "Headstamps": ["RP"], "ExportMode": "ModelOnly",
        "ModelInfo": {"Name": "Legacy", "ModelMode": 7, "ModelType": 0, "ModelVersion": 3, "CommunityModelUID": "uid-1",
                      "PythonTrainingConfig": {"ModelName": "convnext_small", "ImageSize": 480, "Epochs": 12},
                      "AIModelConfig": {"OpenAI_EndpointUrl": "http://x", "OpenAI_APIKey": "secret"}},
    }
    z = tmp_path / "legacy.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("manifest.json", json.dumps(legacy))
        zf.writestr("model\\trainedmodel.zip", b"PK\x03\x04fake")  # Windows backslash entry + legacy name
    m = model_io.import_model(z, model_repo=models, headstamp_repo=hs, community_download=True)
    assert m.model_mode == "convnext_small" and m.model_type == "CommunityManaged"
    assert m.training_config.image_size == 480 and m.model_version == 3
    assert m.model_path.endswith(f"{m.id}.pth") and hs.names(m.id) == ["RP"]

    # a newer version of the same UID updates in place
    legacy["ModelInfo"]["ModelVersion"] = 4
    z2 = tmp_path / "legacy2.zip"
    with zipfile.ZipFile(z2, "w") as zf:
        zf.writestr("manifest.json", json.dumps(legacy))
    m2 = model_io.import_model(z2, model_repo=models, headstamp_repo=hs, community_download=True)
    assert m2.id == m.id and m2.model_version == 4 and len(models.list()) == 1
    assert m2.model_type == "CommunityManaged" and m2.model_path == m.model_path


def test_rejects_traversal_and_bad_entries(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("images/../evil.jpg", b"x")
    with pytest.raises(ValueError, match="traversal"):
        model_io.validate_archive(z)
    z2 = tmp_path / "bad2.zip"
    with zipfile.ZipFile(z2, "w") as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("model/evil.exe", b"x")
    with pytest.raises(ValueError, match="unexpected model entry"):
        model_io.validate_archive(z2)


def test_export_mode_parse():
    assert model_io.ExportMode.parse("model_and_images") is model_io.ExportMode.MODEL_AND_IMAGES
    assert model_io.ExportMode.parse("ImagesOnly") is model_io.ExportMode.IMAGES_ONLY
    with pytest.raises(ValueError):
        model_io.ExportMode.parse("nope")
