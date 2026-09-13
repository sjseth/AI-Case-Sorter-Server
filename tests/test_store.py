from aiserver.db import Database
from aiserver.models import Model, TrainingConfig
from aiserver.store import ClientRepo, HeadstampRepo, JobRepo, ModelRepo, SettingsRepo, utcnow


def test_model_roundtrip():
    db = Database(":memory:")
    repo = ModelRepo(db)
    m = repo.create(Model(name="9mm", cartridge_name="9mm", training_config=TrainingConfig(epochs=3, image_size=480)))
    assert m.id == 1 and m.training_config.epochs == 3 and m.training_config.image_size == 480
    m.serve_enabled = True
    m.serve_alias = "nine"
    repo.update(m)
    assert repo.find_by_alias("NINE").id == 1
    assert repo.unique_name("9mm") == "9mm (2)"
    repo.delete(1)
    assert repo.get(1) is None


def test_headstamps_replace_and_sync():
    db = Database(":memory:")
    m = ModelRepo(db).create(Model(name="x"))
    hs = HeadstampRepo(db)
    assert [h.name for h in hs.replace(m.id, ["WIN", "fc", "FC", " "])] == ["fc", "WIN"]
    hs.sync_from_classes(m.id, ["RP", "WIN"])
    assert hs.names(m.id) == ["fc", "RP", "WIN"]
    hs.rename(m.id, "fc", "FC")
    hs.remove(m.id, "RP")
    assert hs.names(m.id) == ["FC", "WIN"]


def test_client_pairing_and_tokens():
    db = Database(":memory:")
    cr = ClientRepo(db)
    code = cr.new_pairing_code("shop")
    assert cr.redeem_pairing_code("nope", "x") is None
    client, token = cr.redeem_pairing_code(code["code"].lower(), "Shop PC")
    assert client.name == "Shop PC" and token.startswith("csk_")
    assert cr.redeem_pairing_code(code["code"], "again") is None, "single use"
    assert cr.authenticate(token).id == client.id
    cr.revoke(client.id)
    assert cr.authenticate(token) is None


def test_jobs_and_settings():
    db = Database(":memory:")
    jr = JobRepo(db)
    jr.upsert({"id": "a", "kind": "train", "model_id": 1, "status": "running", "created_at": utcnow(), "request": {"x": 1}})
    assert jr.mark_interrupted() == 1
    assert jr.get("a")["status"] == "failed" and jr.get("a")["request"] == {"x": 1}
    sr = SettingsRepo(db)
    sr.set("k", {"nested": [1, 2]})
    assert sr.get("k") == {"nested": [1, 2]} and sr.get("missing", 5) == 5
