import io
import time

from tests.conftest import make_jpeg


def _wait(client, job_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/v1/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_loopback_server_needs_no_password(client):
    assert client.get("/api/admin/auth").json()["password_required"] is False
    assert client.get("/api/v1/server").status_code == 200


def test_model_lifecycle_and_images(client):
    r = client.post("/api/v1/models", json={"name": "Nine", "cartridge_name": "9mm", "headstamps": ["WIN"]})
    assert r.status_code == 201
    mid = r.json()["id"]
    assert r.json()["headstamps"] == ["WIN"]
    # duplicate names are made unique
    assert client.post("/api/v1/models", json={"name": "nine"}).json()["name"] == "nine (2)"

    files = [("files", (f"FC__{i}.jpg", make_jpeg(seed=i), "image/jpeg")) for i in range(3)]
    r = client.post(f"/api/v1/models/{mid}/images", files=files)
    assert r.status_code == 201 and len(r.json()["saved"]) == 3
    r = client.post(f"/api/v1/models/{mid}/images", files=[("files", ("x.jpg", make_jpeg(), "image/jpeg"))], data={"label": "RP"})
    assert r.json()["saved"][0]["label"] == "RP"
    r = client.post(f"/api/v1/models/{mid}/images", files=[("files", ("nolabel.jpg", make_jpeg(), "image/jpeg"))])
    assert r.json()["errors"] and not r.json()["saved"]

    lst = client.get(f"/api/v1/models/{mid}/images", params={"label": "FC"}).json()
    assert lst["total"] == 3 and lst["labels"] == {"FC": 3, "RP": 1}
    name = lst["items"][0]["filename"]
    assert client.get(f"/api/v1/models/{mid}/images/{name}", params={"thumb": "true"}).headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/v1/models/{mid}/images/..%2Fx.jpg").status_code in (400, 404)

    r = client.post(f"/api/v1/models/{mid}/images/bulk", json={"action": "reclassify", "filenames": [name], "label": "WIN"})
    assert len(r.json()["changed"]) == 1
    assert client.get(f"/api/v1/models/{mid}/headstamps").json() == ["FC", "RP", "WIN"]
    r = client.post(f"/api/v1/models/{mid}/headstamps/rename", json={"old": "WIN", "new": "Winchester"})
    assert r.json()["images_renamed"] == 1
    r = client.post(f"/api/v1/models/{mid}/images/bulk", json={"action": "delete", "filenames": [r.json()["headstamps"][0] + "__nope.jpg"]})
    assert r.status_code == 200

    assert client.patch(f"/api/v1/models/{mid}", json={"name": "Renamed", "training_config": {"epochs": 7}}).json()["training_config"]["epochs"] == 7
    assert client.post(f"/api/v1/models/{mid}/serve", json={"enabled": True}).status_code == 400, "no checkpoint yet"
    assert client.delete(f"/api/v1/models/{mid}").status_code == 204
    assert client.get(f"/api/v1/models/{mid}").status_code == 404


def test_train_requires_two_classes(client):
    mid = client.post("/api/v1/models", json={"name": "One"}).json()["id"]
    assert client.post(f"/api/v1/models/{mid}/train").status_code == 400
    client.post(f"/api/v1/models/{mid}/images", files=[("files", ("A__1.jpg", make_jpeg(), "image/jpeg"))])
    assert "two" in client.post(f"/api/v1/models/{mid}/train").json()["detail"]


def test_serve_classify_and_openai_endpoint(client, service):
    from aiserver import paths

    mid = client.post("/api/v1/models", json={"name": "Served"}).json()["id"]
    ckpt = paths.model_checkpoint_path(mid)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"stub")
    m = service.models.get(mid)
    m.model_path = str(ckpt)
    service.models.update(m)

    r = client.post(f"/api/v1/models/{mid}/classify", files={"file": ("a.jpg", make_jpeg(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["label"] == "WIN" and r.json()["topk"][1]["label"] == "FC"

    r = client.post(f"/api/v1/models/{mid}/serve", json={"enabled": True, "alias": "nine"})
    assert r.json()["is_serving"] and r.json()["alias"] == "nine"
    assert "nine" in [d["id"] for d in client.get("/v1/models").json()["data"]]
    assert client.post(f"/api/v1/models/{mid}/serve", json={"enabled": True, "alias": "nine"}).status_code == 200
    other = client.post("/api/v1/models", json={"name": "Other"}).json()["id"]
    assert client.patch(f"/api/v1/models/{other}", json={"serve_alias": "nine"}).status_code == 200  # alias only matters when serving

    assert client.post(f"/api/v1/models/{mid}/evaluate", json={}).status_code == 400, "no images yet"
    client.post(f"/api/v1/models/{mid}/images", files=[("files", ("WIN__1.jpg", make_jpeg(), "image/jpeg"))])
    ev = client.post(f"/api/v1/models/{mid}/evaluate", json={}).json()
    j = _wait(client, ev["id"])
    assert j["status"] == "done" and j["result"]["summary"]["total_accuracy"] == 100.0
    assert client.get(f"/api/v1/models/{mid}/evaluations").json()[0]["total"] == 1


def test_pairing_and_client_auth(client, service):
    code = client.post("/api/admin/clients/pairing-code", json={"label": "Shop"}).json()["code"]
    r = client.post("/api/v1/bind", json={"pairing_code": code, "client_name": "Shop PC"})
    assert r.status_code == 200
    token = r.json()["token"]
    assert client.post("/api/v1/bind", json={"pairing_code": code}).status_code == 400

    # make the server "remote" so anonymous access is refused
    service.config.HOST = "0.0.0.0"
    client.cookies.clear()
    assert client.get("/api/v1/models").status_code == 401
    ok = client.get("/api/v1/models", headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200
    assert client.get("/api/admin/settings", headers={"Authorization": f"Bearer {token}"}).status_code == 403
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {token}"}).status_code == 200

    # admin password + session cookie
    assert client.post("/api/admin/setup", json={"password": "hunter22"}).status_code == 200
    assert client.get("/api/admin/settings").status_code == 200
    client.post("/api/admin/logout")
    assert client.get("/api/admin/settings").status_code == 401
    assert client.post("/api/admin/login", json={"password": "wrong"}).status_code == 401
    assert client.post("/api/admin/login", json={"password": "hunter22"}).status_code == 200

    # revoking a client cuts it off; disabling remote clients cuts all of them off
    cid = service.clients.list()[0].id
    client.post(f"/api/admin/clients/{cid}/revoke")
    client.cookies.clear()  # drop the admin session so only the bearer token counts
    assert client.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_export_import_via_api(client):
    mid = client.post("/api/v1/models", json={"name": "Exp", "headstamps": ["WIN"]}).json()["id"]
    client.post(f"/api/v1/models/{mid}/images", files=[("files", ("WIN__5.jpg", make_jpeg(), "image/jpeg"))])
    r = client.get(f"/api/v1/models/{mid}/export", params={"mode": "ImagesOnly"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    r2 = client.post("/api/v1/models/import", files={"file": ("Exp.zip", r.content, "application/zip")}, data={"name": "Copy"})
    assert r2.status_code == 201 and r2.json()["name"] == "Copy" and r2.json()["image_count"] == 1
    bad = client.post("/api/v1/models/import", files={"file": ("x.zip", b"not a zip", "application/zip")})
    assert bad.status_code == 400
