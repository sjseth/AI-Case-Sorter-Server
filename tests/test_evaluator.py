from pathlib import Path

from aiserver import evaluator
from tests.conftest import make_jpeg


def test_evaluate_folder_and_summary(tmp_path: Path):
    (tmp_path / "WIN__1.jpg").write_bytes(make_jpeg())
    (tmp_path / "WIN__2.jpg").write_bytes(make_jpeg())
    (tmp_path / "FC__3.jpg").write_bytes(make_jpeg())
    (tmp_path / "nolabel.jpg").write_bytes(make_jpeg())
    calls = []

    def classify(img):
        calls.append(img.size)
        return ("WIN", 80.0)

    report = evaluator.evaluate_folder(tmp_path, classify=classify)
    s = report["summary"]
    assert s["total"] == 4 and s["with_original"] == 3 and s["total_matches"] == 2
    assert abs(s["total_accuracy"] - 66.666) < 0.01
    assert report["confusion"]["labels"] == ["FC", "WIN"] and report["confusion"]["matrix"] == [[0, 1], [0, 2]]
    # list_images sorts case-insensitively: FC__3, nolabel, WIN__1, WIN__2
    assert [r["match"] for r in report["results"]] == ["mismatch", "unknown", "match", "match"]


def test_mapping_and_auto_map():
    assert evaluator.map_classification("win", {"WIN": "Winchester"}) == ("Winchester", True)
    assert evaluator.map_classification("x", {}) == ("x", False)
    assert evaluator.suggest_mapping("FC 9mm", ["FC", "WIN"]) == "FC"
    assert evaluator.auto_map(["win", "zzz"], ["WIN", "FC"]) == {"win": "WIN"}


def test_save_and_list_reports(data_root):
    p = evaluator.save_report(7, {"results": [], "summary": {"total": 0}, "confusion": {"labels": [], "matrix": []}})
    assert p.exists()
    assert evaluator.list_reports(7)[0]["name"] == p.name
    assert evaluator.load_report(7, "../x.json") is None
