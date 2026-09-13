from pathlib import Path

from aiserver import image_store as im
from tests.conftest import make_jpeg


def test_safe_label_sanitises():
    assert im.safe_label("WIN 9mm") == "WIN 9mm"
    assert im.safe_label("../x/y:z") == "_x_y_z"
    assert im.safe_label("CON") == "_CON"
    assert im.safe_label("   ") == "unknown"
    assert "__" not in im.safe_label("a__b")


def test_training_filename_and_parse():
    name = im.training_filename("FC")
    assert name.startswith("FC__") and name.endswith(".jpg")
    assert im.parse_label(name) == "FC"
    assert im.parse_label("noseparator.jpg") is None


def test_save_reclassify_delete(tmp_path: Path):
    p = im.save_image_bytes(make_jpeg(), tmp_path, "WIN")
    assert p.exists() and im.class_counts(tmp_path) == {"WIN": 1}
    # keeps ticks from an original name that follows the convention
    p2 = im.save_image_bytes(make_jpeg(), tmp_path, "FC", original_name="FC__123456.png")
    assert p2.name == "FC__123456.jpg"
    moved = im.reclassify(p, "RP")
    assert moved is not None and moved.name.startswith("RP__")
    assert im.class_counts(tmp_path) == {"FC": 1, "RP": 1}
    assert im.delete(moved) and not moved.exists()


def test_filter_images(tmp_path: Path):
    for label in ("WIN", "FC", "weird"):
        im.save_image_bytes(make_jpeg(), tmp_path, label)
    files = im.list_images(tmp_path)
    assert len(im.filter_images(files, "WIN", ["WIN", "FC"])) == 1
    assert [p.name for p in im.filter_images(files, im.UNKNOWN_FILTER, ["WIN", "FC"])][0].startswith("weird__")
    assert len(im.filter_images(files, "All", [])) == 3


def test_rejects_non_image(tmp_path: Path):
    import pytest

    with pytest.raises(ValueError):
        im.save_image_bytes(b"not an image", tmp_path, "WIN")


def test_safe_filename():
    import pytest

    assert im.safe_filename("WIN__1.jpg") == "WIN__1.jpg"
    for bad in ("../x.jpg", "a/b.jpg", "..", ""):
        with pytest.raises(ValueError):
            im.safe_filename(bad)
