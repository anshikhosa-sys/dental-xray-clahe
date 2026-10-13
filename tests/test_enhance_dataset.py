import hashlib
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from enhance_dataset import enhance_image, enhance_folder

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "enhance_dataset.py")


def fake_xray(h, w, seed=0):
    #low contrast gray image with some noise
    rng = np.random.default_rng(seed)
    return rng.integers(90, 130, size=(h, w), dtype=np.uint8)


def make_input(tmp_path):
    #nested folders with different image sizes plus junk files
    src = tmp_path / "raw"
    (src / "d2" / "img").mkdir(parents=True)
    cv2.imwrite(str(src / "a.jpg"), fake_xray(64, 128))
    cv2.imwrite(str(src / "d2" / "img" / "b.png"), fake_xray(100, 50, seed=1))
    cv2.imwrite(str(src / "d2" / "img" / "c.jpg"), fake_xray(37, 91, seed=2))
    (src / ".DS_Store").write_bytes(b"junk")
    (src / "d2" / "notes.txt").write_text("not an image")
    return src


def file_hashes(folder):
    out = {}
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            out[os.path.relpath(path, folder)] = hashlib.md5(open(path, "rb").read()).hexdigest()
    return out


def test_enhance_image_keeps_size_and_type():
    img = fake_xray(50, 80)
    out = enhance_image(img, 2.0, 8)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_enhance_image_adds_contrast():
    img = fake_xray(200, 200)
    assert enhance_image(img, 2.0, 8).std() > img.std()


def test_enhance_image_does_not_change_input():
    img = fake_xray(50, 80)
    before = img.copy()
    enhance_image(img, 4.0, 4)
    assert np.array_equal(img, before)


def test_enhance_image_color_input_becomes_gray():
    img = cv2.cvtColor(fake_xray(40, 60), cv2.COLOR_GRAY2BGR)
    out = enhance_image(img)
    assert out.shape == (40, 60)


def test_folder_keeps_names_structure_and_size(tmp_path):
    src = make_input(tmp_path)
    dst = tmp_path / "enhanced"
    count = enhance_folder(str(src), str(dst), 2.0, 8)

    #only the 3 images come out, at the same relative paths
    assert count == 3
    expected = {"a.jpg", os.path.join("d2", "img", "b.png"), os.path.join("d2", "img", "c.jpg")}
    assert set(file_hashes(dst)) == expected

    #every output has the same height and width as its input
    for rel in expected:
        assert cv2.imread(str(dst / rel), 0).shape == cv2.imread(str(src / rel), 0).shape


def test_folder_never_touches_originals(tmp_path):
    src = make_input(tmp_path)
    before = file_hashes(src)
    enhance_folder(str(src), str(tmp_path / "enhanced"))
    assert file_hashes(src) == before


@pytest.mark.parametrize("out_name", ["raw", os.path.join("raw", "out"), "."])
def test_folder_refuses_to_overlap_input(tmp_path, out_name):
    src = make_input(tmp_path)
    before = file_hashes(src)
    with pytest.raises(SystemExit):
        enhance_folder(str(src), str(tmp_path / out_name))
    assert file_hashes(src) == before


def test_command_line(tmp_path):
    src = make_input(tmp_path)
    dst = tmp_path / "enhanced"
    result = subprocess.run([sys.executable, SCRIPT, "--input", str(src), "--output", str(dst), "--clip", "3", "--tile", "4"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "enhanced 3 images" in result.stdout
    assert (dst / "d2" / "img" / "c.jpg").exists()
