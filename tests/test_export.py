import numpy as np
import pytest
from PIL import Image

from kamiru.core.export import (
    export_single,
    numbered_stem,
    unique_path,
)


@pytest.fixture
def rgba():
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (40, 60, 4), dtype=np.uint8)
    arr[..., 3] = np.linspace(0, 255, 60, dtype=np.uint8)[None, :]
    return arr


def test_numbered_stem_zero_pad():
    assert numbered_stem("foto", 1, 5) == "foto_01"
    assert numbered_stem("foto", 12, 120) == "foto_012"


def test_unique_path_no_overwrite(tmp_path):
    (tmp_path / "a.png").touch()
    (tmp_path / "a-1.png").touch()
    p = unique_path(tmp_path, "a", ".png")
    assert p.name == "a-2.png"


def test_png_roundtrip(tmp_path, rgba):
    out = export_single(rgba, tmp_path, "img", "png")
    back = np.asarray(Image.open(out).convert("RGBA"))
    assert np.array_equal(back, rgba)


def test_png_embeds_icc(tmp_path, rgba):
    icc = b"\x00" * 128  # perfil dummy
    out = export_single(rgba, tmp_path, "img", "png", icc_profile=icc)
    assert Image.open(out).info.get("icc_profile") == icc


def test_tiff8_roundtrip(tmp_path, rgba):
    import tifffile

    out = export_single(rgba, tmp_path, "img", "tiff")
    assert out.suffix == ".tif"
    back = tifffile.imread(out)
    assert back.dtype == np.uint8
    assert np.array_equal(back, rgba)


def test_tiff16_roundtrip(tmp_path, rgba):
    import tifffile

    out = export_single(rgba, tmp_path, "img", "tiff16")
    back = tifffile.imread(out)
    assert back.dtype == np.uint16
    assert np.array_equal(back, rgba.astype(np.uint16) * 257)
    assert back.max() <= 65535


def test_tiff_embeds_icc(tmp_path, rgba):
    import tifffile

    icc = bytes(range(64))
    out = export_single(rgba, tmp_path, "img", "tiff", icc_profile=icc)
    with tifffile.TiffFile(out) as tf:
        tag = tf.pages[0].tags.get(34675)
        assert tag is not None and bytes(tag.value) == icc


def test_psd_single(tmp_path, rgba):
    from psd_tools import PSDImage

    out = export_single(rgba, tmp_path, "img", "psd")
    psd = PSDImage.open(out)
    assert psd.width == 60 and psd.height == 40
    assert len(psd) == 1
