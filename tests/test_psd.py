import numpy as np
import pytest
from psd_tools import PSDImage

from kamiru.core.psd_writer import PsdLayer, _packbits_row, write_psd


def _decode_packbits(data: bytes, expected: int) -> bytes:
    out = bytearray()
    i = 0
    while i < len(data) and len(out) < expected:
        n = data[i]
        i += 1
        if n < 128:
            out += data[i:i + n + 1]
            i += n + 1
        elif n > 128:
            out += bytes([data[i]]) * (257 - n)
            i += 1
        # 128 = no-op
    return bytes(out)


@pytest.mark.parametrize("row", [
    np.zeros(100, dtype=np.uint8),
    np.arange(256, dtype=np.uint8),
    np.array([1, 1, 1, 2, 3, 3, 3, 3, 4], dtype=np.uint8),
    np.full(1000, 42, dtype=np.uint8),
    np.tile(np.array([0, 255], dtype=np.uint8), 300),
    np.random.default_rng(1).integers(0, 256, 5000).astype(np.uint8),
])
def test_packbits_roundtrip(row):
    enc = _packbits_row(row)
    assert _decode_packbits(enc, row.size) == row.tobytes()


def _layer(w, h, color, alpha=255):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = color
    arr[..., 3] = alpha
    return arr


def test_psd_two_layers_roundtrip(tmp_path):
    l1 = _layer(50, 40, (255, 0, 0))
    l2 = _layer(30, 20, (0, 0, 255), alpha=128)
    path = tmp_path / "out.psd"
    write_psd(path, [
        PsdLayer("pieza_01", l1, left=5, top=10),
        PsdLayer("pieza_02", l2, left=60, top=50),
    ], width=120, height=100)

    psd = PSDImage.open(path)
    assert psd.width == 120 and psd.height == 100
    assert len(psd) == 2
    names = [layer.name for layer in psd]
    assert names == ["pieza_01", "pieza_02"]

    lay1 = psd[0]
    assert (lay1.left, lay1.top) == (5, 10)
    px = np.asarray(lay1.composite())
    assert px.shape[:2] == (40, 50)
    assert tuple(px[0, 0, :3]) == (255, 0, 0)
    assert px[0, 0, 3] == 255

    lay2 = psd[1]
    px2 = np.asarray(lay2.numpy() * 255, dtype=np.uint8) if hasattr(lay2, "numpy") else None
    assert (lay2.left, lay2.top) == (60, 50)


def test_psd_composite_matches(tmp_path):
    l1 = _layer(20, 20, (10, 200, 30))
    path = tmp_path / "one.psd"
    write_psd(path, [PsdLayer("obj", l1, left=3, top=4)], width=40, height=30)
    psd = PSDImage.open(path)
    comp = np.asarray(psd.composite())
    assert comp.shape[:2] == (30, 40)
    # dentro de la capa: el color exacto
    assert tuple(comp[10, 10, :3]) == (10, 200, 30)


def test_psd_unicode_layer_name(tmp_path):
    path = tmp_path / "uni.psd"
    write_psd(path, [PsdLayer("pieza_ñandú_01", _layer(8, 8, (1, 2, 3)))],
              width=8, height=8)
    psd = PSDImage.open(path)
    assert psd[0].name == "pieza_ñandú_01"


def test_psd_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        write_psd(tmp_path / "x.psd", [], width=10, height=10)


def test_psd_soft_alpha_preserved(tmp_path):
    arr = _layer(64, 8, (100, 100, 100))
    arr[..., 3] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
    path = tmp_path / "soft.psd"
    write_psd(path, [PsdLayer("degrade", arr)], width=64, height=8)
    psd = PSDImage.open(path)
    lay = np.asarray(psd[0].composite())
    assert lay[0, 0, 3] == 0
    assert lay[0, 63, 3] == 255
    assert 100 < lay[0, 32, 3] < 155  # alfa suave, no binario
