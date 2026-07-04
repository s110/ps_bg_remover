"""Tests del pipeline con un motor falso (sin GPU ni descarga de modelo)."""

import numpy as np
import pytest
from PIL import Image

from kamiru.core.matting import BaseMatter, MatteResult, compose_rgba
from kamiru.core.pipeline import BatchOptions, run_batch


class FakeMatter(BaseMatter):
    """Alfa = píxeles no-grises (los fondos de prueba son grises)."""

    name = "fake"

    def alpha(self, rgb):
        arr = np.asarray(rgb, dtype=np.float32)
        spread = arr.max(axis=2) - arr.min(axis=2)
        return (spread > 40).astype(np.float32)

    def cutout(self, rgb):
        a = self.alpha(rgb)
        return MatteResult(alpha=a, rgba=compose_rgba(rgb, a))


def scene(pieces, size=(300, 200)):
    """Fondo gris con rectángulos de color. pieces = [(x, y, w, h, color)]"""
    arr = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    for x, y, w, h, color in pieces:
        arr[y:y + h, x:x + w] = color
    return Image.fromarray(arr)


@pytest.fixture
def folder(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    scene([(40, 40, 60, 50, (200, 30, 40)), (180, 90, 70, 60, (30, 60, 200)),
           (10, 180, 3, 3, (250, 220, 10))]).save(inp / "dos_piezas.png")
    scene([(100, 60, 80, 80, (20, 160, 60))]).save(inp / "una_pieza.jpg")
    (inp / "rota.jpg").write_bytes(b"no soy imagen")
    (inp / "notas.txt").write_text("ignorame")
    return inp


def test_conjunto_mode(folder, tmp_path):
    out = tmp_path / "out"
    s = run_batch(folder, out, FakeMatter(), BatchOptions(mode="conjunto", fmt="png"))
    assert s.total == 3           # txt ignorado
    assert s.succeeded == 2
    assert len(s.failed) == 1 and s.failed[0].source.name == "rota.jpg"
    names = sorted(p.name for p in out.iterdir())
    assert names == ["dos_piezas.png", "una_pieza.png"]


def test_individual_mode_names_and_min_area(folder, tmp_path):
    out = tmp_path / "out"
    s = run_batch(folder, out, FakeMatter(),
                  BatchOptions(mode="individual", fmt="png", min_area=100))
    names = sorted(p.name for p in out.iterdir())
    # la mota de 3x3=9px queda descartada por min_area=100
    assert names == ["dos_piezas_01.png", "dos_piezas_02.png", "una_pieza_01.png"]
    rep = {r.source.name: r for r in s.reports}
    assert rep["dos_piezas.png"].objects_discarded == 1


def test_individual_mode_keeps_speck_when_area_zero(folder, tmp_path):
    out = tmp_path / "out"
    run_batch(folder, out, FakeMatter(),
              BatchOptions(mode="individual", fmt="png", min_area=1))
    assert len(list(out.glob("dos_piezas_*.png"))) == 3


def test_no_overwrite_on_second_run(folder, tmp_path):
    out = tmp_path / "out"
    opts = BatchOptions(mode="conjunto", fmt="png")
    run_batch(folder, out, FakeMatter(), opts)
    run_batch(folder, out, FakeMatter(), opts)
    names = sorted(p.name for p in out.iterdir())
    assert "una_pieza.png" in names and "una_pieza-1.png" in names


def test_summary_text(folder, tmp_path):
    s = run_batch(folder, tmp_path / "out", FakeMatter(), BatchOptions())
    text = s.text()
    assert "Imágenes procesadas: 2 de 3" in text
    assert "rota.jpg" in text


def test_individual_psd_layered(folder, tmp_path):
    from psd_tools import PSDImage

    out = tmp_path / "out"
    run_batch(folder, out, FakeMatter(),
              BatchOptions(mode="individual", fmt="psd", min_area=100))
    psd = PSDImage.open(out / "dos_piezas.psd")
    assert len(psd) == 2
    assert [l.name for l in psd] == ["dos_piezas_01", "dos_piezas_02"]
    assert psd.width == 300 and psd.height == 200
    # las capas quedan en su posición original (± margen)
    assert abs(psd[0].left - 38) <= 4 and abs(psd[0].top - 38) <= 4


def test_conjunto_psd(folder, tmp_path):
    from psd_tools import PSDImage

    out = tmp_path / "out"
    run_batch(folder, out, FakeMatter(), BatchOptions(mode="conjunto", fmt="psd"))
    psd = PSDImage.open(out / "una_pieza.psd")
    assert len(psd) == 1


def test_tiff16_batch(folder, tmp_path):
    import tifffile

    out = tmp_path / "out"
    run_batch(folder, out, FakeMatter(), BatchOptions(mode="conjunto", fmt="tiff16"))
    data = tifffile.imread(out / "una_pieza.tif")
    assert data.dtype == np.uint16


def test_cancel_stops_early(folder, tmp_path):
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1  # cancela después de la primera imagen

    s = run_batch(folder, tmp_path / "out", FakeMatter(), BatchOptions(), cancel=cancel)
    assert s.total < 3


def test_exif_orientation(tmp_path):
    """Una imagen con EXIF Orientation=6 debe rotarse antes de procesar."""
    from PIL import Image as PILImage

    inp = tmp_path / "in"
    inp.mkdir()
    im = scene([(10, 10, 40, 30, (200, 30, 40))], size=(100, 60))
    exif = PILImage.Exif()
    exif[274] = 6  # rotar 90° CW
    im.save(inp / "rotada.jpg", exif=exif, quality=95)

    out = tmp_path / "out"
    run_batch(inp, out, FakeMatter(),
              BatchOptions(mode="conjunto", fmt="png", margin=0))
    result = PILImage.open(out / "rotada.png")
    # la pieza era 40x30 apaisada; tras rotar EXIF debe salir ~30x40
    assert result.height > result.width
