"""Tests del control de calidad: señales del alfa (nivel 1), consenso entre
modelos (nivel 2) y triage a la carpeta revisar/ — sin GPU ni modelo real."""

import numpy as np
import pytest
from PIL import Image

from kamiru.core.matting import BaseMatter, MatteResult, compose_rgba
from kamiru.core.pipeline import REVIEW_DIRNAME, BatchOptions, run_batch
from kamiru.core.quality import (
    QualityFlag,
    assess_cutout,
    compare_alphas,
    merge_flags,
)


def scene(pieces, size=(300, 200), bg=128):
    """Fondo gris con rectángulos de color. pieces = [(x, y, w, h, color)]"""
    arr = np.full((size[1], size[0], 3), bg, dtype=np.uint8)
    for x, y, w, h, color in pieces:
        arr[y:y + h, x:x + w] = color
    return Image.fromarray(arr)


def codes(assessment):
    return [f.code for f in assessment.flags]


# ------------------------------------------------------------- nivel 1

def test_clean_cutout_has_no_flags():
    rgb = scene([(100, 60, 80, 80, (200, 30, 40))])
    alpha = np.zeros((200, 300), dtype=np.float32)
    alpha[60:140, 100:180] = 1.0
    a = assess_cutout(rgb, alpha)
    assert not a.suspicious
    assert a.metrics["uncertain_ratio"] == 0.0


def test_milky_alpha_flags_borde_difuso():
    rgb = scene([(90, 70, 120, 60, (200, 30, 40))])
    alpha = np.zeros((200, 300), dtype=np.float32)
    alpha[40:160, 60:240] = 0.5   # zona incierta enorme alrededor
    alpha[70:130, 90:210] = 1.0
    a = assess_cutout(rgb, alpha)
    assert "borde_difuso" in codes(a)


def test_object_touching_border_flags_toca_borde():
    rgb = scene([(0, 50, 80, 100, (200, 30, 40))])
    alpha = np.zeros((200, 300), dtype=np.float32)
    alpha[50:150, 0:80] = 1.0
    a = assess_cutout(rgb, alpha)
    assert "toca_borde" in codes(a)


def test_background_colored_object_flags_color_fondo():
    # el "objeto" tiene exactamente el color del fondo: fuga clásica
    rgb = scene([])  # todo gris
    alpha = np.zeros((200, 300), dtype=np.float32)
    alpha[60:140, 100:180] = 1.0
    a = assess_cutout(rgb, alpha)
    assert "color_fondo" in codes(a)


def test_full_frame_object_flags_ocupa_todo():
    rgb = scene([(0, 0, 300, 200, (200, 30, 40))])
    alpha = np.zeros((200, 300), dtype=np.float32)
    alpha[2:-2, 2:-2] = 1.0
    a = assess_cutout(rgb, alpha)
    assert "ocupa_todo" in codes(a)


def test_empty_alpha_has_no_flags():
    a = assess_cutout(scene([]), np.zeros((200, 300), dtype=np.float32))
    assert not a.suspicious


# ------------------------------------------------------------- nivel 2

def test_compare_alphas_identical_and_disjoint():
    a = np.zeros((50, 50), dtype=np.float32)
    a[10:30, 10:30] = 1.0
    b = np.zeros((50, 50), dtype=np.float32)
    b[35:45, 35:45] = 1.0
    assert compare_alphas(a, a).iou == 1.0
    assert compare_alphas(a, b).iou == 0.0
    empty = np.zeros((50, 50), dtype=np.float32)
    assert compare_alphas(empty, empty).iou == 1.0


def test_compare_alphas_shape_mismatch():
    with pytest.raises(ValueError):
        compare_alphas(np.zeros((10, 10)), np.zeros((20, 20)))


def test_merge_flags_consensus_clears_soft_keeps_hard():
    soft = QualityFlag("borde_difuso", "difuso", soft=True)
    hard = QualityFlag("toca_borde", "toca", soft=False)
    agree = compare_alphas(np.ones((10, 10)), np.ones((10, 10)))  # iou 1
    assert merge_flags([soft, hard], agree) == [hard]

    a = np.zeros((10, 10), dtype=np.float32); a[:5] = 1.0
    b = np.zeros((10, 10), dtype=np.float32); b[5:] = 1.0
    disagree = compare_alphas(a, b)  # iou 0
    merged = merge_flags([soft, hard], disagree)
    assert [f.code for f in merged] == ["borde_difuso", "toca_borde", "modelos_difieren"]


# ------------------------------------------------------------- pipeline

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


class EmptyMatter(BaseMatter):
    """Motor de contraste que nunca ve nada: desacuerdo garantizado."""

    name = "empty"

    def alpha(self, rgb):
        return np.zeros((rgb.height, rgb.width), dtype=np.float32)


@pytest.fixture
def folder(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    scene([(100, 60, 80, 80, (20, 160, 60))]).save(inp / "una_pieza.png")
    return inp


def test_disagreeing_verifier_moves_to_revisar(folder, tmp_path):
    out = tmp_path / "out"
    s = run_batch(folder, out, FakeMatter(),
                  BatchOptions(mode="conjunto", fmt="png", move_uncertain=True),
                  verifier=EmptyMatter())
    assert (out / REVIEW_DIRNAME / "una_pieza.png").exists()
    assert not (out / "una_pieza.png").exists()
    assert len(s.to_review) == 1
    assert s.reports[0].consensus_iou == 0.0
    assert "no coinciden" in s.text()
    assert REVIEW_DIRNAME in s.text()


def test_agreeing_verifier_keeps_output_in_place(folder, tmp_path):
    out = tmp_path / "out"
    s = run_batch(folder, out, FakeMatter(),
                  BatchOptions(mode="conjunto", fmt="png", move_uncertain=True),
                  verifier=FakeMatter())
    assert (out / "una_pieza.png").exists()
    assert not (out / REVIEW_DIRNAME).exists()
    assert not s.to_review
    assert s.reports[0].consensus_iou == 1.0


def test_review_reported_without_moving(folder, tmp_path):
    out = tmp_path / "out"
    s = run_batch(folder, out, FakeMatter(),
                  BatchOptions(mode="conjunto", fmt="png", move_uncertain=False),
                  verifier=EmptyMatter())
    assert (out / "una_pieza.png").exists()
    assert not (out / REVIEW_DIRNAME).exists()
    assert len(s.to_review) == 1
    assert "Recortes dudosos: 1" in s.text()


def test_individual_mode_moves_pieces_to_revisar(folder, tmp_path):
    out = tmp_path / "out"
    run_batch(folder, out, FakeMatter(),
              BatchOptions(mode="individual", fmt="png", move_uncertain=True),
              verifier=EmptyMatter())
    assert (out / REVIEW_DIRNAME / "una_pieza_01.png").exists()


def test_broken_verifier_does_not_break_batch(folder, tmp_path):
    class BrokenMatter(BaseMatter):
        def alpha(self, rgb):
            raise RuntimeError("me rompí")

    out = tmp_path / "out"
    s = run_batch(folder, out, FakeMatter(),
                  BatchOptions(mode="conjunto", fmt="png", move_uncertain=True),
                  verifier=BrokenMatter())
    assert s.succeeded == 1
    assert (out / "una_pieza.png").exists()  # sin consenso, sin drama


def test_no_empty_revisar_dir_when_nothing_exported(folder, tmp_path):
    # primario no ve nada (sin export) pero el consenso marca dudoso:
    # revisar/ no debe quedar creada vacía
    out = tmp_path / "out"
    s = run_batch(folder, out, EmptyMatter(),
                  BatchOptions(mode="conjunto", fmt="png", move_uncertain=True),
                  verifier=FakeMatter())
    assert s.reports[0].review          # los modelos no coinciden
    assert not s.reports[0].outputs     # pero no hubo nada que exportar
    assert not (out / REVIEW_DIRNAME).exists()


# ------------------------------------------------------------- CLI

def test_cli_rejects_same_verifier_model():
    from kamiru.cli import main

    with pytest.raises(SystemExit):
        main(["fotos", "salida", "--verificar",
              "--modelo", "birefnet", "--modelo-verificacion", "birefnet"])


def test_cli_verifier_load_failure_is_not_fatal(tmp_path, monkeypatch, capsys):
    """Si el 2º modelo no carga (sin red), el lote sigue con el principal."""
    from kamiru.cli import main

    monkeypatch.setenv("KAMIRU_HOME", str(tmp_path))
    monkeypatch.setattr("kamiru.core.device.detect_device", lambda: object())

    def boom(*_a, **_k):
        raise RuntimeError("sin red")

    monkeypatch.setattr("kamiru.core.matting.load_matter", boom)

    inp = tmp_path / "in"
    inp.mkdir()
    scene([(100, 60, 80, 80, (20, 160, 60))]).save(inp / "una_pieza.png")
    out = tmp_path / "out"

    rc = main([str(inp), str(out), "--motor", "croma", "--verificar"])
    assert rc == 0
    assert (out / "una_pieza.png").exists()
    assert "no se pudo cargar el modelo de contraste" in capsys.readouterr().out


def test_quality_check_can_be_disabled(folder, tmp_path):
    # todo gris: el "objeto" tiene el color del fondo → dudoso con calidad ON
    inp = tmp_path / "gray"
    inp.mkdir()
    arr = np.full((200, 300, 3), 128, dtype=np.uint8)
    arr[60:140, 100:180] = 150  # apenas distinto: FakeMatter no lo ve
    Image.fromarray(arr).save(inp / "gris.png")

    class AllSeeing(BaseMatter):
        def alpha(self, rgb):
            arr = np.asarray(rgb, dtype=np.float32)
            return (np.abs(arr[..., 0] - 150) < 5).astype(np.float32)

        def cutout(self, rgb):
            a = self.alpha(rgb)
            return MatteResult(alpha=a, rgba=compose_rgba(rgb, a))

    out1 = tmp_path / "out1"
    s1 = run_batch(inp, out1, AllSeeing(), BatchOptions(move_uncertain=True))
    assert (out1 / REVIEW_DIRNAME / "gris.png").exists()

    out2 = tmp_path / "out2"
    s2 = run_batch(inp, out2, AllSeeing(),
                   BatchOptions(move_uncertain=True, quality_check=False))
    assert (out2 / "gris.png").exists()
    assert not s2.to_review
