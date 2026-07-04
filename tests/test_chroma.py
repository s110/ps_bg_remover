import numpy as np
from PIL import Image

from kamiru.core.chroma import ChromaMatter, chroma_alpha, sample_background_color


def green_scene():
    arr = np.full((120, 160, 3), (60, 180, 70), dtype=np.uint8)
    arr[40:80, 60:110] = (200, 40, 50)  # objeto rojo
    return Image.fromarray(arr)


def test_sample_background_color():
    im = green_scene()
    c = sample_background_color(im)
    assert abs(c[0] - 60) < 8 and abs(c[1] - 180) < 8 and abs(c[2] - 70) < 8


def test_chroma_alpha_auto():
    a = chroma_alpha(green_scene())
    assert a[10, 10] < 0.1     # fondo
    assert a[60, 80] > 0.9     # objeto


def test_chroma_alpha_manual_color():
    a = chroma_alpha(green_scene(), key_color=(60, 180, 70))
    assert a[10, 10] < 0.1
    assert a[60, 80] > 0.9


def test_chroma_matter_cutout():
    m = ChromaMatter()
    res = m.cutout(green_scene())
    assert res.rgba.mode == "RGBA"
    arr = np.asarray(res.rgba)
    assert arr[10, 10, 3] < 30
    assert arr[60, 80, 3] > 220
