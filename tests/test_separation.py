import numpy as np

from kamiru.core.separation import (
    content_bbox,
    crop_component,
    find_components,
    separate,
    split_touching,
)


def make_alpha(h=200, w=300):
    a = np.zeros((h, w), dtype=np.float32)
    a[20:80, 30:90] = 1.0       # pieza 1: 60x60 = 3600 px
    a[100:180, 150:260] = 1.0   # pieza 2: 80x110 = 8800 px
    a[10:13, 250:253] = 1.0     # mota: 9 px
    return a


def test_find_components_filters_small():
    comps = find_components(make_alpha(), min_area=100)
    assert len(comps) == 2
    areas = sorted(c.area for c in comps)
    assert areas == [3600, 8800]


def test_find_components_keeps_speck_without_filter():
    comps = find_components(make_alpha(), min_area=1)
    assert len(comps) == 3


def test_reading_order():
    comps = find_components(make_alpha(), min_area=100)
    # arriba primero
    assert comps[0].bbox[1] < comps[1].bbox[1]


def test_content_bbox():
    bbox = content_bbox(make_alpha())
    x, y, w, h = bbox
    assert (x, y) == (30, 10)
    assert x + w == 260 and y + h == 180


def test_content_bbox_empty():
    assert content_bbox(np.zeros((50, 50), dtype=np.float32)) is None


def test_crop_component_limits_alpha_to_component():
    a = make_alpha()
    rgba = np.zeros((*a.shape, 4), dtype=np.uint8)
    rgba[..., 3] = (a * 255).astype(np.uint8)
    comps = find_components(a, min_area=100)
    piece = crop_component(rgba, a, comps[0], margin=2)
    # el recorte contiene solo la pieza 1 (más margen), no la pieza 2
    assert piece.shape[0] <= 60 + 8 and piece.shape[1] <= 60 + 8
    assert piece[..., 3].max() == 255


def test_split_touching_divides_two_circles():
    import cv2

    a = np.zeros((200, 400), dtype=np.float32)
    cv2.circle(a, (120, 100), 60, 1.0, -1)
    cv2.circle(a, (240, 100), 60, 1.0, -1)  # se solapan un poco
    comps = find_components(a, min_area=100)
    assert len(comps) == 1  # sin watershed son un solo blob
    pieces = split_touching(comps[0], a, min_area=100)
    assert len(pieces) == 2


def test_separate_with_split_flag():
    import cv2

    a = np.zeros((200, 400), dtype=np.float32)
    cv2.circle(a, (120, 100), 60, 1.0, -1)
    cv2.circle(a, (240, 100), 60, 1.0, -1)
    assert len(separate(a, min_area=100)) == 1
    assert len(separate(a, min_area=100, split_touching_pieces=True)) == 2


def test_split_touching_single_piece_untouched():
    a = make_alpha()
    comps = find_components(a, min_area=100)
    pieces = split_touching(comps[1], a, min_area=100)
    assert len(pieces) == 1
    assert pieces[0] is comps[1]
