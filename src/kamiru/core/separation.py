"""Separación de objetos individuales.

Camino principal: ``cv2.connectedComponentsWithStats`` sobre el alfa
binarizado, con filtro de área mínima para descartar motas y polvo.
Camino avanzado (piezas que se tocan): watershed sobre la transformada de
distancia, que divide un componente en varias piezas cuando hay "cuellos".
El alfa suave original se conserva dentro de cada recorte.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Component:
    label: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area: int
    mask: np.ndarray                 # bool HxW (tamaño completo)


def binarize(alpha01: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (alpha01 >= threshold).astype(np.uint8)


def find_components(
    alpha01: np.ndarray,
    min_area: int = 400,
    threshold: float = 0.5,
) -> list[Component]:
    """Componentes conexos del alfa binarizado, filtrados por área mínima."""
    binary = binarize(alpha01, threshold)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out: list[Component] = []
    for i in range(1, n):  # 0 = fondo
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append(Component(
            label=i,
            bbox=(int(x), int(y), int(w), int(h)),
            area=int(area),
            mask=labels == i,
        ))
    # Orden de lectura: arriba→abajo, izquierda→derecha (por esquina del bbox)
    out.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
    return out


def split_touching(
    component: Component,
    alpha01: np.ndarray,
    min_area: int,
    peak_ratio: float = 0.45,
) -> list[Component]:
    """Divide un componente en piezas que se tocan, vía watershed.

    Marca como semillas los máximos de la transformada de distancia por encima
    de ``peak_ratio * max`` y expande con watershed. Si el resultado es una
    sola pieza, devuelve el componente original intacto.
    """
    x, y, w, h, = component.bbox
    sub = component.mask[y:y + h, x:x + w].astype(np.uint8)

    dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return [component]
    peaks = (dist >= peak_ratio * dist.max()).astype(np.uint8)
    n_seeds, seed_labels = cv2.connectedComponents(peaks)
    if n_seeds <= 2:  # fondo + 1 semilla → no hay nada que separar
        return [component]

    # watershed: 0 = por decidir, 1 = fondo, >=2 = semillas de cada pieza
    markers = np.where(seed_labels > 0, seed_labels + 1, 0).astype(np.int32)
    markers[sub == 0] = 1  # fondo como región 1
    img3 = cv2.cvtColor((dist / dist.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(img3, markers)

    pieces: list[Component] = []
    for lbl in range(2, n_seeds + 1):
        piece = (markers == lbl) & (sub > 0)
        area = int(piece.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(piece)
        bx, by = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max()) - bx + 1, int(ys.max()) - by + 1
        full = np.zeros_like(component.mask)
        full[y:y + h, x:x + w] = piece
        pieces.append(Component(
            label=component.label * 1000 + lbl,
            bbox=(x + bx, y + by, bw, bh),
            area=area,
            mask=full,
        ))
    if len(pieces) <= 1:
        return [component]
    pieces.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
    return pieces


def separate(
    alpha01: np.ndarray,
    min_area: int = 400,
    threshold: float = 0.5,
    split_touching_pieces: bool = False,
) -> list[Component]:
    comps = find_components(alpha01, min_area=min_area, threshold=threshold)
    if not split_touching_pieces:
        return comps
    out: list[Component] = []
    for c in comps:
        out.extend(split_touching(c, alpha01, min_area=min_area))
    out.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
    return out


def content_bbox(alpha01: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, int] | None:
    """Bounding box (x, y, w, h) de todo el contenido, o None si está vacío."""
    ys, xs = np.nonzero(alpha01 >= threshold)
    if ys.size == 0:
        return None
    x, y = int(xs.min()), int(ys.min())
    return (x, y, int(xs.max()) - x + 1, int(ys.max()) - y + 1)


def crop_component(
    rgba: np.ndarray,
    alpha01: np.ndarray,
    component: Component,
    margin: int = 2,
    feather_px: float = 1.5,
) -> np.ndarray:
    """Recorta el RGBA de un componente conservando el alfa suave.

    El alfa dentro del recorte se limita a la máscara del componente
    (dilatada + difuminada levemente) para no arrastrar bordes suaves de
    piezas vecinas, sin volver binario el interior.
    """
    h, w = alpha01.shape
    x, y, cw, ch = component.bbox
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w, x + cw + margin), min(h, y + ch + margin)

    sub_rgba = rgba[y0:y1, x0:x1].copy()
    sub_alpha = alpha01[y0:y1, x0:x1]
    sub_mask = component.mask[y0:y1, x0:x1].astype(np.uint8)

    # Expandir la máscara para cubrir el halo suave del borde del objeto
    grow = max(1, int(round(feather_px * 2)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    soft = cv2.dilate(sub_mask, kernel).astype(np.float32)
    if feather_px > 0:
        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=feather_px)
    limited = np.minimum(sub_alpha, soft)

    sub_rgba[..., 3] = np.clip(np.rint(limited * 255.0), 0, 255).astype(np.uint8)
    return sub_rgba
