"""Croma por color (HSV) — camino alterno al modelo neural.

Sirve para green screen o cartulinas de color uniforme. El color de fondo se
estima automáticamente muestreando las esquinas, o lo da el usuario (clic en
la GUI / --chroma-color en CLI).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .matting import BaseMatter, compose_rgba


def sample_background_color(rgb: Image.Image, patch_frac: float = 0.04) -> tuple[int, int, int]:
    """Estima el color de fondo con la mediana de parches en las 4 esquinas."""
    arr = np.asarray(rgb, dtype=np.uint8)
    h, w = arr.shape[:2]
    ph = max(4, int(h * patch_frac))
    pw = max(4, int(w * patch_frac))
    patches = [
        arr[:ph, :pw], arr[:ph, -pw:], arr[-ph:, :pw], arr[-ph:, -pw:],
    ]
    px = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    med = np.median(px, axis=0)
    return tuple(int(v) for v in med)


def color_distance(arr: np.ndarray, key_color: tuple[int, int, int]) -> np.ndarray:
    """Distancia de cada pixel (RGB uint8 HxWx3) a un color clave, en HSV.

    El matiz (H) es circular y se pondera fuerte, para que un fondo verde no
    arrastre objetos de brillo similar. Si el color clave es casi neutro
    (gris/blanco/negro) el matiz no informa y su peso baja con la saturación.
    """
    import cv2

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    key = cv2.cvtColor(
        np.array([[key_color]], dtype=np.uint8), cv2.COLOR_RGB2HSV
    ).astype(np.float32)[0, 0]

    h = hsv[..., 0] / 180.0
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    kh, ks, kv = key[0] / 180.0, key[1] / 255.0, key[2] / 255.0

    dh = np.abs(h - kh)
    dh = np.minimum(dh, 1.0 - dh) * 2.0  # circular, [0,1]
    hue_w = 3.0 * min(1.0, ks * 4.0)
    return np.sqrt((dh * hue_w) ** 2 + (s - ks) ** 2 + (v - kv) ** 2)


def chroma_alpha(
    rgb: Image.Image,
    key_color: tuple[int, int, int] | None = None,
    tolerance: float = 0.14,
    softness: float = 0.10,
) -> np.ndarray:
    """Alfa float [0,1]: 0 donde el pixel se parece al color de fondo.

    ``tolerance`` es el radio donde el pixel es fondo puro; ``softness`` el
    ancho de la transición suave hacia objeto.
    """
    import cv2

    if key_color is None:
        key_color = sample_background_color(rgb)

    dist = color_distance(np.asarray(rgb, dtype=np.uint8), key_color)

    lo, hi = tolerance, tolerance + max(softness, 1e-6)
    alpha = np.clip((dist - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    # Suavizado leve del borde para evitar escalones duros
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.0)
    return np.clip(alpha, 0.0, 1.0)


class ChromaMatter(BaseMatter):
    """Motor por color, misma interfaz que el motor neural."""

    name = "chroma"

    def __init__(
        self,
        key_color: tuple[int, int, int] | None = None,
        tolerance: float = 0.14,
        softness: float = 0.10,
    ) -> None:
        self.key_color = key_color  # None = automático por esquinas
        self.tolerance = tolerance
        self.softness = softness

    def alpha(self, rgb: Image.Image) -> np.ndarray:
        return chroma_alpha(rgb, self.key_color, self.tolerance, self.softness)

    def cutout(self, rgb: Image.Image):
        from .matting import MatteResult

        a = self.alpha(rgb)
        return MatteResult(alpha=a, rgba=compose_rgba(rgb, a))
