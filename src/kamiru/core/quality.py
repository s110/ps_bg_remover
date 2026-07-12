"""Control de calidad del recorte: señales del alfa + consenso entre modelos.

Nivel 1 — señales baratas del propio alfa, sin segundo modelo:
  - borde difuso: demasiada área del objeto en la banda incierta del alfa
    (0.1–0.9). Un recorte limpio es casi binario con una franja fina de
    borde; un alfa "lechoso" es la firma clásica de un matte dudoso.
  - toca el borde: el objeto llega al borde del encuadre → pieza cortada
    por la foto, o fondo confundido con objeto.
  - color de fondo: píxeles marcados como objeto con el mismo color del
    fondo muestreado en las esquinas (fuga de fondo dentro del recorte).
  - ocupa todo: el "objeto" cubre casi todo el encuadre.

Nivel 2 — consenso: se recorta la misma foto con un segundo modelo y se
comparan las máscaras (IoU). Si los dos modelos coinciden, las señales
*suaves* del nivel 1 se descartan como falsa alarma (p. ej. un objeto
traslúcido da borde difuso legítimo); si divergen, la foto queda marcada
para revisar aunque el nivel 1 no haya visto nada.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .chroma import color_distance, sample_background_color


@dataclass(frozen=True)
class QualityThresholds:
    uncertain_low: float = 0.1
    uncertain_high: float = 0.9
    uncertain_ratio_max: float = 0.35   # área incierta / área del objeto
    border_contact_max: float = 0.02    # fracción del marco tocada por objeto
    background_tolerance: float = 0.14  # radio HSV para "es el color del fondo"
    background_leak_max: float = 0.12   # fracción del objeto con color de fondo
    corner_object_max: float = 0.2      # esquinas con objeto → muestreo inválido
    object_frac_max: float = 0.90       # objeto que cubre casi todo el encuadre
    consensus_iou_min: float = 0.90     # debajo: los modelos no coinciden


DEFAULT_THRESHOLDS = QualityThresholds()


@dataclass(frozen=True)
class QualityFlag:
    code: str      # identificador estable (logs y tests)
    message: str   # texto legible para el resumen
    soft: bool     # True: un consenso alto entre modelos la descarta


@dataclass
class QualityAssessment:
    metrics: dict[str, float] = field(default_factory=dict)
    flags: list[QualityFlag] = field(default_factory=list)

    @property
    def suspicious(self) -> bool:
        return bool(self.flags)


def assess_cutout(
    rgb: Image.Image,
    alpha01: np.ndarray,
    alpha_threshold: float = 0.5,
    thresholds: QualityThresholds = DEFAULT_THRESHOLDS,
) -> QualityAssessment:
    """Señales de nivel 1 sobre un recorte (imagen original + alfa).

    Devuelve métricas y una lista de banderas; lista vacía = recorte
    confiable. No decide qué hacer con la foto: eso es del pipeline.
    """
    t = thresholds
    out = QualityAssessment()
    obj = alpha01 >= alpha_threshold
    area = int(obj.sum())
    out.metrics["object_frac"] = frac = area / alpha01.size
    if area == 0:
        return out  # sin objeto: el pipeline ya lo reporta aparte

    uncertain = int(((alpha01 > t.uncertain_low) & (alpha01 < t.uncertain_high)).sum())
    out.metrics["uncertain_ratio"] = ratio = uncertain / area
    if ratio > t.uncertain_ratio_max:
        out.flags.append(QualityFlag(
            "borde_difuso",
            f"borde muy difuso ({ratio:.0%} del área del objeto es incierta)",
            soft=True,
        ))

    # marco de 1 px; las columnas van sin extremos para no contar 2 veces
    # los píxeles de las esquinas
    frame = np.concatenate([obj[0], obj[-1], obj[1:-1, 0], obj[1:-1, -1]])
    out.metrics["border_contact"] = contact = float(frame.mean())
    if contact > t.border_contact_max:
        out.flags.append(QualityFlag(
            "toca_borde",
            f"el objeto toca el borde de la foto ({contact:.0%} del marco)",
            soft=False,
        ))

    if frac > t.object_frac_max:
        out.flags.append(QualityFlag(
            "ocupa_todo",
            f"el recorte cubre casi toda la foto ({frac:.0%}): "
            "puede haberse colado el fondo",
            soft=False,
        ))

    # Fuga del color de fondo — solo si las esquinas son realmente fondo
    # (si el objeto pisa las esquinas, el muestreo no vale y ya hay bandera
    # de borde de todas formas).
    h, w = obj.shape
    ph, pw = max(4, int(h * 0.04)), max(4, int(w * 0.04))
    corner_obj = float(np.mean([
        obj[:ph, :pw].mean(), obj[:ph, -pw:].mean(),
        obj[-ph:, :pw].mean(), obj[-ph:, -pw:].mean(),
    ]))
    if corner_obj < t.corner_object_max:
        key = sample_background_color(rgb)
        dist = color_distance(np.asarray(rgb, dtype=np.uint8), key)
        out.metrics["background_leak"] = leak = float(
            (dist[obj] < t.background_tolerance).mean())
        if leak > t.background_leak_max:
            out.flags.append(QualityFlag(
                "color_fondo",
                f"parte del objeto tiene el color del fondo ({leak:.0%})",
                soft=True,
            ))
    return out


@dataclass
class Consensus:
    iou: float  # coincidencia de las máscaras binarizadas [0,1]

    def agrees(self, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> bool:
        return self.iou >= thresholds.consensus_iou_min


def compare_alphas(
    alpha_a: np.ndarray,
    alpha_b: np.ndarray,
    threshold: float = 0.5,
) -> Consensus:
    """Nivel 2: compara los alfas de dos motores sobre la misma foto."""
    if alpha_a.shape != alpha_b.shape:
        raise ValueError(f"alfas de tamaños distintos: {alpha_a.shape} vs {alpha_b.shape}")
    a = alpha_a >= threshold
    b = alpha_b >= threshold
    union = int(np.logical_or(a, b).sum())
    inter = int(np.logical_and(a, b).sum())
    return Consensus(iou=1.0 if union == 0 else inter / union)


def merge_flags(
    flags: list[QualityFlag],
    consensus: Consensus,
    thresholds: QualityThresholds = DEFAULT_THRESHOLDS,
) -> list[QualityFlag]:
    """Aplica el veredicto del consenso a las banderas del nivel 1.

    Coinciden → las banderas suaves eran falsa alarma y se quitan (las duras
    quedan). Divergen → se agrega la bandera de desacuerdo entre modelos.
    """
    if consensus.agrees(thresholds):
        return [f for f in flags if not f.soft]
    return [*flags, QualityFlag(
        "modelos_difieren",
        f"los dos modelos no coinciden (coincidencia {consensus.iou:.0%})",
        soft=False,
    )]
