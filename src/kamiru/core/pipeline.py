"""Pipeline por lotes: carpeta de entrada → recortes exportados + resumen.

Por cada imagen: cargar (EXIF/HEIC) → alfa con el motor elegido → RGBA a
resolución completa → modo conjunto o individual → exportar. Una imagen que
falla se registra y se salta, sin abortar el lote.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .export import FORMATS, export_single, extension_for, numbered_stem, unique_path
from .imageio import list_images, load_image
from .matting import BaseMatter
from .separation import Component, content_bbox, crop_component, separate

log = logging.getLogger("kamiru.pipeline")


@dataclass
class BatchOptions:
    mode: str = "conjunto"            # "conjunto" | "individual"
    fmt: str = "png"                  # png | tiff | tiff16 | psd
    min_area: int = 400               # px², filtro de motas/polvo
    alpha_threshold: float = 0.5
    margin: int = 2                   # margen extra del recorte, px
    split_touching: bool = False      # watershed para piezas que se tocan
    psd_layered: bool = True          # en individual+PSD: 1 PSD con capas por foto

    def validate(self) -> None:
        if self.mode not in ("conjunto", "individual"):
            raise ValueError(f"Modo inválido: {self.mode!r}")
        if self.fmt not in FORMATS:
            raise ValueError(f"Formato inválido: {self.fmt!r}")
        if self.min_area < 0:
            raise ValueError("El área mínima no puede ser negativa")


@dataclass
class ImageReport:
    source: Path
    ok: bool
    outputs: list[Path] = field(default_factory=list)
    objects_found: int = 0
    objects_discarded: int = 0        # bajo el área mínima
    seconds: float = 0.0
    error: str | None = None


@dataclass
class BatchSummary:
    reports: list[ImageReport] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.reports)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.reports if r.ok)

    @property
    def failed(self) -> list[ImageReport]:
        return [r for r in self.reports if not r.ok]

    @property
    def files_exported(self) -> int:
        return sum(len(r.outputs) for r in self.reports)

    @property
    def empty(self) -> list[ImageReport]:
        return [r for r in self.reports if r.ok and not r.outputs]

    def text(self) -> str:
        lines = [
            f"Imágenes procesadas: {self.succeeded} de {self.total}",
            f"Archivos exportados: {self.files_exported}",
        ]
        if self.empty:
            lines.append(
                f"Sin objetos detectados: {len(self.empty)} "
                f"({', '.join(r.source.name for r in self.empty)})"
            )
        if self.failed:
            lines.append(f"Con error: {len(self.failed)}")
            for r in self.failed:
                lines.append(f"  - {r.source.name}: {r.error}")
        return "\n".join(lines)


# callback(hechas, total, nombre_actual) para la barra de progreso
ProgressFn = Callable[[int, int, str], None]


def _crop_bbox(rgba: np.ndarray, bbox: tuple[int, int, int, int], margin: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
    return rgba[y0:y1, x0:x1]


def process_image(
    path: Path,
    matter: BaseMatter,
    out_dir: Path,
    opts: BatchOptions,
) -> ImageReport:
    t0 = time.perf_counter()
    report = ImageReport(source=path, ok=True)
    loaded = load_image(path)
    result = matter.cutout(loaded.rgb)
    alpha = result.alpha
    rgba = np.asarray(result.rgba, dtype=np.uint8)
    stem = path.stem

    if opts.mode == "conjunto":
        bbox = content_bbox(alpha, opts.alpha_threshold)
        if bbox is None:
            log.warning("%s: sin contenido detectado, no se exporta", path.name)
        else:
            cropped = _crop_bbox(rgba, bbox, opts.margin)
            out = export_single(cropped, out_dir, stem, opts.fmt, loaded.icc_profile)
            report.outputs.append(out)
            report.objects_found = 1
    else:
        comps = separate(
            alpha,
            min_area=opts.min_area,
            threshold=opts.alpha_threshold,
            split_touching_pieces=opts.split_touching,
        )
        from .separation import find_components

        n_all = len(find_components(alpha, min_area=1, threshold=opts.alpha_threshold))
        n_kept = len(find_components(alpha, min_area=opts.min_area, threshold=opts.alpha_threshold))
        report.objects_discarded = max(0, n_all - n_kept)
        report.objects_found = len(comps)
        if not comps:
            log.warning("%s: sin objetos sobre el área mínima", path.name)
        elif opts.fmt == "psd" and opts.psd_layered:
            # Un PSD por foto con cada objeto en su propia capa, en posición
            layers = []
            for i, c in enumerate(comps, start=1):
                piece = crop_component(rgba, alpha, c, margin=opts.margin)
                x, y = max(0, c.bbox[0] - opts.margin), max(0, c.bbox[1] - opts.margin)
                layers.append((numbered_stem(stem, i, len(comps)), piece, x, y))
            out = unique_path(out_dir, stem, extension_for("psd"))
            from .export import save_psd_layers

            save_psd_layers(layers, out, width=rgba.shape[1], height=rgba.shape[0])
            report.outputs.append(out)
        else:
            for i, c in enumerate(comps, start=1):
                piece = crop_component(rgba, alpha, c, margin=opts.margin)
                out = export_single(
                    piece, out_dir, numbered_stem(stem, i, len(comps)),
                    opts.fmt, loaded.icc_profile,
                )
                report.outputs.append(out)

    report.seconds = time.perf_counter() - t0
    return report


def run_batch(
    inputs: Path | list[Path],
    out_dir: Path,
    matter: BaseMatter,
    opts: BatchOptions | None = None,
    progress: ProgressFn | None = None,
    cancel: Callable[[], bool] | None = None,
) -> BatchSummary:
    """Procesa una carpeta o una lista de archivos. Nunca aborta por una imagen."""
    opts = opts or BatchOptions()
    opts.validate()
    if isinstance(inputs, Path):
        files = list_images(inputs)
    else:
        files = [Path(p) for p in inputs]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = BatchSummary()
    total = len(files)
    log.info("Lote: %d imágenes → %s (modo=%s, formato=%s)", total, out_dir, opts.mode, opts.fmt)
    for i, path in enumerate(files, start=1):
        if cancel and cancel():
            log.info("Lote cancelado por el usuario en %d/%d", i - 1, total)
            break
        if progress:
            progress(i - 1, total, path.name)
        try:
            report = process_image(path, matter, out_dir, opts)
            log.info(
                "%s: %d objeto(s), %d archivo(s), %.1fs",
                path.name, report.objects_found, len(report.outputs), report.seconds,
            )
        except Exception as exc:  # noqa: BLE001 — el lote sigue pase lo que pase
            log.error("%s: ERROR %s\n%s", path.name, exc, traceback.format_exc())
            report = ImageReport(source=path, ok=False, error=str(exc))
        summary.reports.append(report)
    if progress:
        progress(len(summary.reports), total, "")
    log.info("Resumen:\n%s", summary.text())
    return summary


def setup_batch_logging(logs_dir: Path) -> Path:
    """Log a archivo por corrida, para diagnosticar sin terminal."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"kamiru_{time.strftime('%Y%m%d_%H%M%S')}.log"
    handler = logging.FileHandler(logfile, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("kamiru")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return logfile
