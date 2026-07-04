"""Carga de imágenes: EXIF, HEIC y perfil ICC.

Devuelve siempre un PIL.Image en RGB ya rotado según EXIF, junto con el
perfil ICC original (si existe) para re-embederlo en la salida.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 300_000_000  # fotos grandes de cámara, sin warnings

_HEIF_REGISTERED = False

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif",
    ".webp", ".bmp",
}


def _ensure_heif() -> None:
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass
    _HEIF_REGISTERED = True


@dataclass
class LoadedImage:
    rgb: Image.Image           # RGB, orientación EXIF ya aplicada
    icc_profile: bytes | None  # perfil ICC original, para la salida
    source: Path


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def list_images(folder: Path) -> list[Path]:
    """Imágenes soportadas de una carpeta, orden estable por nombre."""
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and is_supported(p)),
        key=lambda p: p.name.lower(),
    )


def load_image(path: Path) -> LoadedImage:
    _ensure_heif()
    with Image.open(path) as im:
        im.load()
        icc = im.info.get("icc_profile")
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return LoadedImage(rgb=im, icc_profile=icc, source=path)
