"""Genera fotos sintéticas de prueba en samples/input/.

Simulan los casos del spec: piezas de papel texturizado sobre madera, green
screen y cartulina de color; varias piezas separadas; motas de polvo; y un
archivo corrupto para verificar el manejo de errores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "samples" / "input"
rng = np.random.default_rng(42)


def wood_background(w: int, h: int) -> Image.Image:
    x = np.linspace(0, 40 * np.pi, w)
    grain = (np.sin(x) * 0.5 + 0.5) * 30
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[..., 0] = 150 + grain
    base[..., 1] = 105 + grain * 0.7
    base[..., 2] = 60 + grain * 0.4
    base += rng.normal(0, 6, (h, w, 3))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def flat_background(w: int, h: int, color: tuple[int, int, int]) -> Image.Image:
    arr = np.full((h, w, 3), color, dtype=np.float32)
    arr += rng.normal(0, 4, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def paper_piece(w: int, h: int, color: tuple[int, int, int]) -> Image.Image:
    """Pieza de "papel" con borde irregular y textura, RGBA."""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # polígono irregular (papel rasgado)
    n = 14
    cx, cy, r = w / 2, h / 2, min(w, h) / 2 - 4
    pts = []
    for i in range(n):
        ang = 2 * np.pi * i / n
        rad = r * (0.82 + 0.18 * rng.random())
        pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    d.polygon(pts, fill=color + (255,))
    # textura de papel
    arr = np.asarray(im).astype(np.float32)
    noise = rng.normal(0, 10, (h, w, 1))
    arr[..., :3] = np.clip(arr[..., :3] + noise, 0, 255)
    im = Image.fromarray(arr.astype(np.uint8))
    return im.filter(ImageFilter.GaussianBlur(0.6))


def paste_shadowed(bg: Image.Image, piece: Image.Image, xy: tuple[int, int]) -> None:
    """Pega una pieza con una sombra suave desplazada (que debe quedar fuera)."""
    shadow = Image.new("RGBA", piece.size, (0, 0, 0, 0))
    alpha = piece.split()[3].point(lambda a: int(a * 0.35))
    shadow.putalpha(alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))
    bg.paste(shadow, (xy[0] + 10, xy[1] + 12), shadow)
    bg.paste(piece, xy, piece)


def add_specks(bg: Image.Image, count: int = 12) -> None:
    d = ImageDraw.Draw(bg)
    w, h = bg.size
    for _ in range(count):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(1, 4))
        d.ellipse([x, y, x + r, y + r], fill=(240, 240, 235))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    W, H = 1600, 1200

    # 1. Madera con 3 piezas de papel separadas + motas
    bg = wood_background(W, H)
    paste_shadowed(bg, paper_piece(380, 320, (235, 228, 210)), (140, 120))
    paste_shadowed(bg, paper_piece(300, 420, (200, 60, 70)), (700, 380))
    paste_shadowed(bg, paper_piece(260, 240, (60, 90, 180)), (1150, 150))
    add_specks(bg)
    bg.save(OUT / "madera_tres_piezas.jpg", quality=92)

    # 2. Green screen con una obra dimensional
    bg = flat_background(W, H, (60, 175, 70))
    paste_shadowed(bg, paper_piece(600, 500, (245, 240, 230)), (480, 330))
    bg.save(OUT / "greenscreen_obra.jpg", quality=92)

    # 3. Cartulina celeste con dos piezas que casi se tocan
    bg = flat_background(W, H, (150, 190, 230))
    paste_shadowed(bg, paper_piece(420, 380, (250, 210, 90)), (350, 380))
    paste_shadowed(bg, paper_piece(400, 360, (120, 70, 140)), (770, 400))
    bg.save(OUT / "cartulina_dos_piezas.png")

    # 4. Papel blanco rasgado sobre fondo claro (bajo contraste)
    bg = flat_background(W, H, (210, 205, 195))
    paste_shadowed(bg, paper_piece(520, 440, (250, 248, 244)), (520, 360))
    bg.save(OUT / "bajo_contraste_papel.jpg", quality=92)

    # 5. Archivo corrupto: verifica que el lote lo salte sin abortar
    (OUT / "corrupta.jpg").write_bytes(b"esto no es un jpeg")

    print(f"Fotos de prueba en {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
