"""Escritor de PSD por capas, puro Python + numpy.

Genera PSD RGB de 8 bits con N capas RGBA (transparencia real por capa) y
composite fusionado, con compresión RLE (PackBits) — el formato clásico que
Photoshop, Affinity, Krita y GIMP leen sin problema.

Formato de referencia: "Adobe Photoshop File Formats Specification".
Secciones escritas: File Header, Color Mode Data (vacía), Image Resources
(vacía), Layer & Mask Information (registros + datos de canal RLE) e Image
Data (composite RLE).
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class PsdLayer:
    name: str
    rgba: np.ndarray  # uint8 (h, w, 4)
    left: int = 0
    top: int = 0


def _packbits_row(row: np.ndarray) -> bytes:
    """Codifica una fila uint8 con PackBits (RLE de Apple/TIFF/PSD)."""
    n = row.size
    if n == 0:
        return b""
    out = bytearray()
    i = 0
    data = row.tobytes()
    while i < n:
        # ¿run de repetidos?
        run = 1
        while i + run < n and run < 128 and data[i + run] == data[i]:
            run += 1
        if run >= 3:
            out.append(257 - run)  # -(run-1) en complemento a dos
            out.append(data[i])
            i += run
            continue
        # literal: avanzar hasta el próximo run de >=3 o 128 bytes
        start = i
        i += run
        while i < n and (i - start) < 128:
            run = 1
            while i + run < n and run < 3 and data[i + run] == data[i]:
                run += 1
            if run >= 3:
                break
            i += run
        length = i - start
        out.append(length - 1)
        out += data[start:i]
    return bytes(out)


def _encode_channel_rle(channel: np.ndarray) -> bytes:
    """Canal (h, w) uint8 → datos RLE PSD: tabla de largos por fila + filas."""
    rows = [_packbits_row(np.ascontiguousarray(channel[y])) for y in range(channel.shape[0])]
    counts = struct.pack(f">{len(rows)}H", *(len(r) for r in rows))
    return counts + b"".join(rows)


def _pascal_string(name: str, pad_to: int = 4) -> bytes:
    raw = name.encode("ascii", errors="replace")[:255]
    s = bytes([len(raw)]) + raw
    if len(s) % pad_to:
        s += b"\x00" * (pad_to - len(s) % pad_to)
    return s


def _unicode_name_block(name: str) -> bytes:
    """Bloque adicional 'luni' para que el nombre con acentos se vea bien."""
    utf16 = name.encode("utf-16-be")
    data = struct.pack(">I", len(name)) + utf16
    if len(data) % 2:
        data += b"\x00"
    return b"8BIM" + b"luni" + struct.pack(">I", len(data)) + data


def _flatten_composite(layers: list[PsdLayer], width: int, height: int) -> np.ndarray:
    """Composite RGBA uint8 fusionando las capas en orden (alpha-over)."""
    comp = np.zeros((height, width, 4), dtype=np.float32)
    for layer in layers:
        h, w = layer.rgba.shape[:2]
        x0, y0 = layer.left, layer.top
        x1, y1 = min(width, x0 + w), min(height, y0 + h)
        sx0, sy0 = max(0, -x0), max(0, -y0)
        x0, y0 = max(0, x0), max(0, y0)
        if x1 <= x0 or y1 <= y0:
            continue
        src = layer.rgba[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)].astype(np.float32)
        dst = comp[y0:y1, x0:x1]
        sa = src[..., 3:4] / 255.0
        da = dst[..., 3:4] / 255.0
        oa = sa + da * (1 - sa)
        safe = np.where(oa > 0, oa, 1.0)
        dst[..., :3] = (src[..., :3] * sa + dst[..., :3] * da * (1 - sa)) / safe
        dst[..., 3:4] = oa * 255.0
    return np.clip(np.rint(comp), 0, 255).astype(np.uint8)


def write_psd(path, layers: list[PsdLayer], width: int, height: int) -> None:
    """Escribe un PSD RGB 8-bit por capas con transparencia."""
    if not layers:
        raise ValueError("Se necesita al menos una capa para el PSD")
    if width < 1 or height < 1 or width > 30000 or height > 30000:
        raise ValueError(f"Tamaño fuera de rango PSD (1..30000): {width}x{height}")

    buf = io.BytesIO()
    # --- File header: firma, versión 1, 4 canales (RGB + alfa), 8 bits, RGB
    buf.write(b"8BPS")
    buf.write(struct.pack(">H6xHIIHH", 1, 4, height, width, 8, 3))
    # --- Color mode data e image resources: vacíos
    buf.write(struct.pack(">I", 0))
    buf.write(struct.pack(">I", 0))

    # --- Layer & mask information
    layer_records = io.BytesIO()
    channel_data = io.BytesIO()
    for layer in layers:
        rgba = np.ascontiguousarray(layer.rgba, dtype=np.uint8)
        h, w = rgba.shape[:2]
        top, left = layer.top, layer.left
        bottom, right = top + h, left + w
        layer_records.write(struct.pack(">iiii", top, left, bottom, right))

        # canales: -1 (transparencia), 0 (R), 1 (G), 2 (B)
        chan_ids = (-1, 0, 1, 2)
        chan_planes = (rgba[..., 3], rgba[..., 0], rgba[..., 1], rgba[..., 2])
        encoded = []
        for plane in chan_planes:
            data = struct.pack(">H", 1) + _encode_channel_rle(plane)  # 1 = RLE
            encoded.append(data)
        layer_records.write(struct.pack(">H", len(chan_ids)))
        for cid, data in zip(chan_ids, encoded):
            layer_records.write(struct.pack(">hI", cid, len(data)))
            channel_data.write(data)

        layer_records.write(b"8BIM" + b"norm")           # blend mode normal
        layer_records.write(struct.pack(">BBBB", 255, 0, 0, 0))  # opacidad, clip, flags, relleno
        extra = io.BytesIO()
        extra.write(struct.pack(">I", 0))                # layer mask: vacío
        extra.write(struct.pack(">I", 0))                # blending ranges: vacío
        extra.write(_pascal_string(layer.name))
        extra.write(_unicode_name_block(layer.name))
        payload = extra.getvalue()
        layer_records.write(struct.pack(">I", len(payload)))
        layer_records.write(payload)

    layer_info = io.BytesIO()
    layer_info.write(struct.pack(">h", len(layers)))
    layer_info.write(layer_records.getvalue())
    layer_info.write(channel_data.getvalue())
    li = layer_info.getvalue()
    if len(li) % 2:
        li += b"\x00"

    lmi = io.BytesIO()
    lmi.write(struct.pack(">I", len(li)))
    lmi.write(li)
    lmi.write(struct.pack(">I", 0))  # global layer mask info: vacío
    lmi_bytes = lmi.getvalue()
    buf.write(struct.pack(">I", len(lmi_bytes)))
    buf.write(lmi_bytes)

    # --- Image data (composite fusionado), RLE, planar R,G,B,A
    comp = _flatten_composite(layers, width, height)
    buf.write(struct.pack(">H", 1))  # compresión RLE
    planes = [comp[..., 0], comp[..., 1], comp[..., 2], comp[..., 3]]
    all_rows = []
    for plane in planes:
        all_rows.append([_packbits_row(np.ascontiguousarray(plane[y])) for y in range(height)])
    for rows in all_rows:
        buf.write(struct.pack(f">{len(rows)}H", *(len(r) for r in rows)))
    for rows in all_rows:
        for r in rows:
            buf.write(r)

    with open(path, "wb") as f:
        f.write(buf.getvalue())
