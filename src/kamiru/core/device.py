"""Detección de dispositivo: CUDA (Windows/RTX), MPS (macOS) o CPU con aviso."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceInfo:
    kind: str          # "cuda" | "mps" | "cpu"
    name: str          # nombre legible para mostrar en la GUI
    warning: str | None = None


def detect_device() -> DeviceInfo:
    import torch

    if torch.cuda.is_available():
        return DeviceInfo("cuda", f"GPU NVIDIA: {torch.cuda.get_device_name(0)}")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return DeviceInfo("mps", "GPU Apple (Metal/MPS)")
    return DeviceInfo(
        "cpu",
        "CPU (sin GPU detectada)",
        warning=(
            "No se detectó GPU: el proceso funcionará igual pero será mucho "
            "más lento (decenas de segundos por foto en vez de ~2-3 s)."
        ),
    )
