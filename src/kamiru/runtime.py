"""Guardas de runtime para la GUI sin consola (pythonw / ejecutable windowed).

Bajo pythonw y en el .exe con console=False, sys.stdout/sys.stderr son None:
cualquier print o barra tqdm de una librería rompe el hilo que lo intente.
Se redirigen a un archivo de log junto a la app y se apagan las barras de
progreso de HuggingFace/tqdm (la GUI ya muestra su propio progreso).
"""

from __future__ import annotations

import os
import sys

from .paths import logs_dir


def ensure_std_streams() -> None:
    if sys.stdout is not None and sys.stderr is not None:
        return
    logfile = logs_dir() / "gui_console.log"
    stream = open(logfile, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def quiet_library_progress() -> None:
    """Sin barras tqdm de librerías en la GUI: el estado ya se muestra ahí."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
