"""Rutas de la aplicación: cache de modelos y logs, siempre locales al repo/app.

El cache vive junto a la app (carpeta ``models/``) para que el modelo se
descargue una sola vez y se reutilice en cada corrida, sin depender del
cache global de HuggingFace del usuario.
"""

from __future__ import annotations

import os
from pathlib import Path


def app_root() -> Path:
    """Raíz de la instalación (carpeta que contiene src/, models/, logs/)."""
    override = os.environ.get("KAMIRU_HOME")
    if override:
        return Path(override)
    # src/kamiru/paths.py -> src/kamiru -> src -> raíz
    return Path(__file__).resolve().parent.parent.parent


def models_dir() -> Path:
    d = app_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = app_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def configure_hf_cache() -> Path:
    """Fija el cache de HuggingFace dentro de la app y apaga telemetría."""
    d = models_dir()
    os.environ.setdefault("HF_HOME", str(d))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # xet arrastra errores de un intento fallido (p.ej. 401 del repo gated)
    # a las descargas siguientes del mismo proceso; HTTP normal es confiable.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return d


def hf_token() -> str | None:
    """Token de HuggingFace (RMBG-2.0 es un repo gated que lo requiere).

    Se busca en la variable de entorno HF_TOKEN o en models/hf_token.txt,
    donde el setup lo guarda una sola vez.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    token_file = models_dir() / "hf_token.txt"
    if token_file.exists():
        content = token_file.read_text(encoding="utf-8").strip()
        return content or None
    return None


def default_model_file() -> Path:
    """Archivo donde el setup deja anotado el modelo por defecto disponible."""
    return models_dir() / "default_model.txt"
