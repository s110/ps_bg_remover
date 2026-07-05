"""Descarga de modelos con progreso detallado (para GUI y CLI).

En vez del snapshot opaco de HuggingFace, se listan los archivos del repo con
sus tamaños y se bajan uno por uno, reportando nombre, tamaño, avance del
total y qué ya estaba en cache. El callback recibe strings listos para
mostrar en la barra de estado.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..paths import configure_hf_cache, hf_token

log = logging.getLogger("kamiru.model_fetch")

ProgressCb = Callable[[str], None]


def _mb(n: int | None) -> str:
    if not n:
        return "?"
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    return f"{n / (1 << 20):.0f} MB"


def ensure_model(repo: str, progress: ProgressCb | None = None) -> str:
    """Descarga (o valida en cache) todos los archivos del repo. → ruta local.

    Reporta por el callback: total a descargar, archivo actual (i/N, tamaño)
    y porcentaje acumulado. Si todo está en cache no descarga nada.
    """
    configure_hf_cache()

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub import try_to_load_from_cache

    def say(msg: str) -> None:
        log.info(msg)
        if progress:
            progress(msg)

    token = hf_token()
    api = HfApi(token=token)
    info = api.model_info(repo, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]

    pending = []
    cached_bytes = 0
    for name, size in files:
        hit = try_to_load_from_cache(repo, name, revision=info.sha)
        if isinstance(hit, str):
            cached_bytes += size
        else:
            pending.append((name, size))

    total = sum(s for _, s in files)
    if not pending:
        say(f"Modelo {repo} ya en cache local ({_mb(total)}), sin descarga.")
    else:
        todo = sum(s for _, s in pending)
        say(f"Descargando {repo}: {len(pending)} archivo(s), {_mb(todo)} "
            f"({_mb(cached_bytes)} ya en cache)...")
        done = 0
        for i, (name, size) in enumerate(pending, start=1):
            say(f"[{i}/{len(pending)}] {name} ({_mb(size)}) — "
                f"{done * 100 // max(todo, 1)}% del total")
            hf_hub_download(repo, name, revision=info.sha, token=token)
            done += size
        say(f"Modelo {repo} descargado y cacheado ({_mb(total)}).")

    # ruta del snapshot local
    path = hf_hub_download(repo, files[0][0], revision=info.sha, token=token)
    from pathlib import Path

    return str(Path(path).parent)
