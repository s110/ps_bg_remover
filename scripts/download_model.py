"""Descarga y cachea el modelo (una sola vez) en la carpeta models/ de la app.

Uso:
    python scripts/download_model.py                # RMBG-2.0 (default)
    python scripts/download_model.py birefnet-hr    # alternativa A/B

RMBG-2.0 es un repo *gated*: requiere cuenta de HuggingFace, aceptar la
licencia y un token en models/hf_token.txt (o variable HF_TOKEN). Si no hay
acceso, este script descarga BiRefNet (misma arquitectura, sin registro) y
lo deja como modelo por defecto para que la app funcione igual.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kamiru.core.matting import DEFAULT_MODEL, GATED_HELP, MODEL_REPOS  # noqa: E402
from kamiru.paths import configure_hf_cache, default_model_file  # noqa: E402


def download(key: str) -> str:
    from kamiru.core.model_fetch import ensure_model

    repo = MODEL_REPOS[key]
    path = ensure_model(repo, progress=print)
    print(f"Listo. Modelo cacheado en: {path}")
    return path


def main() -> int:
    key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    if key not in MODEL_REPOS:
        print(f"Modelo desconocido: {key}. Opciones: {', '.join(MODEL_REPOS)}")
        return 1
    cache = configure_hf_cache()
    print(f"Cache local de modelos: {cache}")

    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        download(key)
        default_model_file().write_text(key, encoding="utf-8")
        print(f"Modelo por defecto de la app: {key}")
        print("Las próximas corridas lo reutilizan sin volver a descargar.")
        return 0
    except (GatedRepoError, HfHubHTTPError) as exc:
        print(f"\nSin acceso a {MODEL_REPOS[key]}: {exc}\n")
        print(GATED_HELP)
        if key != "birefnet":
            print("\nDescargando el modelo alternativo 'birefnet' para que la "
                  "app funcione desde ya...")
            try:
                download("birefnet")
                default_model_file().write_text("birefnet", encoding="utf-8")
                print("Modelo por defecto de la app: birefnet")
                print("(Cuando pegues el token de HuggingFace, vuelve a correr "
                      "este script para pasar a RMBG-2.0.)")
                return 0
            except Exception as exc2:  # noqa: BLE001
                print(f"Tampoco se pudo descargar birefnet: {exc2}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
