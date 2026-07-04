"""Corrida de ejemplo del spec: procesa samples/input con el modelo real.

Ejecuta tres corridas y deja la salida en samples/output/:
  1. modo conjunto → PNG
  2. modo individual → PNG (verifica separación + filtro de motas)
  3. modo individual → PSD por capas

Uso:  python scripts/sample_run.py [modelo]   (default: rmbg-2.0)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kamiru.cli import main as kamiru_main  # noqa: E402
from kamiru.core.matting import resolve_default_model  # noqa: E402

SAMPLES = ROOT / "samples"


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else resolve_default_model()
    inp = SAMPLES / "input"
    if not inp.exists() or not any(inp.iterdir()):
        print("Generando fotos de prueba primero...")
        import make_samples

        make_samples.main()

    runs = [
        ("conjunto_png", ["--modo", "conjunto", "--formato", "png"]),
        ("individual_png", ["--modo", "individual", "--formato", "png", "--area-minima", "400"]),
        ("individual_psd", ["--modo", "individual", "--formato", "psd", "--area-minima", "400"]),
    ]
    for name, extra in runs:
        out = SAMPLES / "output" / name
        print(f"\n=== Corrida: {name} ===")
        code = kamiru_main([str(inp), str(out), "--modelo", model, *extra])
        # la corrida devuelve 1 si hubo errores; la foto corrupta es esperada
        print(f"(exit={code}; el error de corrupta.jpg es intencional)")
    print(f"\nSalida verificable en {SAMPLES / 'output'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
