#!/bin/bash
# ============================================================
#  Kamiru - build del ejecutable autocontenido (macOS)
#  Produce dist/Kamiru/Kamiru y Kamiru-macos.zip (PyTorch con
#  soporte MPS para GPU Apple viene incluido).
# ============================================================
set -e
cd "$(dirname "$0")/.."

# uv para instalar rápido (bootstrap con pip si falta)
if command -v uv >/dev/null 2>&1; then
    UV=uv
else
    python3 -m pip install -q uv
    UV="python3 -m uv"
fi

$UV venv build-env --python 3.12
PY=build-env/bin/python

$UV pip install --python "$PY" torch torchvision
$UV pip install --python "$PY" -e . pyinstaller

"$PY" -m PyInstaller packaging/kamiru.spec --noconfirm --distpath dist --workpath build

echo "Comprimiendo dist/Kamiru..."
ditto -c -k --keepParent dist/Kamiru Kamiru-macos.zip
echo
echo "Listo: dist/Kamiru/Kamiru  y  Kamiru-macos.zip"
echo "Al primer uso la app descarga el modelo a la carpeta models/ junto al ejecutable."
