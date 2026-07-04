#!/bin/bash
# ============================================================
#  Kamiru - instalación en macOS (correr UNA vez)
#  Crea el venv, instala PyTorch (GPU Apple vía MPS), la app
#  y descarga el modelo RMBG-2.0.
# ============================================================
set -e
cd "$(dirname "$0")"

echo
echo "=== Kamiru: instalación ==="
echo

# --- 1. Python 3.10-3.12 (RMBG-2.0 no soporta 3.13)
PYCMD=""
for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        if "$cand" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)'; then
            PYCMD="$cand"; break
        fi
    fi
done
if [ -z "$PYCMD" ]; then
    echo "ERROR: no se encontró Python 3.10–3.12."
    echo "Instálalo con:  brew install python@3.12"
    echo "o desde https://www.python.org/downloads/"
    read -r -p "Enter para cerrar..."; exit 1
fi
echo "Usando: $PYCMD ($($PYCMD --version))"

# --- 2. Entorno virtual
[ -d venv ] || "$PYCMD" -m venv venv
./venv/bin/pip install --upgrade pip

# --- 3. PyTorch (los wheels de macOS traen soporte MPS para GPU Apple)
./venv/bin/pip install torch torchvision

# --- 4. La app y sus dependencias
./venv/bin/pip install -e .

# --- 5. Verificar GPU Apple
./venv/bin/python -c "import torch; ok=torch.backends.mps.is_available(); print('GPU Apple (MPS):', 'disponible' if ok else 'no disponible (se usará CPU, más lento)')"

# --- 6. Token de HuggingFace (RMBG-2.0 es un repo con registro)
mkdir -p models
if [ ! -f models/hf_token.txt ]; then
    echo
    echo "RMBG-2.0 requiere una cuenta GRATIS de HuggingFace:"
    echo "  1. Crear cuenta:      https://huggingface.co/join"
    echo "  2. Aceptar licencia:  https://huggingface.co/briaai/RMBG-2.0"
    echo "  3. Crear token Read:  https://huggingface.co/settings/tokens"
    echo
    read -r -p "Pega aquí el token (o Enter para usar el modelo alternativo BiRefNet): " HFTOK
    [ -n "$HFTOK" ] && printf '%s\n' "$HFTOK" > models/hf_token.txt
fi

# --- 7. Descargar y cachear el modelo (una sola vez)
echo
echo "Descargando el modelo (~1 GB, solo esta vez)..."
./venv/bin/python scripts/download_model.py rmbg-2.0

echo
echo "=== Listo. Para usar la app: doble clic en Kamiru.command ==="
read -r -p "Enter para cerrar..."
