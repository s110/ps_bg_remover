#!/bin/bash
# Kamiru - lanzador de doble clic (macOS)
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    echo "Primera vez: falta instalar. Corriendo setup_macos.command..."
    bash setup_macos.command
    [ -x venv/bin/python ] || exit 1
fi
exec ./venv/bin/python -m kamiru.gui.app
