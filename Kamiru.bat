@echo off
REM Kamiru - lanzador de doble clic (Windows)
cd /d "%~dp0"
if not exist venv\Scripts\pythonw.exe (
    echo Primera vez: falta instalar. Corriendo setup_windows.bat...
    call setup_windows.bat
    if not exist venv\Scripts\pythonw.exe exit /b 1
)
start "" venv\Scripts\pythonw.exe -m kamiru.gui.app
