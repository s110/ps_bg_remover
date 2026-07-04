@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  Kamiru - instalacion en Windows (correr UNA vez, Sebastian)
REM  Crea el venv, instala PyTorch CUDA para la RTX 5070 Ti,
REM  instala la app y descarga el modelo RMBG-2.0.
REM ============================================================
cd /d "%~dp0"

echo.
echo === Kamiru: instalacion ===
echo.

REM --- 1. Encontrar Python 3.10-3.12 (RMBG-2.0 no soporta 3.13)
set "PYCMD="
for %%V in (3.12 3.11 3.10) do (
    if not defined PYCMD (
        py -%%V -c "print()" >nul 2>&1 && set "PYCMD=py -%%V"
    )
)
if not defined PYCMD (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
    echo ERROR: No se encontro Python 3.10, 3.11 o 3.12.
    echo Instala Python 3.12 desde https://www.python.org/downloads/
    echo IMPORTANTE: marcar "Add Python to PATH" al instalar.
    pause
    exit /b 1
)
echo Usando: %PYCMD%

REM --- 2. Crear el entorno virtual
if not exist venv (
    %PYCMD% -m venv venv || (echo ERROR creando venv & pause & exit /b 1)
)
set "PIP=venv\Scripts\python.exe -m pip"
%PIP% install --upgrade pip

REM --- 3. PyTorch: CUDA 12.8 si hay GPU NVIDIA (5070 Ti = Blackwell), CPU si no
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo Instalando PyTorch con CUDA 12.8 ^(GPU NVIDIA detectada^)...
    %PIP% install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || (
        echo ERROR instalando PyTorch CUDA & pause & exit /b 1
    )
) else (
    echo AVISO: no se detecto GPU NVIDIA. Instalando PyTorch para CPU ^(mas lento^).
    %PIP% install torch torchvision
)

REM --- 4. Instalar la app y sus dependencias
%PIP% install -e . || (echo ERROR instalando dependencias & pause & exit /b 1)

REM --- 5. Verificar GPU
venv\Scripts\python.exe -c "import torch; ok=torch.cuda.is_available(); print('CUDA disponible:', ok); print('GPU:', torch.cuda.get_device_name(0) if ok else 'ninguna (se usara CPU, mas lento)')"

REM --- 6. Token de HuggingFace (RMBG-2.0 es un repo con registro)
if not exist models mkdir models
if not exist models\hf_token.txt (
    echo.
    echo RMBG-2.0 requiere una cuenta GRATIS de HuggingFace:
    echo   1. Crear cuenta:      https://huggingface.co/join
    echo   2. Aceptar licencia:  https://huggingface.co/briaai/RMBG-2.0
    echo   3. Crear token Read:  https://huggingface.co/settings/tokens
    echo.
    set /p HFTOK="Pega aqui el token (o Enter para usar el modelo alternativo BiRefNet): "
    if defined HFTOK (echo !HFTOK!)> models\hf_token.txt
)

REM --- 7. Descargar y cachear el modelo (una sola vez)
echo.
echo Descargando el modelo (~1 GB, solo esta vez)...
venv\Scripts\python.exe scripts\download_model.py rmbg-2.0 || (
    echo ERROR descargando el modelo. Revisa la conexion y reintenta.
    pause
    exit /b 1
)

echo.
echo === Listo. Para usar la app: doble clic en Kamiru.bat ===
echo (Puedes crear un acceso directo de Kamiru.bat en el escritorio de Kamila)
pause
