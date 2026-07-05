@echo off
setlocal
REM ============================================================
REM  Kamiru - build del ejecutable autocontenido (Windows)
REM  Produce dist\Kamiru\Kamiru.exe y Kamiru-win64.zip.
REM  Con GPU NVIDIA presente empaqueta PyTorch CUDA 12.8 (el
REM  build GPU pesa varios GB: distribuirlo por USB/red local).
REM ============================================================
cd /d "%~dp0.."

REM --- uv para instalar rapido (se bootstrapea con pip si falta)
where uv >nul 2>&1
if %errorlevel%==0 (
    set "UV=uv"
) else (
    py -3.12 -m pip install -q uv || python -m pip install -q uv
    set "UV=py -3.12 -m uv"
)

%UV% venv build-env --python 3.12 || (echo ERROR creando build-env & exit /b 1)
set "PY=build-env\Scripts\python.exe"

where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo Empaquetando con PyTorch CUDA 12.8...
    %UV% pip install --python %PY% torch torchvision --index-url https://download.pytorch.org/whl/cu128
) else (
    echo Empaquetando con PyTorch CPU...
    %UV% pip install --python %PY% torch torchvision
)
%UV% pip install --python %PY% -e . pyinstaller

%PY% -m PyInstaller packaging\kamiru.spec --noconfirm --distpath dist --workpath build
if errorlevel 1 (echo ERROR en PyInstaller & exit /b 1)

echo Comprimiendo dist\Kamiru...
powershell -NoLogo -Command "Compress-Archive -Force -Path dist\Kamiru -DestinationPath Kamiru-win64.zip"
echo.
echo Listo: dist\Kamiru\Kamiru.exe  y  Kamiru-win64.zip
echo Al primer uso la app descarga el modelo a la carpeta models\ junto al exe.
pause
