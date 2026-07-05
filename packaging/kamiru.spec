# -*- mode: python ; coding: utf-8 -*-
# Spec de PyInstaller para el ejecutable autocontenido de Kamiru.
# Modo onedir: la carpeta dist/Kamiru contiene Kamiru(.exe) y todo lo demás;
# models/ y logs/ se crean junto al ejecutable en el primer uso.

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
# transformers carga código remoto (BiRefNet) dinámicamente: hay que llevar
# el paquete completo. tkinterdnd2 trae los binarios de tkdnd como data.
for pkg in ("transformers", "tkinterdnd2", "timm", "kornia", "einops",
            "safetensors", "tokenizers"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += [
    "PIL._tkinter_finder",
    "pillow_heif",
    "cv2",
    "tifffile",
    "scipy",
    "scipy.signal",
    "huggingface_hub",
]

a = Analysis(
    ["entry.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "psd_tools"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="Kamiru",
    console=False,          # sin terminal: es la app de Kamila
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="Kamiru",
    upx=False,
)
