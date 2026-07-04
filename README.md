# Kamiru — recorte de fondos por lotes con IA

App local (Windows y macOS) que toma una carpeta de fotos y extrae los
objetos de su fondo con calidad de borde alta, usando **RMBG-2.0** (BRIA,
arquitectura BiRefNet). Pensada para piezas físicas con textura (papel
arrugado, obra dimensional) sobre fondos que cambian entre tandas (madera,
green screen, cartulina). Todo corre en la máquina: **sin nube, sin créditos,
sin terminal** para el uso diario.

> Uso **no comercial** (licencia de RMBG-2.0).

---

## Instalación (una sola vez — guía para Sebastian)

### Windows (PC con RTX 5070 Ti)

1. Instala **Python 3.12** desde <https://www.python.org/downloads/>
   marcando **"Add Python to PATH"**. (RMBG-2.0 no soporta 3.13; sirve
   3.10–3.12.)
2. Descarga o clona este repositorio donde quieras que viva la app.
3. Doble clic en **`setup_windows.bat`**. El script:
   - crea el entorno virtual (`venv/`),
   - instala PyTorch con **CUDA 12.8** (la 5070 Ti es Blackwell; wheels
     `cu128`) o CPU si no hay GPU,
   - instala la app y sus dependencias,
   - verifica `torch.cuda.is_available()`,
   - pide el **token de HuggingFace** (ver abajo) y descarga el modelo
     (~1 GB) a `models/` — **una sola vez**.
4. Crea un acceso directo de **`Kamiru.bat`** en el escritorio de Kamila.

### macOS

1. Ten Python 3.10–3.12 (`brew install python@3.12` o python.org).
2. Doble clic en **`setup_macos.command`** (si Gatekeeper se queja:
   clic derecho → Abrir; o `xattr -d com.apple.quarantine *.command`).
   Instala PyTorch con soporte **MPS** (GPU Apple) y cachea el modelo.
3. Para el día a día: doble clic en **`Kamiru.command`**.

### Token de HuggingFace (para RMBG-2.0)

RMBG-2.0 es un repositorio *gated*: gratis, pero con registro.

1. Crear cuenta: <https://huggingface.co/join>
2. Aceptar la licencia: <https://huggingface.co/briaai/RMBG-2.0>
3. Crear un token (Read): <https://huggingface.co/settings/tokens>
4. Pegarlo cuando el setup lo pida (queda en `models/hf_token.txt`).

Si se salta este paso, el setup descarga automáticamente **BiRefNet** (misma
arquitectura, sin registro) y lo deja como modelo por defecto; se puede
volver a RMBG-2.0 después pegando el token y re-corriendo
`scripts/download_model.py`.

### Verificación rápida

```bash
# corrida de ejemplo sobre fotos sintéticas de prueba (genera samples/)
venv/bin/python scripts/sample_run.py        # macOS/Linux
venv\Scripts\python scripts\sample_run.py    # Windows
```

Los tests unitarios (no requieren GPU ni modelo):

```bash
venv/bin/python -m pip install pytest psd-tools
venv/bin/python -m pytest tests/
```

---

## Uso diario (Kamila)

Ver la guía de una página: **[GUIA_KAMILA.md](GUIA_KAMILA.md)**. En corto:
doble clic en Kamiru → arrastrar la carpeta de fotos → elegir dónde guardar →
Procesar.

---

## Qué hace exactamente

Por cada imagen de la carpeta (JPG, PNG, TIFF, HEIC, WebP, BMP):

1. Carga respetando la **orientación EXIF** y el **perfil ICC**.
2. Infiere el alfa con **RMBG-2.0** a la resolución de proceso (1024 por
   defecto; 1536/2048 para más detalle de borde) y lo reescala al tamaño
   original — el RGBA final es a **resolución completa**.
3. Según el modo:
   - **Conjunto:** todos los objetos de la foto en un archivo, recortado al
     contenido.
   - **Individual:** separa objetos con `connectedComponentsWithStats`,
     descarta motas con el **filtro de área mínima**, y exporta cada pieza
     recortada a su bounding box **conservando el alfa suave** (limitado a la
     pieza para no arrastrar bordes de vecinas).
4. Exporta **sin sobrescribir** (`foto.png`, `foto-1.png`, ...):
   - **PNG-24** transparente (por defecto), con ICC embebido.
   - **TIFF con alfa** 8-bit o **16-bit** (deflate, ICC embebido).
   - **PSD**: en modo individual, **un PSD por foto con cada pieza en su
     propia capa** (en su posición original); en conjunto, PSD de una capa.
     Compatible con Photoshop, Affinity, Krita y GIMP.

Nomenclatura: conjunto → `nombre_origen.ext`; individual →
`nombre_origen_01.ext`, `_02`, ... con relleno de ceros.

Una imagen que falla **se registra y se salta** sin abortar el lote; al final
hay resumen de exportados y errores, y queda log por corrida en `logs/`.

## Opciones avanzadas (GUI → "Avanzado", o CLI)

| Opción | Qué hace |
|---|---|
| Motor / modelo | `rmbg-2.0` (default), `birefnet`, `birefnet-hr`, `ben2` para A/B de nitidez de borde, o **croma por color** |
| Resolución de proceso | 1024 / 1536 / 2048 — subir si un borde fino se pierde |
| Área mínima (px²) | filtro de motas y polvo (default 400) |
| Separar piezas que se tocan | watershed sobre la transformada de distancia (experimental) |
| Croma | color de fondo automático (muestreo de esquinas) o elegido a mano — útil como camino alterno en green screen / cartulina |

### CLI (mismo core que la GUI)

```bash
kamiru fotos/ salida/ --modo individual --formato png --area-minima 400
kamiru fotos/ salida/ --modo individual --formato psd          # PSD por capas
kamiru fotos/ salida/ --modelo birefnet-hr --resolucion 2048   # A/B de borde
kamiru fotos/ salida/ --motor croma --chroma-color 00ff00      # green screen
kamiru fotos/ salida/ --separar-tocandose                      # watershed
```

## Estructura

```
src/kamiru/            paquete (core reutilizable + GUI + CLI)
  core/matting.py        motores RMBG-2.0 / BiRefNet / BEN2 (cutout → RGBA)
  core/separation.py     connected components + watershed + filtro de área
  core/chroma.py         croma HSV (auto por esquinas o color manual)
  core/export.py         PNG / TIFF 8-16 bit / PSD + nomenclatura
  core/psd_writer.py     escritor PSD por capas (RLE), puro numpy
  core/pipeline.py       lote, progreso, errores, resumen, logging
  gui/app.py             ventana de arrastrar y soltar
  cli.py                 línea de comandos
scripts/               download_model.py, make_samples.py, sample_run.py
tests/                 tests sin GPU: separación, export, PSD, pipeline
models/                cache local del modelo (no se re-descarga)
logs/                  un log por corrida
```

## Notas y decisiones

- **Piezas que se tocan:** por defecto salen como un solo objeto (regla
  práctica: dejar espacio entre piezas al fotografiar). La opción
  "separar piezas que se tocan" usa watershed clásico; funciona bien con
  piezas convexas unidas por cuellos. El spec mencionaba SAM3 para esto:
  no existe como modelo público, por lo que el watershed cubre el caso hoy
  y el motor es intercambiable si aparece algo mejor.
- **Sombras:** RMBG-2.0 las excluye del matte; además el recorte usa el alfa,
  no el color, así que la sombra no viaja al archivo final.
- **GPU:** se detecta CUDA → MPS → CPU al iniciar, con aviso si no hay GPU.
  En la 5070 Ti el objetivo es < 3 s por foto (el modelo corre en fp16).
- **Cache:** `HF_HOME` apunta a `models/` dentro de la app; el modelo se
  descarga en el setup y no vuelve a bajarse.
