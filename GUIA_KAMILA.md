# Kamiru — guía rápida para Kamila 🎨

## Abrir la app

- **Windows:** doble clic en **Kamiru** (el acceso directo del escritorio).
- **Mac:** doble clic en **Kamiru.command**.

## Recortar una tanda de fotos (4 pasos)

1. **Arrastra la carpeta** con tus fotos a la ventana — o fotos sueltas, o
   usa los botones *Carpeta…* / *Fotos…*.
2. En **"Guardar en"** elige la carpeta donde quieres los recortes.
3. Elige el **Modo**:
   - **Conjunto** → todo lo que hay en la foto queda junto en un solo
     archivo transparente.
   - **Individual** → cada pieza sale en su propio archivo
     (`foto_01.png`, `foto_02.png`, …).
4. Elige el **Formato**:
   - **PNG** → para web, Canva, etc. (el normal, ya viene elegido).
   - **TIFF** → para imprenta.
   - **PSD por capas** → para Photoshop: cada pieza en su capa.

¿Quieres que los archivos lleven una palabra tuya? Escríbela en **Sufijo**:
con `recorte` los archivos salen como `foto_recorte.png`.

**Tip:** aprieta **Vista previa** primero — recorta solo la primera foto y
te la muestra sobre cuadritos para que veas cómo va a quedar antes de
procesar todo.

Aprieta **Procesar** y espera la barra. Al final sale un **resumen** (cuántos
archivos se exportaron; si una foto da error se salta sola y el resto sigue)
y el botón **Abrir salida** te lleva directo a tus recortes.

La app **recuerda tu configuración**: mañana se abre con las mismas carpetas
y opciones de hoy.

## La carpeta «revisar»

La app revisa cada recorte sola: mira si el borde salió raro, si se coló el
fondo, y hasta recorta la foto con **dos modelos distintos** para comparar.
Si un recorte le parece dudoso, lo guarda igual pero dentro de la carpeta
**`revisar/`** en tu salida, y el resumen te dice el motivo de cada uno.

O sea: lo que está **fuera** de `revisar/` ya está bien — no tienes que
calar todas las fotos una por una, solo mirar las poquitas de esa carpeta.

## Tips

- No importa si el fondo es madera, verde o cartulina: la IA lo quita igual,
  todo en una sola pasada.
- Deja **espacio entre las piezas** al fotografiar: si dos piezas se tocan,
  saldrán juntas como una sola.
- Las motitas de polvo se descartan solas.
- Nunca se sobrescribe nada: si el archivo ya existe, el nuevo sale como
  `foto-1.png`.
- ¿Un borde salió mordido? Ábrelo de nuevo con Sebastian: en "Avanzado" se
  puede subir la resolución de proceso a 1536 o 2048.

## Si algo no funciona

- ¿La ventana no abre? Avísale a Sebastian (se instala una sola vez con
  `setup_windows.bat`).
- ¿Una foto salió mal recortada? Mándasela a Sebastian; hay opciones
  avanzadas (otro modelo, croma por color) para casos difíciles.
