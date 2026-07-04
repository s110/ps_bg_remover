"""GUI de Kamiru: ventana simple de arrastrar y soltar.

Controles principales (los que ve Kamila): carpeta de entrada, carpeta de
salida, modo (Conjunto/Individual), formato (PNG/TIFF/TIFF 16-bit/PSD),
botón Procesar, barra de progreso y resumen. Un panel «Avanzado» plegado
esconde el resto (motor, resolución, área mínima, separación de piezas que
se tocan, croma).
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

from .. import APP_NAME, __version__
from ..core.imageio import is_supported, list_images
from ..core.pipeline import BatchOptions, run_batch, setup_batch_logging
from ..paths import logs_dir

log = logging.getLogger("kamiru.gui")

MODEL_CHOICES = ["rmbg-2.0", "birefnet", "birefnet-hr", "ben2", "croma (por color)"]
FORMAT_LABELS = {
    "PNG (transparente)": "png",
    "TIFF con alfa (8-bit)": "tiff",
    "TIFF con alfa (16-bit)": "tiff16",
    "PSD por capas": "psd",
}
RESOLUTIONS = ["1024", "1536", "2048"]


def _try_dnd_root():
    """Ventana raíz con drag-and-drop si tkinterdnd2 está disponible."""
    try:
        from tkinterdnd2 import TkinterDnD

        return TkinterDnD.Tk(), True
    except Exception:
        return tk.Tk(), False


class KamiruApp:
    def __init__(self) -> None:
        self.root, self.has_dnd = _try_dnd_root()
        self.root.title(f"{APP_NAME} — recorte de fondos {__version__}")
        self.root.minsize(560, 460)

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        from ..core.matting import resolve_default_model

        self.mode = tk.StringVar(value="conjunto")
        self.format_label = tk.StringVar(value="PNG (transparente)")
        self.model = tk.StringVar(value=resolve_default_model())
        self.resolution = tk.StringVar(value="1024")
        self.min_area = tk.StringVar(value="400")
        self.split_touching = tk.BooleanVar(value=False)
        self.chroma_auto = tk.BooleanVar(value=True)
        self.chroma_color: tuple[int, int, int] | None = None
        self.dropped_files: list[Path] = []

        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()
        self._matter = None
        self._matter_key: tuple | None = None

        self._build_ui()
        setup_batch_logging(logs_dir())
        self._show_device_status()
        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        drop = tk.Label(
            main,
            text=("Arrastra aquí una carpeta o fotos\n(o usa los botones de abajo)"
                  if self.has_dnd else "Elige las carpetas con los botones"),
            relief="groove", height=4, bg="#f2f0ec", fg="#555",
        )
        drop.pack(fill="x", **pad)
        if self.has_dnd:
            from tkinterdnd2 import DND_FILES

            drop.drop_target_register(DND_FILES)
            drop.dnd_bind("<<Drop>>", self._on_drop)

        row1 = ttk.Frame(main); row1.pack(fill="x", **pad)
        ttk.Label(row1, text="Fotos (entrada):", width=16).pack(side="left")
        ttk.Entry(row1, textvariable=self.input_dir).pack(side="left", fill="x", expand=True)
        ttk.Button(row1, text="Elegir…", command=self._pick_input).pack(side="left", padx=(6, 0))

        row2 = ttk.Frame(main); row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Guardar en:", width=16).pack(side="left")
        ttk.Entry(row2, textvariable=self.output_dir).pack(side="left", fill="x", expand=True)
        ttk.Button(row2, text="Elegir…", command=self._pick_output).pack(side="left", padx=(6, 0))

        row3 = ttk.Frame(main); row3.pack(fill="x", **pad)
        ttk.Label(row3, text="Modo:", width=16).pack(side="left")
        ttk.Radiobutton(row3, text="Conjunto (todo en un archivo)",
                        variable=self.mode, value="conjunto").pack(side="left")
        ttk.Radiobutton(row3, text="Individual (un archivo por pieza)",
                        variable=self.mode, value="individual").pack(side="left", padx=(10, 0))

        row4 = ttk.Frame(main); row4.pack(fill="x", **pad)
        ttk.Label(row4, text="Formato:", width=16).pack(side="left")
        ttk.Combobox(row4, textvariable=self.format_label, state="readonly",
                     values=list(FORMAT_LABELS)).pack(side="left", fill="x", expand=True)

        # ------- panel avanzado plegable
        self._adv_visible = tk.BooleanVar(value=False)
        toggle = ttk.Checkbutton(main, text="Avanzado", style="Toolbutton",
                                 variable=self._adv_visible, command=self._toggle_advanced)
        toggle.pack(anchor="w", padx=10)
        self.adv = ttk.LabelFrame(main, text="Opciones avanzadas")

        a1 = ttk.Frame(self.adv); a1.pack(fill="x", **pad)
        ttk.Label(a1, text="Motor:", width=16).pack(side="left")
        ttk.Combobox(a1, textvariable=self.model, state="readonly",
                     values=MODEL_CHOICES, width=18).pack(side="left")
        ttk.Label(a1, text="  Resolución:").pack(side="left")
        ttk.Combobox(a1, textvariable=self.resolution, state="readonly",
                     values=RESOLUTIONS, width=6).pack(side="left")

        a2 = ttk.Frame(self.adv); a2.pack(fill="x", **pad)
        ttk.Label(a2, text="Área mínima (px):", width=16).pack(side="left")
        ttk.Entry(a2, textvariable=self.min_area, width=8).pack(side="left")
        ttk.Checkbutton(a2, text="Separar piezas que se tocan (experimental)",
                        variable=self.split_touching).pack(side="left", padx=(14, 0))

        a3 = ttk.Frame(self.adv); a3.pack(fill="x", **pad)
        ttk.Label(a3, text="Croma:", width=16).pack(side="left")
        ttk.Checkbutton(a3, text="Color de fondo automático (esquinas)",
                        variable=self.chroma_auto).pack(side="left")
        ttk.Button(a3, text="Elegir color…", command=self._pick_chroma_color).pack(
            side="left", padx=(10, 0))
        self._chroma_swatch = tk.Label(a3, text="  auto  ", relief="sunken")
        self._chroma_swatch.pack(side="left", padx=(8, 0))

        # ------- acción, progreso, resumen
        actions = ttk.Frame(main); actions.pack(fill="x", **pad)
        self._actions_frame = actions
        self.run_btn = ttk.Button(actions, text="Procesar", command=self._start)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(actions, text="Cancelar", command=self._cancel_run,
                                     state="disabled")
        self.cancel_btn.pack(side="left", padx=(8, 0))
        self.device_lbl = ttk.Label(actions, text="", foreground="#666")
        self.device_lbl.pack(side="right")

        self.progress = ttk.Progressbar(main, mode="determinate")
        self.progress.pack(fill="x", **pad)
        self.status = ttk.Label(main, text="Lista.")
        self.status.pack(fill="x", padx=10)

        self.summary_box = tk.Text(main, height=7, state="disabled",
                                   bg="#faf9f7", relief="flat", wrap="word")
        self.summary_box.pack(fill="both", expand=True, padx=10, pady=(4, 10))

    def _toggle_advanced(self) -> None:
        if self._adv_visible.get():
            self.adv.pack(fill="x", padx=10, pady=4, before=self._actions_frame)
        else:
            self.adv.pack_forget()

    def _show_device_status(self) -> None:
        def work():
            try:
                from ..core.device import detect_device

                info = detect_device()
                self._queue.put(("device", info))
            except Exception as exc:  # torch aún no instalado, etc.
                self._queue.put(("device_error", str(exc)))
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------- eventos
    def _on_drop(self, event) -> None:
        paths = [Path(p) for p in self.root.tk.splitlist(event.data)]
        dirs = [p for p in paths if p.is_dir()]
        files = [p for p in paths if p.is_file() and is_supported(p)]
        if dirs:
            self.input_dir.set(str(dirs[0]))
            self.dropped_files = []
            self._set_status(f"Carpeta: {dirs[0].name} ({len(list_images(dirs[0]))} fotos)")
        elif files:
            self.dropped_files = files
            self.input_dir.set(f"{len(files)} foto(s) sueltas")
            self._set_status(f"{len(files)} foto(s) para procesar")

    def _pick_input(self) -> None:
        d = filedialog.askdirectory(title="Carpeta con las fotos")
        if d:
            self.input_dir.set(d)
            self.dropped_files = []

    def _pick_output(self) -> None:
        d = filedialog.askdirectory(title="Carpeta donde guardar los recortes")
        if d:
            self.output_dir.set(d)

    def _pick_chroma_color(self) -> None:
        color = colorchooser.askcolor(title="Color del fondo a quitar")
        if color and color[0]:
            self.chroma_color = tuple(int(c) for c in color[0])
            self.chroma_auto.set(False)
            hexcol = "#%02x%02x%02x" % self.chroma_color
            self._chroma_swatch.configure(text="        ", bg=hexcol)

    # ------------------------------------------------------------- proceso
    def _collect_inputs(self):
        if self.dropped_files:
            return self.dropped_files
        raw = self.input_dir.get().strip()
        if not raw:
            return None
        p = Path(raw)
        if not p.is_dir():
            return None
        return p

    def _build_matter(self):
        choice = self.model.get()
        if choice.startswith("croma"):
            from ..core.chroma import ChromaMatter

            key = None if self.chroma_auto.get() else self.chroma_color
            return ChromaMatter(key_color=key)
        key = (choice, int(self.resolution.get()))
        if self._matter is not None and self._matter_key == key:
            return self._matter  # ya cargado, no re-descargar ni re-cargar
        from ..core.matting import load_matter

        self._queue.put(("status", f"Cargando modelo {choice}… (la primera vez descarga ~1 GB)"))
        matter = load_matter(choice, process_resolution=int(self.resolution.get()))
        self._matter, self._matter_key = matter, key
        return matter

    def _start(self) -> None:
        inputs = self._collect_inputs()
        if inputs is None:
            messagebox.showwarning(APP_NAME, "Elige una carpeta de fotos válida (o arrastra fotos).")
            return
        out = self.output_dir.get().strip()
        if not out:
            messagebox.showwarning(APP_NAME, "Elige la carpeta donde guardar los recortes.")
            return
        try:
            min_area = int(self.min_area.get())
        except ValueError:
            messagebox.showwarning(APP_NAME, "El área mínima debe ser un número entero de píxeles.")
            return

        opts = BatchOptions(
            mode=self.mode.get(),
            fmt=FORMAT_LABELS[self.format_label.get()],
            min_area=min_area,
            split_touching=self.split_touching.get(),
        )
        self._cancel.clear()
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.configure(value=0)
        self._set_summary("")

        def work():
            try:
                matter = self._build_matter()
                summary = run_batch(
                    inputs, Path(out), matter, opts,
                    progress=lambda done, total, name: self._queue.put(
                        ("progress", done, total, name)),
                    cancel=self._cancel.is_set,
                )
                self._queue.put(("done", summary))
            except Exception as exc:
                log.exception("Fallo del lote")
                self._queue.put(("fatal", str(exc)))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _cancel_run(self) -> None:
        self._cancel.set()
        self._set_status("Cancelando al terminar la foto actual…")

    # ------------------------------------------------------------- cola UI
    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, done, total, name = msg
                    self.progress.configure(maximum=max(total, 1), value=done)
                    if name:
                        self._set_status(f"Procesando {done + 1} de {total}: {name}")
                elif kind == "status":
                    self._set_status(msg[1])
                elif kind == "device":
                    info = msg[1]
                    self.device_lbl.configure(text=info.name)
                    if info.warning:
                        self._set_status(f"⚠ {info.warning}")
                elif kind == "device_error":
                    self.device_lbl.configure(text="PyTorch no disponible")
                elif kind == "done":
                    summary = msg[1]
                    self.run_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    self.progress.configure(value=self.progress["maximum"])
                    self._set_status("Terminado.")
                    self._set_summary(summary.text())
                    messagebox.showinfo(
                        f"{APP_NAME} — resumen",
                        summary.text() or "No había imágenes para procesar.",
                    )
                elif kind == "fatal":
                    self.run_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    self._set_status("Error.")
                    messagebox.showerror(APP_NAME, f"El lote no pudo iniciarse:\n{msg[1]}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def _set_summary(self, text: str) -> None:
        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("1.0", text)
        self.summary_box.configure(state="disabled")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    KamiruApp().run()


if __name__ == "__main__":
    main()
