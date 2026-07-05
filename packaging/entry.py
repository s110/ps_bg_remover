"""Entry point del ejecutable PyInstaller (Kamiru.exe / Kamiru.app)."""

import multiprocessing

multiprocessing.freeze_support()

from kamiru.gui.app import main

if __name__ == "__main__":
    main()
