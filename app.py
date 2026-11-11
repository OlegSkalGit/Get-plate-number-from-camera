import os
import sys

# Коренева точка запуску програми (підтримка PyInstaller EXE та запуску зі скрипту)
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.chdir(BASE_DIR)

from scripts.ui.main_window import main


if __name__ == "__main__":
    main()
