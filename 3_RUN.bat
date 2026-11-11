@echo off
cd /d "%~dp0"
chcp 65001 > nul

if not exist ".venv\Scripts\pythonw.exe" (
    echo [ПОМИЛКА] Віртуальне оточення .venv не знайдено! Спочатку запустіть 1_SETUP.bat.
    pause
    exit /b 1
)

start "" ".venv\Scripts\pythonw.exe" "app.py"
