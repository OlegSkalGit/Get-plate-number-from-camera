@echo off
cd /d "%~dp0"
chcp 65001 > nul

if not exist ".venv\Scripts\python.exe" (
    echo [ПОМИЛКА] Віртуальне оточення .venv не знайдено! Спочатку запустіть 1_SETUP.bat.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "app.py"
