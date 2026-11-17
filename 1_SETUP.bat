@echo off
cd /d "%~dp0"
chcp 65001 > nul

setlocal enabledelayedexpansion

set "PY_VERSION=3.12.10"
set "PY_DIR_VER=312"

title ANPR Multi-Cam NVR - Налаштування оточення
echo ========================================================
echo   ANPR MULTI-CAM NVR - НАЛАШТУВАННЯ ОТОЧЕННЯ
echo ========================================================
echo.

set "PYTHON_CMD=python"

REM 1. Перевірка наявності Python у системі (ігноруємо заглушку Microsoft Store)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python НЕ знайдено у системі.
    if not exist "setup\python-%PY_VERSION%-amd64.exe" (
        echo Інсталятор Python не знайдено. Завантажуємо версію %PY_VERSION%...
        if not exist "setup" mkdir "setup"
        curl -L "https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-amd64.exe" -o "setup\python-%PY_VERSION%-amd64.exe"
        if !ERRORLEVEL! neq 0 (
            echo.
            echo ===============================================================================================
            echo Для ПЕРШОГО запуску програми необхідний інтернет для завантаження Python.
            echo ===============================================================================================
            echo.
            echo [Помилка] Не вдалося завантажити Python. Перевірте з'єднання з інтернетом.
            pause
            exit /b 1
        )
    )
    echo Інсталятор Python знайдено. Встановлюємо...
    start /wait "" ".\setup\python-%PY_VERSION%-amd64.exe" /quiet PrependPath=1 InstallAllUsers=0 Include_launcher=0 TargetDir="%LOCALAPPDATA%\Programs\Python\Python%PY_DIR_VER%"
    if !ERRORLEVEL! neq 0 (
        echo [Помилка] Не вдалося встановити Python.
        pause
        exit /b 1
    )
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python%PY_DIR_VER%\python.exe"
) else (
    echo [OK] Python знайдено у системі.
)

REM 2. Створення віртуального оточення
if not exist ".venv\Scripts\python.exe" (
    echo Віртуальне оточення .venv відсутнє. Створюємо...
    "!PYTHON_CMD!" -m venv .venv
    if !ERRORLEVEL! neq 0 (
        echo [Помилка] Не вдалося створити віртуальне оточення .venv.
        pause
        exit /b 1
    )
    echo [OK] Віртуальне оточення створено.
) else (
    echo [OK] Віртуальне оточення .venv знайдено.
)

REM 3. Перевірка та встановлення бібліотек комп'ютерного зору (типово CPU ONNX Runtime)
echo Перевірка необхідних модулів...
".venv\Scripts\python.exe" -c "import cv2, numpy, PIL, onnxruntime, yaml, fast_alpr, fast_plate_ocr, imageio_ffmpeg, psutil, av" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Встановлюємо або дозавантажуємо необхідні бібліотеки ^(типово CPU^)...
    echo ^(Завантаження OpenCV, ONNX Runtime, PyAV, FFmpeg та psutil може зайняти 1-2 хвилини^)
    echo.
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install opencv-python numpy pillow onnxruntime pyyaml fast-alpr fast-plate-ocr imageio-ffmpeg psutil av
    if !ERRORLEVEL! neq 0 (
        echo.
        echo ===============================================================================================
        echo [Помилка] Не вдалося встановити бібліотеки. Перевірте з'єднання з інтернетом.
        echo ===============================================================================================
        pause
        exit /b 1
    )
    echo [OK] Усі базові бібліотеки успішно встановлено.
) else (
    echo [OK] Усі базові бібліотеки присутні.
)

REM 4. Автопідготовка прискорення ONNX Runtime (у окремих папках ort_backends\ без перевстановлення)
echo.
echo --------------------------------------------------------
echo   Прискорення нейромереж (ONNX Runtime)
echo --------------------------------------------------------
echo Автоматична перевірка GPU та підготовка оптимального бекенду в ort_backends\...
".venv\Scripts\python.exe" -m scripts.core.hardware_info --ensure-optimal-ort .
if !ERRORLEVEL! neq 0 (
    echo [УВАГА] Використано базовий CPU-режим ONNX Runtime ^(перемикання також доступне у Налаштування - Залізо та графіка^).
)

echo.
echo ========================================================
echo   Налаштування завершено! Запускайте 3_RUN.bat
echo ========================================================
endlocal
