@echo off
cd /d "%~dp0"
chcp 65001 > nul

title ANPR Multi-Cam NVR - Компіляція EXE (ONNX)
echo ========================================================
echo   Компіляція ANPR Multi-Cam NVR (ONNX EXE)
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ПОМИЛКА] .venv не знайдено. Спочатку запустіть 1_SETUP.bat.
    pause
    exit /b 1
)

if not exist "models\best.onnx" (
    echo [ПОМИЛКА] models\best.onnx відсутній!
    pause
    exit /b 1
)

if not exist "models\cct_xs_v2_global.onnx" (
    echo [ПОМИЛКА] models\cct_xs_v2_global.onnx відсутній!
    pause
    exit /b 1
)

if not exist "models\cct_xs_v2_global_plate_config.yaml" (
    echo [ПОМИЛКА] models\cct_xs_v2_global_plate_config.yaml відсутній!
    pause
    exit /b 1
)

if not exist "models\yolo11n.onnx" (
    echo [ПОМИЛКА] models\yolo11n.onnx відсутній!
    pause
    exit /b 1
)

if not exist "models\scrfd_face_detector.onnx" (
    echo [УВАГА] models\scrfd_face_detector.onnx не знайдено ^(модуль облич буде недоступний^).
)
if not exist "models\face_recognizer.onnx" (
    echo [УВАГА] models\face_recognizer.onnx не знайдено ^(модуль облич буде недоступний^).
)

echo Перевірка psutil та PyInstaller...
".venv\Scripts\python.exe" -c "import psutil" >nul 2>nul
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install psutil
)
".venv\Scripts\python.exe" -m pip show pyinstaller >nul 2>nul
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install pyinstaller
)

echo Початок збірки...
".venv\Scripts\python.exe" -m PyInstaller ^
    --name "ANPR_Monitor" ^
    --noconsole ^
    --onedir ^
    --noupx ^
    --clean ^
    --paths "." ^
    --collect-submodules scripts.core ^
    --collect-submodules scripts.engine ^
    --collect-submodules scripts.ui ^
    --collect-all onnxruntime ^
    --collect-all fast_alpr ^
    --collect-all fast_plate_ocr ^
    --collect-all imageio_ffmpeg ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    --hidden-import PIL ^
    --hidden-import PIL.Image ^
    --hidden-import PIL.ImageTk ^
    --hidden-import PIL.ImageDraw ^
    --hidden-import PIL.ImageFont ^
    --hidden-import yaml ^
    --hidden-import psutil ^
    --hidden-import winreg ^
    --hidden-import uuid ^
    app.py

if %ERRORLEVEL% neq 0 (
    echo [ПОМИЛКА] Збірка зазнала невдачі.
    pause
    exit /b 1
)

echo Копіювання ресурсів і зовнішніх модулів...
if not exist "dist\ANPR_Monitor\models" mkdir "dist\ANPR_Monitor\models"
xcopy /e /y /i "models\*" "dist\ANPR_Monitor\models\" >nul
if not exist "dist\ANPR_Monitor\scripts\modules" mkdir "dist\ANPR_Monitor\scripts\modules"
xcopy /e /y /i "scripts\modules\*" "dist\ANPR_Monitor\scripts\modules\" >nul
if exist "dist\ANPR_Monitor\scripts\modules\__pycache__" rd /s /q "dist\ANPR_Monitor\scripts\modules\__pycache__"
if exist "_redist_" (
    if not exist "dist\ANPR_Monitor\_redist_" mkdir "dist\ANPR_Monitor\_redist_"
    xcopy /e /y /i "_redist_\*" "dist\ANPR_Monitor\_redist_\" >nul
) else if exist "..\_redist_" (
    if not exist "dist\ANPR_Monitor\_redist_" mkdir "dist\ANPR_Monitor\_redist_"
    xcopy /e /y /i "..\_redist_\*" "dist\ANPR_Monitor\_redist_\" >nul
)
if exist "config.json" copy /y "config.json" "dist\ANPR_Monitor\config.json" >nul
if exist "config.ini" copy /y "config.ini" "dist\ANPR_Monitor\config.ini" >nul
if exist "README.md" copy /y "README.md" "dist\ANPR_Monitor\" >nul

if exist "build" rd /s /q "build"
if exist "ANPR_Monitor.spec" del /f /q "ANPR_Monitor.spec"

echo.
echo ========================================================
echo   Збірка успішно завершена!
echo ========================================================
echo Файл програми: dist\ANPR_Monitor\ANPR_Monitor.exe
echo Зовнішні модулі: dist\ANPR_Monitor\scripts\modules\
echo.
pause
