@echo off
cd /d "%~dp0"
chcp 65001 > nul

title ANPR Monitor - Компіляція EXE (ONNX)
echo ========================================================
echo   Компіляція ANPR Video Monitor (Легкий ONNX EXE)
echo ========================================================
echo.

if not exist ".venv" (
    echo [ПОМИЛКА] .venv не знайдено. Спочатку запустіть run.bat.
    pause
    exit /b 1
)

if not exist "model\best.onnx" (
    echo [ПОМИЛКА] model\best.onnx відсутній!
    pause
    exit /b 1
)

echo Перевірка PyInstaller...
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
    --collect-all onnxruntime ^
    app.py

if %ERRORLEVEL% neq 0 (
    echo [ПОМИЛКА] Збірка зазнала невдачі.
    pause
    exit /b 1
)

echo Копіювання ресурсів...
if not exist "dist\ANPR_Monitor\model" mkdir "dist\ANPR_Monitor\model"
copy /y "model\best.onnx" "dist\ANPR_Monitor\model\best.onnx" >nul
if exist "config.ini" copy /y "config.ini" "dist\ANPR_Monitor\config.ini" >nul
if exist "README.md" copy /y "README.md" "dist\ANPR_Monitor\" >nul

if exist "build" rd /s /q "build"
if exist "ANPR_Monitor.spec" del /f /q "ANPR_Monitor.spec"

echo.
echo ========================================================
echo   Збірка успішно завершена! Розмір зменшено до ~70 МБ.
echo ========================================================
echo Файл програми: dist\ANPR_Monitor\ANPR_Monitor.exe
echo.
pause
