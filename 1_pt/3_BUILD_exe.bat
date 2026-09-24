@echo off
cd /d "%~dp0"
chcp 65001 > nul

title ANPR Monitor - Компіляція EXE
echo ========================================================
echo   Компіляція ANPR Video Monitor у виконуваний файл (.exe)
echo ========================================================
echo.

REM 1. Перевірка віртуального оточення
if not exist ".venv" (
    echo [ПОМИЛКА] Віртуальне оточення .venv не знайдено!
    echo Спочатку налаштуйте оточення через run.bat.
    pause
    exit /b 1
)

REM 2. Перевірка наявності файлу моделі
if not exist "model\best.pt" (
    echo [ПОМИЛКА] Файл model\best.pt відсутній!
    echo Перед компіляцією переконайтеся, що ваги розміщені у папці model.
    pause
    exit /b 1
)

REM 3. Перевірка та встановлення PyInstaller
echo Перевірка PyInstaller у .venv...
".venv\Scripts\python.exe" -m pip show pyinstaller >nul 2>nul
if errorlevel 1 (
    echo Встановлення PyInstaller...
    ".venv\Scripts\python.exe" -m pip install pyinstaller
    if errorlevel 1 (
        echo [ПОМИЛКА] Не вдалося встановити PyInstaller.
        pause
        exit /b 1
    )
)

REM 4. Видалення проблемного пакета polars (якщо є)
echo Очищення несумісних пакетів (polars)...
".venv\Scripts\python.exe" -m pip uninstall -y polars >nul 2>nul

echo.
echo ========================================================
echo   Початок збірки... (Це може зайняти 1-3 хвилини)
echo ========================================================
echo.

REM 5. Компіляція проєкту з повним збором залежностей ШІ
".venv\Scripts\python.exe" -m PyInstaller ^
    --name "ANPR_Monitor" ^
    --noconsole ^
    --onedir ^
    --noupx ^
    --clean ^
    --collect-all ultralytics ^
    --collect-all torchvision ^
    --copy-metadata ultralytics ^
    --copy-metadata torchvision ^
    --copy-metadata torch ^
    --hidden-import yolov5 ^
    --exclude-module polars ^
    app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ПОМИЛКА] Збірка завершилася невдачею.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo   Копіювання моделі, конфігурації та ресурсів у збірку...
echo ========================================================

REM 6. Створення папки model та копіювання файлу ваг
if not exist "dist\ANPR_Monitor\model" mkdir "dist\ANPR_Monitor\model"
copy /y "model\best.pt" "dist\ANPR_Monitor\model\best.pt" >nul

REM 7. Обов'язкове копіювання config.ini (та config.json якщо залишився)
if exist "config.ini" copy /y "config.ini" "dist\ANPR_Monitor\config.ini" >nul
if exist "config.json" copy /y "config.json" "dist\ANPR_Monitor\config.json" >nul
if exist "README.md" copy /y "README.md" "dist\ANPR_Monitor\" >nul

echo.
echo Очищення тимчасових файлів збірки...
if exist "build" rd /s /q "build"
if exist "ANPR_Monitor.spec" del /f /q "ANPR_Monitor.spec"

echo.
echo ========================================================
echo   Збірка успішно завершена!
echo ========================================================
echo.
echo Готовий додаток знаходиться тут:
echo dist\ANPR_Monitor\ANPR_Monitor.exe
echo.
pause
