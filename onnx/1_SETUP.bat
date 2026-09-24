@echo off
cd /d "%~dp0"
chcp 65001 > nul

setlocal enabledelayedexpansion

set "PY_VERSION=3.12.10"
set "PY_DIR_VER=312"

title ANPR Video Monitor - Запуск
echo ========================================================
echo   ANPR VIDEO MONITOR (ДЕТЕКЦІЯ ТА ЗБІЛЬШЕННЯ НОМЕРІВ)
echo ========================================================
echo.

set "PYTHON_CMD=python"

REM 1. Перевірка наявності Python у системі
python --version >nul 2>nul
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
    set "PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python%PY_DIR_VER%\python.exe""
) else (
    echo [OK] Python знайдено у системі.
)

REM 2. Створення віртуального оточення
if not exist ".venv" (
    echo Віртуальне оточення .venv відсутнє. Створюємо...
    !PYTHON_CMD! -m venv .venv
    if !ERRORLEVEL! neq 0 (
        echo [Помилка] Не вдалося створити віртуальне оточення .venv.
        pause
        exit /b 1
    )
    echo [OK] Віртуальне оточення створено.
) else (
    echo [OK] Віртуальне оточення .venv знайдено.
)

REM 3. Перевірка та встановлення бібліотек комп'ютерного зору
echo Перевірка необхідних модулів...
".venv\Scripts\python.exe" -c "import cv2, numpy, PIL, onnxruntime" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Необхідні бібліотеки НЕ знайдено. Встановлюємо...
    echo (Завантаження PyTorch та OpenCV може зайняти 1-3 хвилини)
    echo.
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install opencv-python numpy pillow onnxruntime
    if !ERRORLEVEL! neq 0 (
        echo.
        echo ===============================================================================================
        echo [Помилка] Не вдалося встановити бібліотеки. Перевірте з'єднання з інтернетом.
        echo ===============================================================================================
        pause
        exit /b 1
    )
    echo [OK] Бібліотеки успішно встановлено.
) else (
    echo [OK] Усі необхідні бібліотеки присутні.
)

REM 4. Перевірка наявності файлу навченої моделі
if not exist "model\best.pt" (
    echo.
    echo ===============================================================================================
    echo [ПОМИЛКА] Файл моделі не знайдено за шляхом: model\best.pt
    echo Створіть папку "model" поруч із цим файлом та покладіть туди файл "best.pt".
    echo ===============================================================================================
    echo.
    pause
    exit /b 1
)

endlocal

if %ERRORLEVEL% neq 0 (
    echo.
    echo ========================================================
    echo [УВАГА] Додаток завершився з помилкою (код %ERRORLEVEL%).
    echo Текст помилки наведено вище.
    echo ========================================================
    pause
)
