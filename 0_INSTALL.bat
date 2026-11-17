@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 > nul
title ANPR - NVR — Автоматичне встановлення

echo ========================================================
echo   Завантаження проєкту з GitHub...
echo ========================================================
set "REPO_ZIP=%~dp0anpr_repo_main.zip"
set "REPO_EXT=%~dp0_anpr_repo_extract"
set "REPO_URL=https://github.com/OlegSkalGit/Get-plate-number-from-camera/archive/refs/heads/main.zip"

if exist "%REPO_ZIP%" del /f /q "%REPO_ZIP%"
if exist "%REPO_EXT%" rmdir /s /q "%REPO_EXT%"

where curl.exe >nul 2>&1
if %errorlevel%==0 (
    curl.exe -fL "%REPO_URL%" -o "%REPO_ZIP%"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%REPO_URL%' -OutFile '%REPO_ZIP%'"
)
if errorlevel 1 (
    echo [ПОМИЛКА] Не вдалося завантажити репозиторій з GitHub.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo   Розпакування файлів...
echo ========================================================
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Expand-Archive -Path '%REPO_ZIP%' -DestinationPath '%REPO_EXT%' -Force"
if errorlevel 1 (
    echo [ПОМИЛКА] Не вдалося розпакувати архів.
    if exist "%REPO_ZIP%" del /f /q "%REPO_ZIP%"
    pause
    exit /b 1
)

robocopy "%REPO_EXT%\Get-plate-number-from-camera-main" "%~dp0." /E /MOVE /IS /IT > nul
if errorlevel 8 (
    echo [ПОМИЛКА] Помилка копіювання файлів проєкту.
    pause
    exit /b 1
)

if exist "%REPO_ZIP%" del /f /q "%REPO_ZIP%"
if exist "%REPO_EXT%" rmdir /s /q "%REPO_EXT%"

if not exist "%~dp01_SETUP.bat" (
    echo [ПОМИЛКА] Файл 1_SETUP.bat не знайдено після розпакування.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo   Запуск 1_SETUP.bat...
echo ========================================================
call "%~dp01_SETUP.bat"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ПОМИЛКА] 1_SETUP.bat не створив віртуальне оточення .venv.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo   Завантаження моделей для модулів...
echo ========================================================
if not exist "%~dp0models" mkdir "%~dp0models"
for %%M in (best.onnx cct_xs_v2_global.onnx cct_xs_v2_global_plate_config.yaml scrfd_face_detector.onnx face_recognizer.onnx yolo11n.onnx) do (
    set "NEED_DL=0"
    if not exist "%~dp0models\%%M" (
        set "NEED_DL=1"
    ) else (
        for %%A in ("%~dp0models\%%M") do if %%~zA LSS 512 set "NEED_DL=1"
    )
    if "!NEED_DL!"=="1" (
        echo Завантаження %%M...
        where curl.exe >nul 2>&1
        if !errorlevel!==0 (
            curl.exe -fL "https://media.githubusercontent.com/media/OlegSkalGit/Get-plate-number-from-camera/main/models/%%M" -o "%~dp0models\%%M"
        ) else (
            powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://media.githubusercontent.com/media/OlegSkalGit/Get-plate-number-from-camera/main/models/%%M' -OutFile '%~dp0models\%%M'"
        )
    )
)
"%~dp0.venv\Scripts\python.exe" -c "from scripts.core.module_registry import ModuleRegistry; r = ModuleRegistry('.'); [print(e['id'], '->', e['spec']['ensure_resources']('.', download=True).get('message', 'OK')) for e in r.list_modules() if callable(e['spec'].get('ensure_resources'))]"

if not exist "%~dp04_BUILD_exe.bat" (
    echo [ПОМИЛКА] Файл 4_BUILD_exe.bat не знайдено.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo   Збірка EXE (4_BUILD_exe.bat)...
echo ========================================================
call "%~dp04_BUILD_exe.bat"
if not exist "%~dp0dist\ANPR_Monitor\ANPR_Monitor.exe" (
    echo [ПОМИЛКА] Збірка ANPR_Monitor не знайдена у dist\ANPR_Monitor.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo   Перенесення ANPR_Monitor та очищення тимчасових файлів...
echo ========================================================
if exist "%~dp0ANPR_Monitor" rmdir /s /q "%~dp0ANPR_Monitor"
move "%~dp0dist\ANPR_Monitor" "%~dp0ANPR_Monitor" > nul
if errorlevel 1 (
    robocopy "%~dp0dist\ANPR_Monitor" "%~dp0ANPR_Monitor" /E /MOVE > nul
)

for /d %%D in ("%~dp0*") do (
    if /i not "%%~nxD"=="ANPR_Monitor" rmdir /s /q "%%D"
)
for %%F in ("%~dp0*") do (
    if /i not "%%~nxF"=="%~nx0" del /f /q "%%F"
)

echo.
echo ========================================================
echo   Готово! Програма знаходиться у папці ANPR_Monitor
echo ========================================================
pause
