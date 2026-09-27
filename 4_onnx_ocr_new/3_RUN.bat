@echo off
cd /d "%~dp0"
chcp 65001 > nul

start "" ".venv\Scripts\pythonw.exe" "app.py"
