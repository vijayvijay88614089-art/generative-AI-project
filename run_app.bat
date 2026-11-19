@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo InsightForge AI virtual environment was not found.
    echo Install project requirements first, then run this file again.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run app.py --server.headless false
if errorlevel 1 pause
