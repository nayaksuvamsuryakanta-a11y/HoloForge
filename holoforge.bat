@echo off
setlocal
cd /d "%~dp0"
echo HoloForge v1.1.0 - Webcam Spatial Computing
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run scripts\setup_windows.bat first.
    exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m src.main %*
