@echo off
setlocal EnableDelayedExpansion

set "VENV_DIR=.venv"
set "MODEL_URL=https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
set "MODEL_PATH=models\hand_landmarker.task"

echo ============================================================
echo HoloForge - Windows setup
echo ============================================================

rem ---- 1. Locate Python -------------------------------------------------
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found on PATH.
    echo         Install Python 3.10+ from python.org, then re-run this script.
    exit /b 1
)

echo [1/4] Python: %python%
python --version

rem ---- 2. Create the virtual environment --------------------------------
if exist "%VENV_DIR%" (
    echo [2/4] Virtual environment already exists at %VENV_DIR%; reusing it.
) else (
    echo [2/4] Creating virtual environment at %VENV_DIR%...
    python -m venv "%VENV_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create the virtual environment.
        exit /b 1
    )
)

rem ---- 3. Activate and install dependencies -----------------------------
echo [3/4] Installing dependencies from requirements.txt...
call "%VENV_DIR%\Scripts\activate.bat"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate the virtual environment.
    exit /b 1
)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Dependency installation failed.
    exit /b 1
)

rem ---- 4. Download the MediaPipe hand landmarker model ------------------
echo [4/4] Downloading hand_landmarker.task to %MODEL_PATH%...
if not exist "models" mkdir "models"

powershell -NoProfile -Command ^
    "$ErrorActionPreference='Stop'; ^
    Invoke-WebRequest -Uri '%MODEL_URL%' -OutFile '%MODEL_PATH%'; ^
    Write-Host 'Download complete: %MODEL_PATH%'"

if %ERRORLEVEL% neq 0 (
    echo [WARNING] The model download failed. You can retry it manually with:
    echo          powershell -Command "Invoke-WebRequest -Uri ^"^'%MODEL_URL%^'^" -OutFile ^'^'%MODEL_PATH%^'^""
    echo          Gesture detection will not work until the model is present.
)

echo ============================================================
echo Setup complete. Run:
echo   call %VENV_DIR%\Scripts\activate.bat
echo   python -m src.main
echo ============================================================
endlocal