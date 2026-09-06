@echo off
setlocal enabledelayedexpansion

echo ======================================================
echo             ScholAR Windows Setup Wizard
echo ======================================================
echo.

:: 1. Check Python
echo [1/5] Checking Python...
where python >nul 2>nul
if %errorlevel% neq 0 (
    where py >nul 2>nul
    if %errorlevel% neq 0 (
        echo [ERROR] Python is not installed or not in your PATH.
        echo Please download and install Python 3.11 or 3.12 from:
        echo https://www.python.org/downloads/
        echo (IMPORTANT: Check the box "Add python.exe to PATH" during installation)
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=py
    )
) else (
    set PYTHON_CMD=python
)
%PYTHON_CMD% --version

:: 2. Check Node.js and npm
echo.
echo [2/5] Checking Node.js and npm...
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in your PATH.
    echo Please download and install Node.js 18 or 20 LTS from:
    echo https://nodejs.org/
    pause
    exit /b 1
)
node --version
npm --version

:: 3. Check Ollama
echo.
echo [3/5] Checking Ollama...
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Ollama is not installed or not in your PATH.
    echo To run local models, download and install Ollama for Windows from:
    echo https://ollama.com/download
    echo After installing, you can continue this setup.
    echo.
) else (
    ollama --version
)

:: 4. Setup Python Virtual Environment and Dependencies
echo.
echo [4/5] Setting up Python virtual environment (.venv)...
if not exist ".venv" (
    %PYTHON_CMD% -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo Installing Python backend dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Lockfile installation failed, attempting fallback to base dependencies...
    python -m pip install -r requirements/base.in
)

:: Setup backend .env if missing
if not exist "backend\.env" (
    if exist "backend\.env.example" (
        copy "backend\.env.example" "backend\.env" >nul
        echo Created backend\.env from example.
    )
)

:: 5. Setup Frontend Dependencies
echo.
echo [5/5] Installing Frontend dependencies (npm)...
cd frontend
call npm install
cd ..

:: 6. Auto-detect Hardware and Configure Model
echo.
echo ======================================================
echo           Hardware Detection & Model Selection
echo ======================================================
python scripts\setup_models.py

echo.
echo ======================================================
echo             ScholAR Setup Complete!
echo ======================================================
echo You can now start ScholAR anytime by running:
echo     run_windows.bat
echo.
pause
