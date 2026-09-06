@echo off
setlocal

echo ======================================================
echo              Starting ScholAR Local System
echo ======================================================
echo.

:: 1. Verify virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment (.venv) not found.
    echo Please run setup_windows.bat first to configure the environment.
    pause
    exit /b 1
)

:: 2. Check if Ollama is running, start it in background if not
echo [1/3] Checking Ollama service...
curl -s http://127.0.0.1:11434 >nul 2>nul
if %errorlevel% neq 0 (
    echo Starting Ollama in background...
    start /min "Ollama Service" ollama serve
    timeout /t 3 /nobreak >nul
) else (
    echo Ollama is already running.
)

:: 3. Start Backend in a separate window
echo [2/3] Starting FastAPI Backend on port 8000...
start "ScholAR Backend (FastAPI)" cmd /k "call .venv\Scripts\activate.bat && uvicorn backend.main:app --port 8000 --reload"

:: 4. Wait for backend to come up
timeout /t 3 /nobreak >nul

:: 5. Start Frontend in a separate window
echo [3/3] Starting Next.js Frontend on port 3000...
start "ScholAR Frontend (Next.js)" cmd /k "cd frontend && npm run dev"

:: 6. Wait and open browser
timeout /t 4 /nobreak >nul
echo.
echo ======================================================
echo ScholAR is running!
echo - Web UI:  http://localhost:3000
echo - API:     http://localhost:8000
echo - Docs:    http://localhost:8000/docs
echo ======================================================
echo.
start http://localhost:3000

echo To stop ScholAR, simply close the Backend and Frontend command windows.
pause
