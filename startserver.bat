@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0python_e\python.exe"
if not exist "%PYTHON%" (
    echo [WARN] Portable Python not found at "%PYTHON%" - falling back to "python" on PATH.
    set "PYTHON=python"
)

echo Checking Python environment...
"%PYTHON%" setup_torch.py
if errorlevel 1 (
    echo.
    echo Setup failed - see output above.
    pause
    exit /b 1
)

echo.
echo Starting CaseSorter AI Server...
"%PYTHON%" server.py

pause
