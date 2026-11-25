@echo off
setlocal EnableExtensions
title TA001_Bootstrapper

set "REPO_DIR=%~dp0.."
pushd "%REPO_DIR%"
set "REPO_DIR=%CD%"
set "REPO_DRIVE=%~d0"

echo ==============================================================
echo   TradingAgent-001 — Windows 10 + MT5 One-Click Launcher
echo   Workspace: %REPO_DIR%
echo ==============================================================

REM 1. Ensure MetaTrader 5 is running (reads MT5_TERMINAL_PATH from .env, else JustMarkets / standard MT5)
set "MT5_EXE="
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if /I "%%A"=="MT5_TERMINAL_PATH" set "MT5_EXE=%%B"
    )
)
if "%MT5_EXE%"=="" (
    if exist "C:\Program Files\JustMarkets MetaTrader 5\terminal64.exe" (
        set "MT5_EXE=C:\Program Files\JustMarkets MetaTrader 5\terminal64.exe"
    ) else (
        set "MT5_EXE=C:\Program Files\MetaTrader 5\terminal64.exe"
    )
)

tasklist | find /I "terminal64.exe" >NUL
if errorlevel 1 (
    if exist "%MT5_EXE%" (
        echo [1/4] Launching MetaTrader 5: %MT5_EXE%
        start "" "%MT5_EXE%"
        ping 127.0.0.1 -n 7 >NUL
    )
) else (
    echo [1/4] MetaTrader 5 is already running.
)

REM 2. Ensure Local Ollama AI Server is running (checks External Drive OllamaApp first, then LocalAppData)
set "OLLAMA_EXE="
if exist "%REPO_DRIVE%\OllamaApp\ollama.exe" (
    set "OLLAMA_EXE=%REPO_DRIVE%\OllamaApp\ollama.exe"
) else if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
)

tasklist | find /I "ollama.exe" >NUL
if errorlevel 1 (
    if not "%OLLAMA_EXE%"=="" (
        echo [2/4] Starting Ollama Local AI Server: %OLLAMA_EXE%
        start "Ollama_Server" /min "%OLLAMA_EXE%" serve
        ping 127.0.0.1 -n 4 >NUL
    ) else (
        echo [2/4] Ollama not installed — Deterministic Multi-Agent Rubric active.
    )
) else (
    echo [2/4] Ollama AI Server is already running.
)

REM 3. Stop any previous TA001_Live_Engine or stale listener on port 8000 (never matches TA001_Bootstrapper!)
echo [3/4] Cleaning previous port 8000 sessions...
taskkill /FI "WINDOWTITLE eq TA001_Live_Engine*" /F >NUL 2>&1
for /f "tokens=5" %%P in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%P >NUL 2>&1
)

REM 4. Launch Unified Web Control Center + 15s Gold-First Autopilot directly via .venv python.exe
echo [4/4] Starting TradingAgent Control Center + 15s Autopilot on http://localhost:8000 ...
start "TA001_Live_Engine" /D "%REPO_DIR%" "%REPO_DIR%\.venv\Scripts\python.exe" -m tradingagent.cli dashboard --autopilot --interval 15

ping 127.0.0.1 -n 5 >NUL
start "" http://localhost:8000
popd
exit /b 0
