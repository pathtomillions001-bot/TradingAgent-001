@echo off
setlocal EnableExtensions
title TradingAgent-001 Launcher

set "REPO_DIR=%~dp0.."
pushd "%REPO_DIR%"
set "REPO_DIR=%CD%"

echo ==============================================================
echo   TradingAgent-001 — Windows 10 + MT5 One-Click Restart
echo ==============================================================

REM 1. Read MT5_TERMINAL_PATH from .env if configured, else check JustMarkets / default MT5
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

tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I "terminal64.exe" >NUL
if errorlevel 1 (
    if exist "%MT5_EXE%" (
        echo [1/3] Launching MetaTrader 5: "%MT5_EXE%"
        start "" "%MT5_EXE%"
        timeout /t 7 /nobreak >NUL
    ) else (
        echo [1/3] Note: MT5 executable not found at "%MT5_EXE%" — continuing.
    )
) else (
    echo [1/3] MetaTrader 5 is already running.
)

REM 2. Stop any previous TradingAgent server or stale listener on port 8000
echo [2/3] Stopping any previous TradingAgent session...
for /f "tokens=5" %%P in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%P >NUL 2>&1
)
taskkill /FI "WINDOWTITLE eq TradingAgent*" /F >NUL 2>&1

REM 3. Start unified Web Control Center + 15s Gold-First Autopilot in one clean background window
echo [3/3] Starting Unified Dashboard + 15s Gold-First Autopilot on http://localhost:8000 ...
start "TradingAgent Server" /min cmd /k "cd /d "%REPO_DIR%" && call .venv\Scripts\activate.bat && tradingagent dashboard --autopilot --interval 15"

timeout /t 3 /nobreak >NUL
start "" http://localhost:8000
popd
exit /b 0
