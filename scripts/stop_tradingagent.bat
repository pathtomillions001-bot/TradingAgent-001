@echo off
title Stop TradingAgent-001
echo Stopping TradingAgent-001 Local Server & Autopilot...
for /f "tokens=5" %%P in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%P >NUL 2>&1
)
taskkill /FI "WINDOWTITLE eq TradingAgent*" /F >NUL 2>&1
echo TradingAgent-001 stopped cleanly. (Server-side MT5 SL/TP orders remain active on broker.)
timeout /t 2 >NUL
exit /b 0
