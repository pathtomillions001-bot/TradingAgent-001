@echo off
title TA001_Stopper
echo Stopping TradingAgent-001 Local Server ^& Autopilot...
taskkill /FI "WINDOWTITLE eq TA001_Live_Engine*" /F >NUL 2>&1
for /f "tokens=5" %%P in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%P >NUL 2>&1
)
echo TradingAgent-001 stopped cleanly. (Server-side MT5 SL/TP orders remain active on broker.)
ping 127.0.0.1 -n 3 >NUL
exit /b 0
