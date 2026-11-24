@echo off
echo ============================================================================
echo  TradingAgent-001: Windows 10 (16GB RAM) Zero-Fee Setup Wizard
echo ============================================================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.11+ is not installed or not in PATH.
    echo Download free from https://www.python.org/downloads/windows/ and check "Add Python to PATH".
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [1/4] Creating isolated Python virtual environment (.venv)...
    python -m venv .venv
)

echo [2/4] Activating .venv and installing core zero-fee packages + MetaTrader5 IPC...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[windows,dev]"

if not exist ".env" (
    echo [3/4] Creating .env from .env.example...
    copy .env.example .env
)

echo [4/4] Running local system diagnostics...
tradingagent doctor

echo.
echo ============================================================================
echo  OPTIONAL ZERO-FEE LOCAL AI MODELS (Ollama for Windows 10 / 16GB RAM):
echo  1. Install Ollama free from https://ollama.com/download/windows
echo  2. Pull lightweight 16GB-RAM-friendly models:
echo       ollama pull qwen3:4b
echo       ollama pull qwen3:8b
echo  3. Launch dashboard:
echo       .venv\Scripts\activate ^&^& tradingagent dashboard
echo ============================================================================
pause
