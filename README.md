# TradingAgent-001

**100% Zero-Fee Local Multi-Agent Scalping & Intraday Trading Engine for Windows 10 + MetaTrader 5**

Built for a **Windows 10 (16 GB RAM)** laptop with **zero cloud hosting fees** and **zero LLM API fees**:
- **Gold-First (`XAUUSD`) Priority Hierarchy:** Scans `XAUUSD` first on every cycle during prime London & New York liquidity windows, falling back to `NAS100` (`US100`), `BTCUSD`, `EURUSD`, `GBPUSD`, and `USDCAD` only when Gold has no A/A+ setup or max Gold exposure is active.
- **3-Layer Quant-Gated Agentic Architecture:**
  1. **Layer 1 — Millisecond Quant Scanner (`src/tradingagent/scanner.py`):** Vectorized Smart Money Concepts (Liquidity Sweeps + Fair Value Gaps), Session VWAP/EMA pullbacks, Opening Range Breakouts, and live spread-to-SL filters (`0` LLM calls on empty bars).
  2. **Layer 2 — Multi-Agent Trading Committee (`src/tradingagent/agents.py` & `memory.py`):** Inspired by *TauricResearch/TradingAgents* and *QuantHarness*. Runs `SMC_Microstructure_Analyst`, `Session_Macro_Analyst`, and `Adversarial_Trap_Critic` + **Settled Reflection Memory** (SQLite) using local **Ollama** (`qwen3:4b` / `qwen3:8b`, fitting comfortably inside 16 GB RAM alongside MT5) with an automatic offline deterministic rubric fallback.
  3. **Layer 3 — Deterministic Pre-Trade Risk Firewall (`src/tradingagent/risk.py`):** Inspired by *deterministic-trader* & *FinRobot V2*. Pure tested Python enforcing exact `tick_value` lot sizing, Daily Loss Kill-Switch (`2.5%`), Net USD Directional Correlation Cap (`±1.25%` across `XAUUSD`/`EURUSD`/`GBPUSD`/`USDCAD`/`BTCUSD`), +1R Break-Even management, and **mandatory Broker Server-Side Hard SL & TP** on every MT5 order.
- **Native Windows 10 MT5 IPC (`src/tradingagent/broker/mt5_windows.py`):** Sub-2ms Named Pipe connection via the official free `MetaTrader5` Python package + `order_check()` preflight + automatic broker symbol suffix discovery (`XAUUSDm`, `XAUUSD.pro`, `USTEC`, etc.).
- **Spread-Aware Ablation Backtester (`src/tradingagent/backtest.py`):** Zero-lookahead event simulation with conservative intra-bar SL-first priority and live bid/ask spread deduction, comparing *Raw Quant Baseline* vs. *Multi-Agent Committee + Risk Gate*.

---

## Quick Start on Your Windows 10 Laptop (Zero Fees)

### 1. One-Click Setup
Open **Command Prompt** or **PowerShell** inside the cloned repository and run:

```bat
scripts\setup_windows10.bat
```

*(Or manually)*:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[windows,dev]"
copy .env.example .env
```

### 2. Optional Free Local LLM Setup (Tuned for 16 GB RAM)
1. Install **Ollama for Windows** (free): `https://ollama.com/download/windows`
2. Pull the 16 GB RAM-optimized models (~5 GB memory footprint, leaving ~11 GB for Windows 10 + MT5):
   ```powershell
   ollama pull qwen3:4b
   ollama pull qwen3:8b
   ```
   *(Note: Even if Ollama is not running yet, `TradingAgent-001` runs seamlessly using its built-in Deterministic Multi-Agent Rubric at zero cost!)*

### 3. Connect Your MetaTrader 5 Account (Demo First!)
1. Open your **MetaTrader 5** terminal on Windows 10, log into your **Demo Account**, and click **Algo Trading** (green button in the top toolbar).
2. Edit `.env`:
   - Keep `TRADING_MODE=paper` to test locally with simulated quotes first.
   - Switch to `TRADING_MODE=mt5_demo` to read live broker ticks and execute on your MT5 Demo account.
   - Only switch to `TRADING_MODE=mt5_live` after verifying 100+ profitable demo trades.

### 4. Run Diagnostics, Backtests, Live Loop, or Web Control Center
```powershell
# 1. Check MT5 connection, Symbol Auto-Mapper, Ollama status, and Risk Firewall
tradingagent doctor

# 2. Run Spread-Aware Ablation Backtest on Gold (XAUUSD)
tradingagent backtest --symbol XAUUSD --bars 2800

# 3. Launch Local Web Control Center Dashboard (http://localhost:8000)
tradingagent dashboard

# 4. Run Autonomous Continuous Loop in Terminal
tradingagent run --interval 15
```

---

## Documentation
- Step-by-Step Operations & Windows 10 Launch Guide: [`docs/LAUNCH_AND_OPERATIONS_GUIDE.md`](docs/LAUNCH_AND_OPERATIONS_GUIDE.md)
- Full Research & Open-Source Teardown: [`docs/RESEARCH_AND_ARCHITECTURE.md`](docs/RESEARCH_AND_ARCHITECTURE.md)
- Symbol Priority & Spread Rules: [`config/symbols.yaml`](config/symbols.yaml)
