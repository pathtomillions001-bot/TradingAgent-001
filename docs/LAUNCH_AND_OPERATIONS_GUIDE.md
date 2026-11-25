# TradingAgent-001: Step-by-Step Operations & Windows 10 Launch Guide

This guide walks you through **two things**:
1. **Exactly how `TradingAgent-001` thinks, filters, sizes, executes, and manages trades under the hood** (every 15 seconds on your laptop).
2. **A foolproof, step-by-step Windows 10 launch & verification checklist** so you can connect your MT5 account safely, confirm every piece works, and avoid risking real money prematurely.

---

## Part 1: How Our Agent Works Under the Hood (Step-by-Step Lifecycle)

Every cycle (default every **15 seconds**, configurable via `--interval`), `TradingAgentEngine` (`src/tradingagent/engine.py`) runs a **6-Stage Pipeline**:

```text
[Stage 1: Manage Active Trades (<5ms)]
         │
         ▼
[Stage 2: Gold-First Priority Router (XAUUSD -> NAS100/BTC -> FX)]
         │
         ▼
[Stage 3: Deterministic Quant Setup Scanner (0 LLM Calls)]
         │  (95% of checks stop here if market is choppy or spread is too wide)
         ▼  (Only when a clean A/B setup appears ~5–12x/day)
[Stage 4: Multi-Agent Committee + Settled Memory (Ollama Local LLM)]
         │  (SMC Specialist + Macro/Session Specialist + Trap Critic -> Head Trader)
         ▼
[Stage 5: Zero-Trust Deterministic Risk Firewall (9 Hard Python Gates)]
         │
         ▼
[Stage 6: MT5 Preflight `order_check()` -> Server-Side Hard SL/TP Order -> SQLite Audit]
```

### Stage 1: Sub-Second Management of Open Trades (`src/tradingagent/risk.py`)
Before looking for new trades, the engine checks any open positions tagged with its `MT5_MAGIC_NUMBER` (`2026001` — meaning **it never touches manual trades** you open yourself on MT5!):
- When an open scalp/intraday position reaches **`+1.0R` unrealized profit** (e.g., if your Stop Loss was `$2.50` away on Gold and price moves `+$2.50` in your favor), the agent automatically modifies the order on the MT5 server to move your **Stop Loss to Break-Even + Spread Buffer**, making the trade risk-free.

### Stage 2: Gold-First (`XAUUSD`) Priority Router (`src/tradingagent/scanner.py` & `config/symbols.yaml`)
Your agent does **not** treat all symbols equally. It scans in strict priority order:
1. **Tier 1 — `XAUUSD` (Gold):** Always checked **first**. If Gold is not already at `MAX_GOLD_POSITIONS=1` and Gold has a strong setup (`QuantScore >= 75`), the scanner **locks onto Gold immediately** and skips secondary pairs so your margin and daily risk budget go to your best Gold opportunity.
2. **Tier 2 — `NAS100` (`US100`/`USTEC`) & `BTCUSD`:** Checked **only** when Gold has no clean setup or already has an active trade running. (`NAS100` shines during the NY Cash Open; `BTCUSD` provides 24/7/weekend setups if broker spread passes the spread gate).
3. **Tier 3 — `EURUSD`, `GBPUSD`, `USDCAD`:** Clean intraday forex fallbacks when both Tier 1 and Tier 2 are quiet.

### Stage 3: Millisecond Quantitative Setup Detection (`src/tradingagent/indicators.py` & `scanner.py`)
For each symbol, the agent pulls **260 M5 bars** (execution timeframe) + **120 H1 bars** (higher-timeframe trend bias) + **live Bid/Ask tick spread** from MT5:
- **Instant Spread Reject:** If `XAUUSD` spread is currently `> 35 points` (`$0.35`) or consumes `> 12%` of the Stop Loss distance, it rejects immediately (protects you from news spikes and rollover spread widening).
- **3 Institutional Intraday Playbooks Checked:**
  1. **`LIQUIDITY_SWEEP_FVG` (Smart Money / ICT Reversal-Continuation):** Price sweeps a 20-bar liquidity high/low (stop hunt), aggressively displaces back inside structure creating a 3-candle **Fair Value Gap (FVG)**, and reclaims `EMA 20` without RSI exhaustion.
  2. **`VWAP_EMA_PULLBACK` (Trend Continuation):** `H1` trend (`EMA 20 > EMA 50`) aligns with `M5` price holding above **Session Anchored VWAP**, pulling back into `M5 EMA 20` and printing a rejection continuation candle.
  3. **`SESSION_ORB_BREAKOUT` (London / NY Killzone Momentum):** During prime UTC liquidity hours (`07:00–11:00 UTC` London or `12:00–17:00 UTC` NY Overlap) with `ADX >= 24`, price breaks cleanly out of the prior 1-hour consolidation box in alignment with `H1`.
- **Dynamic ATR Stops & Targets:** Stop Loss is placed at **`1.45 × M5 ATR`** and Take Profit at **`3.20 × M5 ATR`** (`~1:2.21` gross R:R, ensuring `> 1:1.80` net R:R after spread).

### Stage 4: Local Zero-Fee Multi-Agent Committee + Settled Memory (`src/tradingagent/agents.py` & `memory.py`)
Only when Stage 3 finds a valid setup does the Multi-Agent Brain wake up (saving 99% of CPU/RAM on your 16 GB laptop!):
1. **`SMC_Microstructure_Analyst`:** Evaluates sweep quality, FVG displacement, and distance from Session VWAP (flags overextension if price is `> 2.6 ATR` away from VWAP).
2. **`Session_Macro_Analyst`:** Evaluates whether we are in London/NY institutional killzones vs. dead Asian chop, and whether `M5` aligns with or fights the `H1` trend.
3. **`Adversarial_Trap_Critic`:** Actively tries to **kill bad trades**—issuing a hard `VETO` if `RSI > 73` on a buy (buying the top of an inducement spike), `RSI < 27` on a sell, or low-ADX fakeouts.
4. **Settled Reflection Memory (`SQLite`):** Pulls past lessons recorded on that exact symbol (`XAUUSD`, etc.).
5. **Head Trader Verdict (`Ollama qwen3:8b` or Offline Deterministic Rubric):** Synthesizes all 3 specialists + memory into a strict typed decision: `EXECUTE_LONG`, `EXECUTE_SHORT`, or `NO_TRADE`, graded `A+`, `A`, `B`, or `SKIP`. *(Only `A+` and `A` setups with zero VETOes proceed!)*

### Stage 5: Zero-Trust Deterministic Risk Firewall (`src/tradingagent/risk.py`)
Even when the AI Committee votes `EXECUTE`, **the AI is never trusted with your account math**. Pure Python enforces 9 sequential gates:
1. **Emergency Kill Switch** is OFF.
2. **Daily Drawdown Lock:** Today's equity drawdown `< 2.50%` (`MAX_DAILY_LOSS_PCT`).
3. **Total Drawdown Lock:** Peak-to-valley drawdown `< 6.00%` (`MAX_TOTAL_DRAWDOWN_PCT`).
4. **Max Open Positions:** Total open `< 2`, Gold open `< 1`.
5. **Anti-Overtrading Cooldown:** At least `15 minutes` since the last trade on that symbol.
6. **Live Spread & Broker `stops_level` Verification:** Re-verifies live tick spread right before order send.
7. **Net R:R After Spread:** Must be `>= 1.80`.
8. **Net USD Directional Correlation Guard (`±1.25%`):** Prevents hidden correlated blowups (e.g., going Long `XAUUSD` + Long `EURUSD` + Long `GBPUSD` all at once = 3x leveraged Short-USD!).
9. **Exact `tick_value` Lot Sizing + 150% Free Margin Buffer:** Calculates exact lots floored to your broker's `volume_step` (`0.01`) so risk never exceeds `0.75%` of equity.

### Stage 6: MT5 `order_check()` Preflight & Server-Side Hard SL/TP Execution (`src/tradingagent/broker/mt5_windows.py`)
1. Runs MetaTrader 5's native `mt5.order_check(request)` dry-run first (verifying broker margin, market open state, filling mode `FOK`/`IOC`, and price precision).
2. Sends `mt5.order_send(request)` **with mandatory server-side `sl` and `tp` attached to the ticket**—so even if your laptop battery dies or Wi-Fi disconnects 10 seconds later, your broker server already holds your hard Stop Loss and Take Profit!
3. Logs the entire setup, specialist debate, risk calculations, and MT5 ticket number permanently into `data/state/tradingagent.sqlite3`.

---

## Part 2: Complete Step-by-Step Guide to Launch & Verify on Your Windows 10 Laptop

Follow these **6 phases in exact order**. Do not skip from Paper to Live—each step proves your setup works cleanly.

### Step 1: Install Prerequisites on Windows 10 (100% Free)
1. **MetaTrader 5 Terminal:** Install from your broker (e.g., Exness, IC Markets, Pepperstone, FTMO, etc.) and log into a **Demo Account** first.
2. **Python 3.11 or 3.12 (64-bit):** Download from [python.org/downloads/windows](https://www.python.org/downloads/windows/).
   - **IMPORTANT:** During installation, check the box **"Add python.exe to PATH"**.
3. **Git for Windows:** Download from [git-scm.com](https://git-scm.com/download/win) (if you haven't already).
4. **Ollama for Windows (Optional but Recommended for Local LLM Reasoning):**
   - Download free from [ollama.com/download/windows](https://ollama.com/download/windows).
   - Open Command Prompt (`cmd`) and pull the 16 GB RAM-tuned models (~5 GB RAM usage):
     ```bat
     ollama pull qwen3:4b
     ollama pull qwen3:8b
     ```

---

### Step 2: Clone & Run the One-Click Windows 10 Setup Script
Open **Command Prompt** (`cmd.exe`) or **PowerShell** on your laptop:

```bat
git clone https://github.com/pathtomillions001-bot/TradingAgent-001.git
cd TradingAgent-001
git checkout arena/23655f5d-tradingagent-001
scripts\setup_windows10.bat
```

**What this does automatically:**
- Creates an isolated `.venv` virtual environment.
- Installs all required libraries **plus** the official Windows `MetaTrader5` Python IPC package.
- Copies `.env.example` to `.env` (starting safely in `TRADING_MODE=paper`).
- Runs `tradingagent doctor` to confirm everything installed cleanly.

---

### Step 3: Verify Offline Logic & Backtester First (`TRADING_MODE=paper`)
Before connecting MT5, verify the automated test suite and backtester on your laptop:

```bat
.venv\Scripts\activate

:: 1. Run the 5 automated safety & execution tests (should say "5 passed")
pytest -v

:: 2. Run a spread-aware Gold backtest
tradingagent backtest --symbol XAUUSD --bars 2800

:: 3. Launch the Local Control Center Dashboard
tradingagent dashboard
```
Open **`http://localhost:8000`** in Chrome/Edge on your laptop:
- Click **`⚡ Scan & Execute Cycle`** — watch it scan `XAUUSD`, log the 3 specialist opinions, calculate exact lots (`0.18`–`0.30` lots on `$10k`), and open a simulated ticket.
- Click **`🛑 Emergency Kill Switch`** — verify it immediately closes the open ticket and changes the top badge to **`HALTED (KILL SWITCH)`**. Click **`✅ Resume Agent Trading`** to reset.

---

### Step 4: Prepare Your MetaTrader 5 Terminal for Python Connection
Inside your open **MetaTrader 5** desktop window on Windows 10:
1. Log into your **MT5 Demo Account**.
2. Look at the top toolbar and click **`Algo Trading`** so the icon turns **Green (Play triangle)** (shortcut `Ctrl + E`).
3. Go to **`Tools` → `Options` → `Expert Advisors` tab**:
   - Check **`Allow algorithmic trading`**.
   - *(Leave "Disable algorithmic trading when account/profile changes" checked or unchecked per your preference, and click OK).*
4. In the **`Market Watch`** panel (`Ctrl + M`) on the left side of MT5:
   - Right-click → **`Show All`** (or make sure your broker's Gold, Nasdaq, BTC, EURUSD, GBPUSD, and USDCAD symbols are visible).

---

### Step 5: Link Your MT5 Demo Account & Verify Live Connection
Open `.env` in Notepad or VS Code and update only `TRADING_MODE` (and `MT5_TERMINAL_PATH` if your broker installed MT5 in a custom folder):

```ini
TRADING_MODE=mt5_demo
MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```
*(Tip: If MT5 is already open and logged into your Demo account on your desktop, you can leave `MT5_LOGIN`, `MT5_PASSWORD`, and `MT5_SERVER` blank—Python attaches directly to the currently logged-in terminal!)*

Now run the **4-Point Live Verification Check** in Command Prompt:

```bat
.venv\Scripts\activate

:: Check 1: Verify Python connects to MT5, reads your real Demo Equity, and maps your broker's exact symbol names!
tradingagent doctor
```
**What you MUST see in `tradingagent doctor`:**
- `Execution Mode: PASS | MT5_DEMO (Broker Server: YourBroker-Demo | Equity: $...)`
- `Symbol Auto-Mapper: PASS | XAUUSD->XAUUSDm` *(or whatever suffix your broker uses!)*
- `Zero-Fee Brain (Ollama): ONLINE` *(if Ollama is running)* or `FALLBACK READY`

```bat
:: Check 2: Pull 20,000 real M5 candles from YOUR exact MT5 broker to Parquet files
python scripts\export_mt5_history.py

:: Check 3: Run ONE read-only scan against your live MT5 Demo feed (places NO orders)
tradingagent scan
```
Look at the JSON printed: confirm `spread_points` matches your MT5 Market Watch window!

---

### Step 6: Launch Autonomous Demo Trading & Monitor Daily
Once Checks 1–3 pass, you have **two ways** to run the agent live on your laptop:

#### Option A: One-Process Dashboard + Built-In 15s Autopilot + Global Hotkey (`Ctrl + Alt + T`)
Install the Desktop Shortcuts & Global Keyboard Hotkeys once:
```bat
.venv\Scripts\activate
tradingagent shortcuts --startup
```
- Press **`Ctrl + Alt + T`** anytime on your Windows 10 laptop to automatically start **MetaTrader 5** + **Ollama** + **Control Center (`http://localhost:8000`) with 15s Gold-First Autopilot ON**!
- Press **`Ctrl + Alt + S`** anytime to stop the local dashboard/autopilot cleanly.
- Or run directly in one terminal window:
  ```bat
  .venv\Scripts\activate
  tradingagent dashboard --autopilot --interval 15
  ```
- *(External Drive / Low `C:` Disk Space Mode)*: The entire repository (`.venv`, SQLite, DuckDB) and Ollama (`OllamaApp` + `OllamaModels` via `OLLAMA_MODELS` / NTFS directory junctions `mklink /J`) can live on an external USB drive (e.g., `G:\Trading\TradingAgent-001`, `G:\OllamaApp`, `G:\OllamaModels`), freeing 100% of model and dataset storage from `C:` while keeping `terminal64.exe` on `C:`.

#### Option B: Separate Terminal Windows (Manual Split Mode)
- **Terminal Window 1 (Dashboard UI at `http://localhost:8000`):**
  ```bat
  .venv\Scripts\activate
  tradingagent dashboard
  ```
- **Terminal Window 2 (Continuous 15-Second Autonomous Trader):**
  ```bat
  .venv\Scripts\activate
  tradingagent run --interval 15
  ```

#### Crucial Windows 10 Laptop Tips So Your Agent Runs Smoothly
1. **Prevent Windows 10 From Sleeping During London/NY Sessions:**
   - Go to **Windows Settings → System → Power & sleep** → Set **"When plugged in, PC goes to sleep after"** to **`Never`** while trading! (You can turn off your screen, just don't put the CPU to sleep).
   - Keep your laptop plugged into AC power during active trading hours (`07:00–17:00 UTC` / `10:00 AM – 8:00 PM EAT`).
2. **Remember What Normal Healthy Behavior Looks Like:**
   - **90%+ of 15-second cycles will print `Candidates=0` or `NO_TRADE`!** Do *not* think it is broken if it doesn't trade in the first 10 minutes. A disciplined Gold scalper waits for clean Liquidity Sweeps, VWAP pullbacks, or Killzone breakouts (~2 to 6 high-grade trades per day).
3. **What Happens If Your Internet Drops While a Trade Is Open?**
   - Because every order sends **hard `sl` and `tp` directly to your MT5 broker server**, your Stop Loss and Take Profit remain 100% active on your broker's server even if your laptop loses Wi-Fi or shuts down. When `tradingagent run` restarts, it automatically detects open positions matching `MT5_MAGIC_NUMBER=2026001` and resumes managing Break-Even stops!
4. **Graduation Rule Before Switching to `TRADING_MODE=mt5_live`:**
   - Run on `mt5_demo` for **at least 2–3 weeks (50–100+ settled trades)** across London & NY sessions.
   - Verify your demo **Profit Factor > 1.30**, **Max Daily Drawdown never breached `2.50%`**, and live broker slippage on `XAUUSD` matches expectations.
   - Only then change `.env` to `TRADING_MODE=mt5_live` (and start with conservative `RISK_PER_TRADE_PCT=0.50`).
