# TradingAgent-001: Extensive Research & Architectural Blueprint

> **Goal:** Build and run a 100% local, laptop-first autonomous multi-agent trading system linked directly to **MetaTrader 5 (MT5)** — with zero dependency on external cloud hosts — focused on **short-term scalping and intraday trading** with a **Gold-First (`XAUUSD`)** priority hierarchy (`XAUUSD` primary; `EURUSD`, `GBPUSD`, `USDCAD`, `BTCUSD`, `NAS100/US100` secondary fallback setups).

---

## 1. Repository Audit (`TradingAgent-001`)

- **Current State:** Clean greenfield repository (`README.md` with `# TradingAgent-001`, commit `de26ca5`).
- **Opportunity:** Because we are starting clean before writing production code, we can avoid the two biggest traps in open-source AI trading repos:
  1. **Research toys that cannot execute or survive spread/slippage** (pure LLM debate scripts built for daily US stocks).
  2. **Unsafe prompt-driven bots** where an LLM hallucinates stop-loss prices, lot sizes, or overtrades 30 times a day into news spikes.

---

## 2. Deep-Dive Evaluation of Reference Open-Source Repositories

### A. Multi-Agent & AI Trading Frameworks (The "Brain")

| Repository | Stars / Status | Core Architecture | Strengths to Adopt ("Steal") | Critical Gaps for MT5 Scalping / Intraday |
| :--- | :--- | :--- | :--- | :--- |
| **[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)** | ~124k ⭐ (Apache-2.0, Active v1.0) | LangGraph multi-agent "trading firm": Parallel Analysts (Market, News, Sentiment, Macro) → Bull vs. Bear Debate → Research Manager → Trader → 3-Voice Risk Debate → Portfolio Manager + Settled Memory Log. Supports **Ollama / vLLM** (local) & cloud APIs. | 1. **Role specialization & Bull/Bear adversarial debate** (reduces single-prompt directional bias).<br>2. **Settled Memory Log**: scores past decisions after holding period and feeds lessons back.<br>3. **Two-tier LLM routing** (fast/small model for scanning; deep reasoning model for final call).<br>4. **`REVIEW` / `NO_TRADE` fallback** when output is uncertain. | **Does NOT place trades or simulate portfolios.** Built for multi-day US equities (SEC EDGAR, daily closes). Calling 11+ sequential LLM prompts takes 20–90 seconds — **too slow if put directly inside a tick-by-tick scalping loop** without a fast quantitative trigger gate in front of it. Also, its "risk team" is just text prompts, not hard math. |
| **[Y-Research-SBU/QuantHarness](https://github.com/Y-Research-SBU/QuantHarness)** *(QuantAgent)* | Active (MIT, 2025–2026) | LangGraph price-driven multi-agent system specifically for **short-horizon / high-frequency** technical trading: Indicator Agent + Pattern Agent + Trend Agent → Decision Agent. | **Short-horizon focus!** Unlike TradingAgents (which focuses on company filings), QuantHarness decomposes short-term price action into momentum extremes (RSI/Stoch/MACD), candlestick/structural patterns, and multi-timeframe trend alignment. | Lacks native MT5 execution, spread/commission-aware intraday backtesting, and deterministic lot/margin enforcement. |
| **[AI4Finance-Foundation](https://github.com/ai4finance-foundation)** (`FinRobot`, `FinRL`, `FinGPT`) | 60k+ ⭐ combined (MIT / Apache-2.0) | **FinRobot (V2 Desktop):** Typed agent pipelines with deterministic compute & audit operators.<br>**FinRL / ElegantRL:** Deep Reinforcement Learning (PPO, SAC, DQN) on Gymnasium market envs.<br>**FinGPT:** Financial sentiment LoRA models. | 1. **FinRobot's core rule:** *LLMs never do arithmetic* — Python computes exact indicators/metrics, and audit operators verify numbers.<br>2. **FinRL:** Great if we later train fast RL policy filters (e.g., optimal trailing stop or regime classification) that run in `<2ms` locally. | Pure DRL (`FinRL`) overfits badly on noisy M1/M5 forex/gold data unless heavily regularized and paired with strict regime filters. Heavy training overhead for v1. |
| **[HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading)** & **[HKUDS/AI-Trader](https://github.com/HKUDS/AI-Trader)** | ~38k ⭐ & ~23k ⭐ (MIT, Active 2026) | Personal autonomous trading agents with CLI/Web UI, tool contracts, paper-trading harness, and **committed risk mandates + kill switches**. | **Bounded Live Mandate & Kill Switch architecture:** The agent operates strictly inside a pre-committed risk envelope (max daily drawdown, symbol whitelist, max orders). | General-purpose equity/crypto focus; lacks specialized XAUUSD session/spread microstructure and native MT5 IPC. |
| **[hozaifa4you/x-particle](https://github.com/hozaifa4you/x-particle)** & **[FXDavid-OffbeatForex/TLC](https://github.com/FXDavid-OffbeatForex/TLC)** | Niche MT5 Agent Repos (2026) | LangGraph / Subagent councils wired directly to **MetaTrader 5** for Forex & `XAUUSD` with ATR stops, cooldowns, and `NO_TRADE` defaults. | Real-world lessons learned on MT5: **80%+ of runs should output `NO_TRADE`**, mandatory trade cooldowns (prevents LLM overtrading), spread-aware ATR stops, and multi-timeframe (M5/M15/H1) confirmation. | Small single-maintainer codebases without institutional backtesting engines (no VectorBT / NautilusTrader walk-forward validation). |
| **[jkspeaks/deterministic-trader](https://github.com/jkspeaks/deterministic-trader)** | Active (MIT, Oct 2026) | *"An AI agent that trades a real account and is never allowed to make an unchecked decision."* Every risk/sizing rule is in tested Python code, validated via JSON Schema. | **Architectural gold standard for safety:** Hard separation between AI reasoning/proposals and deterministic Python pre-trade risk/execution gates. | Built for equities (Robinhood), needs adaptation for MT5 margin/pip/tick-value mechanics. |

---

### B. Backtesting & Quantitative Engines (The "Proving Ground")

| Engine | Stars / Status | Core Architecture | Verdict for Our Laptop Stack |
| :--- | :--- | :--- | :--- |
| **[polakowo/vectorbt](https://github.com/polakowo/vectorbt)** | ~10k+ ⭐ (v1.2.x with **Rust + Numba** backends, Nov 2026) | Array/matrix-vectorized research engine. Tests 10,000+ parameter/session/ATR combinations across years of M1/M5 bars in seconds on a laptop. | **ADOPT (Stage 1 Fast Discovery & Optimization).** Unbeatable for rapidly testing XAUUSD scalping setups, session killzones, spread sensitivity, and walk-forward parameter stability on a laptop CPU. *Caveat:* Vectorized engines simplify intra-bar tick ordering, so winners must pass Stage 2. |
| **[nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader)** | ~31k ⭐ (LGPL-3.0, Rust core + Python API, Active Nov 2026) | Institutional event-driven engine. **Identical Python strategy code for backtest and live.** Supports tick-level bid/ask simulation, latency models, fill probability, slippage, and Margin/Netting/Hedging accounts. Community `mt5-connector` exists! | **ADOPT (Stage 2 High-Fidelity Tick/Bar Validation & Core Event Engine).** For scalping `XAUUSD` on M1/M5, a 25-cent spread widening or 150ms slippage turns a fake winner into a real loser. NautilusTrader models real bid/ask ticks, order modifications, partial fills, and margin accurately on a local laptop using Parquet catalogs. |
| **[mementum/backtrader](https://github.com/mementum/backtrader)** | Classic (GPL-3.0, **Dormant — last commit April 2023**) | Pure-Python single-threaded event-driven backtester. | **DO NOT ADOPT.** Unmaintained since 2023, very slow on millions of M1/tick bars for intraday Gold/NAS100, and superseded in every way by `vectorbt` (for speed) + `nautilus_trader` (for realistic event-driven execution). |

---

### C. MetaTrader 5 Local Connectivity & Tooling Ecosystem

| Tool / Approach | OS Support | Latency | Assessment |
| :--- | :--- | :--- | :--- |
| **Official `MetaTrader5` Python Package** (`pip install MetaTrader5`) | **Windows Native** *(or inside Wine prefix)* | **< 1–2 ms** (Local Named Pipes IPC) | **Primary Choice if your laptop runs Windows.** Zero network hop, native access to ticks (`copy_ticks_from`), bars (`copy_rates_from_pos`), symbol specs (`trade_tick_value`, `stops_level`), `order_check()`, and `order_send()`. |
| **Custom MQL5 ZeroMQ Expert Advisor (`pyzmq` ↔ `MQL5 EA`)** *(Darwinex `dwx-zeromq` / `dwxconnect` pattern)* | **Windows, macOS, Linux** (Cross-platform TCP `127.0.0.1`) | **~1–5 ms** | **Primary Choice if your laptop runs macOS or Linux** (where MT5 runs inside Wine/CrossOver/Bottles), AND great on Windows if we want real-time push tick events (`PUB/SUB`) + synchronous order execution (`REQ/REP`) + local chart visual overlays inside MT5! |
| **MT5 MCP Servers** (`vincentwongso/mt5-trading-mcp`, `CodeWithBehnam/meta-trader-5-mcp`) | Windows / macOS Wine / Docker | ~10–50 ms | Great **design reference** for clean Pydantic tool schemas (`check_order`, `calc_margin`, audit logs, idempotency keys), and optional interactive debugging from an AI assistant. For the autonomous trading loop itself, calling our Python `MT5Adapter` directly in-process is faster and more reliable. |
| **Smart Money & TA Libraries** (`pandas-ta`, `ta-lib`, `joshyattridge/smart-money-concepts`) | Cross-platform | < 5 ms | Deterministic Python calculation of **VWAP, EMA ribbons, ATR, RSI, Bollinger/Keltner squeezes, Liquidity Sweeps, Order Blocks, Fair Value Gaps (FVG), and Session High/Low sweeps** — critical for Gold & Nasdaq scalping. |

---

## 3. Crucial Reality Check: Why Naive LLM Agents Fail at Scalping (and How We Solve It)

Most people who connect `TradingAgents` directly to MT5 on M1/M5 charts fail for **four specific reasons**:

1. **Latency vs. Candle Speed:** If an M1/M5 Gold candle sweeps London liquidity, waiting 45 seconds for 8 LLM agents to debate means price has already moved 35 pips ($3.50/oz) away from your optimal entry, ruining your Risk:Reward ratio.
2. **LLM Cost & Laptop Heat:** Running a multi-agent debate on *every tick or every M1 bar* across 6 symbols equals 8,640 multi-agent runs per day — burning hundreds of dollars in API tokens or pegging your laptop GPU at 100% 24/7.
3. **LLM Arithmetic & Hallucination:** LLMs frequently miscalculate pip values across heterogeneous assets (`XAUUSD` 1 lot = 100 oz vs. `NAS100` contract size vs. `BTCUSD` vs. `USDCAD` quote currency conversion) or place stops inside the broker's `SYMBOL_TRADE_STOPS_LEVEL`.
4. **Backtesting an LLM is Impossible Over 100,000 Candles if Every Candle Calls an LLM:** You cannot backtest 2 years of M5 XAUUSD (150,000 bars) if each bar makes 10 LLM calls.

### The Solution: **Two-Speed Hybrid "Quant-Gated Agentic" Architecture**

Instead of having LLMs watch every tick, we combine **millisecond deterministic quant code** with **selective multi-agent LLM reasoning**:

- **Speed Layer 1 — Continuous Deterministic Scanner & Trade Manager (Runs every 100ms–1s in Python/Rust, 0 LLM calls):**
  - Streams live MT5 ticks/bars for `XAUUSD` first, and secondary watchlists (`EURUSD`, `GBPUSD`, `USDCAD`, `BTCUSD`, `NAS100`).
  - Enforces **Session & Spread Filters** (e.g., blocks XAUUSD scalps during the 17:00 EST rollover spread spike or dead Asian chop unless volatility breakout occurs).
  - Computes exact multi-timeframe structure (H1/M15 bias + M5/M1 trigger: Liquidity Sweep + Displacement/FVG, VWAP Band Mean-Reversion, or London/NY Opening Range Breakout).
  - Manages **open positions in real time** (hard SL/TP already on MT5 server + break-even move at +1R, partial close, ATR trailing stop, news-freeze).
  - **Only when a high-probability candidate setup fires (~3 to 12 times per day)** does it wake up the Multi-Agent Brain!

- **Speed Layer 2 — Asynchronous Macro/Regime Context + Triggered Multi-Agent Committee (`TradingAgents` + `QuantHarness` inspired):**
  - **Background Macro/News Digest (runs every 15–30 mins):** Updates economic calendar countdowns (CPI, NFP, FOMC, Powell speeches — crucial for Gold & USD pairs!), DXY / US10Y correlation regime, and sentiment cache.
  - **Fast Tactical Committee (runs ONLY on candidate setup, completes in 2–5 seconds using parallel calls):**
    1. *Technical/Microstructure Analyst* (evaluates multi-TF packet, session high/low, order flow, spread-to-ATR ratio).
    2. *Macro & Cross-Asset Analyst* (checks XAUUSD vs. DXY/EURUSD divergence, upcoming high-impact news within 30 mins).
    3. *Adversarial Bear/Trap Skeptic* ("Why is this Gold setup a fakeout/inducement trap?").
    4. *Head Trader Synthesizer* → Outputs structured Pydantic JSON: `EXECUTE_LONG | EXECUTE_SHORT | PLACE_LIMIT | NO_TRADE`, setup grade (`A+`, `A`, `B`, `SKIP`), invalidation price level, target structure, and reasoning trace.

- **Speed Layer 3 — Deterministic Pre-Trade Risk & Execution Firewall (`deterministic-trader` inspired):**
  - **100% Pure Tested Python Code — AI cannot override this.**
  - Validates: Max risk per trade (e.g., `0.5%–1.0%` equity), Max Daily Loss Lock (e.g., `2.5%–3.0%` hard kill switch), Max Open Correlated USD Exposure (won't go simultaneously max-short `XAUUSD`, `EURUSD`, `GBPUSD`, and `BTCUSD` = 4x hidden long-USD bet!), minimum R:R (`>= 1:1.8` net of current live spread), broker `order_check()` margin & stop-level verification, and idempotent order submission with magic number tagging.

---

## 4. Proposed End-to-End Local Laptop Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                        YOUR LOCAL LAPTOP (100% Self-Hosted)                  │
│                                                                              │
│  ┌─────────────────────────┐         ┌────────────────────────────────────┐  │
│  │  LLM Inference Engine   │         │   Local Data & Telemetry Store     │  │
│  │ • Ollama / vLLM / LMSt. │         │ • DuckDB + Parquet (Ticks & Bars)  │  │
│  │   (Qwen3 / DeepSeek-R1  │         │ • SQLite (Orders, Audit, Memory)   │  │
│  │    / Llama local)       │         │ • Economic Calendar & News Cache   │  │
│  │ • Optional API fallback │         └─────────────────┬──────────────────┘  │
│  └────────────▲────────────┘                           │                     │
│               │ Structured JSON / Tool Calls           ▼                     │
│  ┌────────────┴───────────────────────────────────────────────────────────┐  │
│  │              LAYER 2: MULTI-AGENT DECISION COMMITTEE                   │  │
│  │         (LangGraph / Async Parallel — Woken ONLY on Setup Alert)       │  │
│  │  [Structure/SMC Analyst]  [Macro/News & USD Correlation]  [Trap Critic]│  │
│  │                           ▼                                            │  │
│  │            [Head Trader Verdict + Settled Reflection Memory]           │  │
│  └────────────▲────────────────────────────────────────┬──────────────────┘  │
│               │ Candidate Setup Packet                 │ Trade Proposal      │
│  ┌────────────┴──────────────────────┐   ┌─────────────▼──────────────────┐  │
│  │ LAYER 1: QUANT SCANNER & PRIORITY │   │ LAYER 3: DETERMINISTIC RISK &  │  │
│  │ ROUTER (Fast Python/Rust Loop)    │   │ EXECUTION ENGINE (Zero-Trust)  │  │
│  │ 1. Priority 1: XAUUSD (Gold)      │   │ • Exact Lot Sizing (TickValue) │  │
│  │ 2. Fallback: NAS100, BTCUSD,      │   │ • Daily Drawdown Kill-Switch   │  │
│  │    EURUSD, GBPUSD, USDCAD         │   │ • USD Correlation Cap          │  │
│  │ • Session / Spread / News Filter  │   │ • Spread-to-SL Ratio Gate      │  │
│  │ • Real-time Trailing / BE Manager │   │ • MT5 order_check() Preflight  │  │
│  └────────────▲──────────────────────┘   └─────────────┬──────────────────┘  │
│               │ Ticks / OHLCV                          │ Verified Orders     │
│  ┌────────────┴────────────────────────────────────────▼──────────────────┐  │
│  │               MT5 BRIDGE ADAPTER (Dual-Mode Abstraction)               │  │
│  │  • Mode A (Windows Native): Official `MetaTrader5` Python IPC (<2ms)   │  │
│  │  • Mode B (macOS/Linux Wine or Any OS): Custom MQL5 ZeroMQ Bridge EA   │  │
│  └────────────▲────────────────────────────────────────┬──────────────────┘  │
│               │                                        ▼                     │
│  ┌────────────┴───────────────────────────────────────────────────────────┐  │
│  │             MetaTrader 5 Terminal (Logged into Your Broker)            │  │
│  │            Demo Account (Validation)  ──►  Live Account                │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Implementing Your Gold-First (`XAUUSD`) Multi-Asset Priority Logic

Your requirement is crystal clear: **Focus primarily on `XAUUSD` (Gold) for scalps & intraday setups, and scan the secondary basket (`EURUSD`, `GBPUSD`, `USDCAD`, `BTCUSD`, `NAS100`) when Gold has no clean setup.**

Here is how we codify that cleanly into the Priority Router:

| Tier | Instrument(s) | Prime Intraday / Scalp Windows (UTC) | Specific Microstructure Rules Built Into Scanner |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Primary Focus)** | **`XAUUSD` (Gold)** | **London Open** (`07:00–10:00 UTC`)<br>**NY Open / Overlap** (`12:30–16:00 UTC`) | Highest priority allocation. Checks live spread vs. 5-min ATR (rejects trade if `spread > 12% of SL`), tracks Asian Range High/Low sweeps, London/NY session VWAP, M15/H1 Order Blocks/FVGs, and freezes new entries 15 mins before/after Tier-1 US Red-Folder news (CPI, NFP, FOMC, US Yields). |
| **Tier 2 (High-Beta Indices & Crypto)** | **`NAS100` (`US100` / `USTEC`), `BTCUSD`** | **NAS100:** NY Cash Open (`13:30–16:30 UTC`, `14:30 UTC` winter)<br>**BTCUSD:** 24/7 (especially weekend/off-hours when Gold & FX are closed or quiet!) | Triggered when `XAUUSD` is in low-ADX chop, excessively wide spread, or already at max Gold position limit. `NAS100` Opening Range Breakout / Silver Bullet window (`14:00–15:00 UTC`) is an ideal complement right after Gold's initial `12:30–13:30 UTC` NY push settles. `BTCUSD` requires strict fee+spread hurdle check (many MT5 CFD brokers charge wide BTC spreads — scanner auto-checks if spread makes M5 scalping negative EV and shifts to M15/H1 intraday if needed). |
| **Tier 3 (Major FX Pairs)** | **`EURUSD`, `GBPUSD`, `USDCAD`** | **EUR/GBP:** London (`07:00–11:00 UTC`) & NY Overlap (`12:30–15:30 UTC`)<br>**USDCAD:** NY + CAD data/Oil open (`12:30–16:00 UTC`) | Activated when Gold is ranging/untradable. **Crucial Cross-Correlation Guard:** `XAUUSD`, `EURUSD`, `GBPUSD`, and `USDCAD` all trade heavily against the US Dollar (`DXY`). Risk engine enforces a **Net USD Directional Cap** (e.g., max 1.5% total risk exposed to Short-USD across all open positions simultaneously). |

> **Broker Symbol Auto-Mapper:** Different MT5 brokers name symbols differently (`XAUUSD`, `XAUUSDm`, `XAUUSD.pro`, `GOLD`, `USTEC`, `NAS100`, `US100.cash`). Our `SymbolMapper` module will auto-discover and map your broker's exact symbol names, contract sizes, digit precision, tick values, and trading sessions on startup.

---

## 6. How We Backtest & Prove Profitability Before Risking $1

A major pitfall of LLM trading bots is that you cannot affordably run 500,000 LLM calls to backtest 3 years of M5 candles. Our **3-Stage Verification Funnel** solves this completely on your laptop:

1. **Stage 1: Historical Broker Data Extraction + `VectorBT` Base Strategy Discovery**
   - Pull 1–3 years of **your actual MT5 broker's M1 bars & historical tick spreads** directly into local `Parquet` files (`DuckDB` + `Parquet` catalog).
   - Use **`vectorbt`** (Rust/Numba) to test the *Layer 1 Quantitative Setup Generators* across `XAUUSD` and secondary pairs with realistic spread + commission + slippage penalties.
   - **Rule:** The underlying quantitative setup MUST show positive expectancy (Profit Factor > 1.25, controlled drawdown) *on its own* across walk-forward out-of-sample splits before an LLM ever touches it! An LLM committee can turn a `1.3` Profit Factor strategy into `1.7` by filtering out bad macro/trap regimes — it **cannot** rescue a negative-expectancy strategy eaten alive by spread.

2. **Stage 2: `NautilusTrader` Event-Driven Replay + Agent Ablation Backtest**
   - Replay the exact candidate setups (~500–1,500 historical setups) through **`nautilus_trader`** (or a dedicated tick-accurate MT5 event simulator) with **point-in-time strictness** (the LLM only sees candles/news prior to timestamp `T`, with model cutoff awareness).
   - **Ablation Test (borrowed from `awesome-agentic-trading`):** Compare:
     - *Baseline A:* Pure Quant Signal (0 LLM calls)
     - *Baseline B:* Single LLM Call Filter
     - *Full System:* Multi-Agent Committee + Settled Memory Log
   - Measure net Sharpe, Sortino, Profit Factor, Win Rate, Avg R-Multiple, Max Drawdown, Spread Cost %, and LLM latency/cost per trade.

3. **Stage 3: MT5 Demo Forward-Testing ("Shadow & Paper Harness")**
   - Run live on your laptop connected to an **MT5 Demo Account** on the exact same broker/account type (e.g., Raw Spread / ECN) for at least 2–4 weeks (100+ trades) to verify live slippage, requotes, connection recovery, and sleep/wake resilience before enabling live capital.

---

## 7. Laptop Hardware, Local LLM vs. API & Resilience Considerations

Running on your own laptop without external cloud hosts is 100% achievable, provided we design for **laptop constraints**:

1. **Operating System:**
   - **Windows 10/11 Laptop:** Simplest & fastest — native MT5 terminal + native `MetaTrader5` Python library (`<2ms` IPC).
   - **macOS (Apple Silicon M1/M2/M3/M4) or Linux Laptop:** MT5 runs smoothly via the official MetaQuotes macOS/Linux installer (Wine/Bottles), and our Python agent connects locally over `127.0.0.1` via our **MQL5 ZeroMQ Bridge EA** (or Wine Python bridge).
2. **Local LLM (`Ollama` / `LM Studio` / `vLLM`) vs. Hybrid API:**
   - Because our **Quant Gate** filters out 99% of empty candles and only wakes the agents ~5–12 times a day:
     - **100% Local Offline Option (Zero API Cost):** Requires ~16GB+ RAM / Apple Unified Memory or an 8GB–16GB+ VRAM GPU running `Qwen3-8B/14B` or `DeepSeek-R1-Distill-14B` via **Ollama** with strict JSON schema outputs.
     - **Hybrid Low-Cost API Option (Works even on an 8GB RAM laptop):** Runs all scanning, backtesting, risk, and trade management locally on CPU, and calls fast/cheap reasoning endpoints (e.g., DeepSeek, Groq, Gemini Flash, Claude Haiku/Sonnet, or OpenAI) only on those 5–12 setups/day (costs ~$0.05–$0.30/day total).
   - **Recommendation:** Build a unified **LiteLLM / OpenAI-compatible provider adapter** so you can switch between `ollama/qwen3` (100% local, zero cost) and any API model with a single `.env` line!
3. **Laptop Safety Must-Haves (Crucial!):**
   - **Server-Side Hard SL & TP on Every Order:** Never use "virtual/stealth stops" only in Python on a laptop! If your laptop Wi-Fi drops, battery dies, or lid closes, your SL and TP **must** live inside the MT5 broker server on the order itself.
   - **Prevent OS Sleep During Active Sessions** + **Startup Position Reconciliation** (if the agent restarts, it immediately queries open MT5 positions by `magic_number` and resumes managing them).
   - **Optional Local Web Dashboard + Telegram Mobile Alerts:** Monitor equity, agent debate logs, and hit an emergency **"PANIC CLOSE ALL / PAUSE AGENT"** button from your phone or local browser.

---

## 8. Recommended Tech Stack & Phased Build Roadmap

### Recommended Core Stack (`Python 3.12+` managed via `uv`)
- **Execution & Broker IPC:** `MetaTrader5` (Windows native) + Custom `MQL5 ZeroMQ Bridge EA` (`pyzmq` cross-platform) behind one unified `BrokerAdapter` protocol.
- **Data & Storage:** `Polars` / `Pandas`, `DuckDB` + `Parquet` (historical M1/tick store), `SQLite` (live trade state, idempotency, audit trail, settled agent memory).
- **Quant & Technical Features:** `pandas-ta` / `numba` + custom vectorized **SMC/ICT & Microstructure module** (Session High/Low sweeps, FVG, Order Blocks, VWAP bands, ATR volatility regime, DXY synthetic index).
- **Backtesting & Validation:** `vectorbt` (fast parameter/setup research) + `nautilus_trader` / custom spread-aware tick simulator (high-fidelity validation).
- **Multi-Agent Orchestration:** `LangGraph` + `Pydantic v2` structured outputs + `LiteLLM` / `Ollama` (local-first LLM support).
- **Local Control UI:** Lightweight `FastAPI` + local dashboard (plus optional Telegram bot kill-switch).

### Proposed Step-by-Step Implementation Phases
- **Phase 1 — Foundation, MT5 Local Bridge & Data Pipeline:** Project structure (`uv`), unified MT5 Adapter (Windows native + ZeroMQ EA fallback), Symbol Auto-Mapper (`XAUUSD` + 5 secondary symbols), live health check, and historical M1/Tick downloader to Parquet.
- **Phase 2 — Gold-First Quant Scanner & Backtesting Harness:** Multi-timeframe feature engine, Session/Spread/News filters, XAUUSD scalp/intraday setup detectors + fallback router, and `vectorbt` + event-driven spread-aware backtester to prove baseline edge first.
- **Phase 3 — Deterministic Risk Firewall & Order Management System (OMS):** Exact lot calculator (account currency/tick value aware), Daily Loss Kill Switch, Net USD Correlation Guard, Break-Even / Partial TP / ATR Trailing manager, and idempotent MT5 execution (`dry_run` + `demo` modes).
- **Phase 4 — Multi-Agent LLM Committee & Settled Memory (`TradingAgents` + `QuantHarness` Fusion):** Parallel Technical, Macro/Cross-Asset, and Trap-Critic agents + Head Trader synthesizer + post-trade reflection memory + local Ollama & API support + Ablation backtester.
- **Phase 5 — Local Control Center, Alerts & Demo Forward-Test Burn-in:** Local dashboard (live chart setups, agent debate viewer, PnL, risk meters, Kill Switch) + Telegram notifications + MT5 Demo burn-in checklist.
