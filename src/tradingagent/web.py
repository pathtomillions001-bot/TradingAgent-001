"""FastAPI Local Control Center & Dashboard Server (binds 0.0.0.0, CORS/Proxy friendly)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from tradingagent.backtest import SpreadAwareEventBacktester, generate_realistic_intraday_dataset
from tradingagent.broker.paper_mt5 import DEFAULT_PRICES, PaperMT5Adapter
from tradingagent.config import AppSettings
from tradingagent.engine import TradingAgentEngine

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


class BacktestRequest(BaseModel):
    symbol: str = "XAUUSD"
    bars: int = 1800
    starting_equity: float = 10000.0


class AutopilotToggleRequest(BaseModel):
    enabled: bool


def create_app(settings: AppSettings | None = None) -> FastAPI:
    engine = TradingAgentEngine(settings=settings)
    scan_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app_inst: FastAPI) -> AsyncIterator[None]:
        stop_evt = asyncio.Event()

        async def _autopilot_worker() -> None:
            while not stop_evt.is_set():
                try:
                    if app_inst.state.autopilot_enabled and not engine.risk.kill_switch_active:
                        async with scan_lock:
                            await engine.run_single_cycle(auto_execute=True)
                        app_inst.state.autopilot_cycles += 1
                        app_inst.state.autopilot_last_run = datetime.now(timezone.utc).isoformat()
                except Exception as exc:
                    logger.warning("Autopilot background cycle warning: %s", exc)

                interval = max(3, int(engine.settings.autopilot_interval_seconds))
                try:
                    await asyncio.wait_for(stop_evt.wait(), timeout=float(interval))
                except asyncio.TimeoutError:
                    pass

        worker_task = asyncio.create_task(_autopilot_worker())
        try:
            yield
        finally:
            stop_evt.set()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="TradingAgent-001 Local Control Center",
        version="0.1.0",
        description="100% Zero-Fee Local Windows 10 + MT5 Multi-Agent Scalping & Intraday Engine",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.engine = engine
    app.state.autopilot_enabled = bool(engine.settings.autopilot_enabled)
    app.state.autopilot_cycles = 0
    app.state.autopilot_last_run = None

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        html_path = STATIC_DIR / "index.html"
        return html_path.read_text(encoding="utf-8")

    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        if isinstance(engine.broker, PaperMT5Adapter):
            engine.broker.step_market()

        account = engine.broker.get_account_snapshot()
        positions = engine.broker.get_open_positions()
        ollama_ok, ollama_models = await engine.committee.check_ollama_health()
        resolved_model = engine.committee.resolve_ollama_model(ollama_models)

        watchlist = []
        for prof in engine.scanner.ordered_symbols():
            try:
                q = engine.broker.get_quote(prof.canonical)
                norm_spread = round(prof.canonical_spread_points(q.spread_price), 1)
                unit_info = engine.risk.compute_contract_unit_telemetry(prof, q)
                watchlist.append({
                    "symbol": prof.canonical,
                    "broker_symbol": q.broker_symbol,
                    "tier": prof.tier,
                    "asset_class": prof.asset_class,
                    "available": True,
                    "bid": q.bid,
                    "ask": q.ask,
                    "spread_points": norm_spread,
                    "max_spread_points": prof.max_spread_points,
                    "spread_ok": norm_spread <= prof.max_spread_points,
                    "min_lot": unit_info["min_lot"],
                    "unit_label": unit_info["unit_label"],
                    "min_lot_cap_pct": engine.risk.effective_min_lot_risk_cap_pct(prof.canonical),
                })
            except Exception as exc:
                watchlist.append({
                    "symbol": prof.canonical,
                    "broker_symbol": engine.broker.mapper.to_broker(prof.canonical),
                    "tier": prof.tier,
                    "asset_class": prof.asset_class,
                    "available": False,
                    "bid": 0.0,
                    "ask": 0.0,
                    "spread_points": 0.0,
                    "max_spread_points": prof.max_spread_points,
                    "spread_ok": False,
                    "min_lot": 0.01,
                    "unit_label": "—",
                    "min_lot_cap_pct": engine.risk.effective_min_lot_risk_cap_pct(prof.canonical),
                    "error": str(exc),
                })

        net_usd_pct = engine.risk.compute_open_net_usd_risk_pct(positions, account.equity)

        return {
            "trading_mode": engine.settings.trading_mode,
            "kill_switch_active": engine.risk.kill_switch_active,
            "autopilot": {
                "enabled": bool(app.state.autopilot_enabled),
                "interval_seconds": engine.settings.autopilot_interval_seconds,
                "cycles_completed": app.state.autopilot_cycles,
                "last_run": app.state.autopilot_last_run,
            },
            "ollama": {
                "online": ollama_ok,
                "configured_quick_model": engine.settings.ollama_quick_model,
                "configured_deep_model": engine.settings.ollama_deep_model,
                "active_model": resolved_model,
                "installed_models": ollama_models,
                "active_brain": (
                    f"Ollama Local ({resolved_model})"
                    if (ollama_ok and ollama_models)
                    else "Deterministic Multi-Agent Rubric (Zero-Fee Offline Ready)"
                ),
            },
            "risk_limits": {
                "risk_per_trade_pct": engine.settings.risk_per_trade_pct,
                "max_min_lot_risk_pct": engine.settings.max_min_lot_risk_pct,
                "gold_min_lot_max_risk_pct": engine.settings.gold_min_lot_max_risk_pct,
                "max_daily_loss_pct": engine.settings.max_daily_loss_pct,
                "max_total_drawdown_pct": engine.settings.max_total_drawdown_pct,
                "max_net_usd_risk_pct": engine.settings.max_net_usd_risk_pct,
                "current_net_usd_risk_pct": net_usd_pct,
                "daily_drawdown_pct": account.daily_drawdown_pct,
                "total_drawdown_pct": account.total_drawdown_pct,
            },
            "account": account.model_dump(mode="json"),
            "open_positions": [p.model_dump(mode="json") for p in positions],
            "watchlist": watchlist,
            "recent_audits": engine.memory.get_recent_audits(limit=12),
            "gold_lessons": engine.memory.get_lessons_for_symbol("XAUUSD", limit=5),
        }

    @app.post("/api/scan")
    async def trigger_scan_cycle(auto_execute: bool = True) -> dict[str, Any]:
        async with scan_lock:
            return await engine.run_single_cycle(auto_execute=auto_execute)

    @app.post("/api/autopilot")
    async def toggle_autopilot(req: AutopilotToggleRequest) -> dict[str, Any]:
        app.state.autopilot_enabled = bool(req.enabled)
        return {
            "enabled": app.state.autopilot_enabled,
            "interval_seconds": engine.settings.autopilot_interval_seconds,
            "cycles_completed": app.state.autopilot_cycles,
        }

    @app.post("/api/kill-switch")
    async def activate_kill_switch() -> dict[str, Any]:
        return engine.trigger_emergency_kill_switch()

    @app.post("/api/reset-kill-switch")
    async def deactivate_kill_switch() -> dict[str, Any]:
        return engine.reset_kill_switch()

    @app.post("/api/close/{ticket}")
    async def close_single_ticket(ticket: int) -> dict[str, Any]:
        receipt = engine.broker.close_position(ticket, reason="DASHBOARD_MANUAL")
        return receipt.model_dump(mode="json")

    @app.post("/api/backtest")
    async def run_ablation_backtest(req: BacktestRequest) -> dict[str, Any]:
        sym = req.symbol.upper()
        if sym not in engine.profiles:
            sym = "XAUUSD"
        prof = engine.profiles[sym]
        _, _, _, tick_val = DEFAULT_PRICES.get(sym, (100.0, 0.5, 10.0, 1.0))
        bars_count = max(400, min(5000, req.bars))

        def _compute_ablation() -> tuple[Any, Any]:
            df = generate_realistic_intraday_dataset(symbol=sym, n_bars=bars_count, seed=101)
            tester = SpreadAwareEventBacktester(
                settings=engine.settings,
                profile=prof,
                memory=engine.memory,
                tick_value_per_point=tick_val,
            )
            raw_q = tester.run(df, starting_equity=req.starting_equity, use_agent_committee=False)
            agent_f = tester.run(df, starting_equity=req.starting_equity, use_agent_committee=True)
            return raw_q, agent_f

        raw_quant, agent_filtered = await asyncio.to_thread(_compute_ablation)
        return {
            "symbol": sym,
            "bars_tested": bars_count,
            "raw_quant_baseline": raw_quant.model_dump(mode="json"),
            "agent_committee_filtered": agent_filtered.model_dump(mode="json"),
        }

    return app


app = create_app()
