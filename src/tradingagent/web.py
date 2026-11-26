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
import pandas as pd
from pydantic import BaseModel

from tradingagent.backtest import SpreadAwareEventBacktester, load_backtest_dataset
from tradingagent.broker.paper_mt5 import PaperMT5Adapter
from tradingagent.config import AppSettings
from tradingagent.engine import TradingAgentEngine
from tradingagent.indicators import compute_ema, detect_fvg_and_sweeps
from tradingagent.models import OpenPosition, Side

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


class BacktestRequest(BaseModel):
    symbol: str = "XAUUSD"
    bars: int = 8500
    starting_equity: float = 0.0  # 0.0 = use live connected MT5 account equity by default!
    source: str = "auto"          # "auto" = Real MT5 History -> Parquet Cache -> Synthetic Fallback


class AutopilotToggleRequest(BaseModel):
    enabled: bool


def _enrich_position_lifecycle(pos: OpenPosition) -> dict[str, Any]:
    """Computes live R-multiple, 5-stage Chandelier trade lifecycle step, and SL->Entry->BE->TP progress geometry."""
    d = pos.model_dump(mode="json")
    is_long = pos.side == Side.LONG

    sl_at_or_past_be = False
    if pos.sl > 0:
        sl_at_or_past_be = (pos.sl > pos.entry_price) if is_long else (pos.sl < pos.entry_price)
    be_locked = bool(pos.be_moved or sl_at_or_past_be)

    tp_dist = abs(pos.tp - pos.entry_price) if pos.tp > 0 else 0.0
    raw_sl_dist = abs(pos.entry_price - pos.sl) if pos.sl > 0 else 0.0
    initial_r_dist = (tp_dist / 3.20) if (be_locked and tp_dist > 0) else raw_sl_dist
    if initial_r_dist <= 1e-9:
        initial_r_dist = max(tp_dist / 3.20, abs(pos.entry_price) * 0.0015, 1e-5)

    favorable = (pos.current_price - pos.entry_price) if is_long else (pos.entry_price - pos.current_price)
    current_r = round(favorable / initial_r_dist, 2)
    target_r = round(tp_dist / initial_r_dist, 2) if tp_dist > 0 else 3.20
    locked_r = round((abs(pos.sl - pos.entry_price) / initial_r_dist), 2) if sl_at_or_past_be else 0.0

    span_r = 1.0 + max(1.5, target_r)
    progress_pct = round(min(100.0, max(0.0, ((current_r + 1.0) / span_r) * 100.0)), 1)
    entry_pct = round((1.0 / span_r) * 100.0, 1)
    be_pct = round(((1.0 + 1.15) / span_r) * 100.0, 1)

    is_chandelier = pos.trail_stage >= 2 or locked_r >= 0.45
    if is_chandelier:
        step = 5
        step_label = f"Chandelier ATR Trail Active (+{locked_r:.2f}R Locked)"
    elif be_locked:
        step = 4
        step_label = f"Profit-Plus BE Shield (+{locked_r:.2f}R Banked)"
    elif current_r >= 0.30:
        step = 3
        rem_r = max(0.0, 1.15 - current_r)
        step_label = f"Pushing for +1.15R BE Lock ({rem_r:.2f}R away)"
    elif pos.sl > 0 and pos.tp > 0:
        step = 2
        step_label = "Structural Wick SL & 3.2R TP Armed"
    else:
        step = 1
        step_label = "MT5 Market Order Filled"

    be_trigger_price = (
        round(pos.entry_price + 1.15 * initial_r_dist, 5)
        if is_long
        else round(pos.entry_price - 1.15 * initial_r_dist, 5)
    )

    d.update({
        "be_locked": be_locked,
        "locked_r": locked_r,
        "current_r": current_r,
        "target_r": target_r,
        "be_trigger_price": be_trigger_price,
        "progress_pct": progress_pct,
        "entry_pct": entry_pct,
        "be_pct": be_pct,
        "lifecycle_step": step,
        "lifecycle_label": step_label,
    })
    return d


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
                        app_inst.state.next_autopilot_at = (
                            datetime.now(timezone.utc).timestamp() + float(engine.settings.autopilot_interval_seconds)
                        )
                except Exception as exc:
                    logger.warning("Autopilot background cycle warning: %s", exc)

                interval = max(3, int(engine.settings.autopilot_interval_seconds))
                app_inst.state.next_autopilot_at = datetime.now(timezone.utc).timestamp() + float(interval)
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
        version="0.2.0",
        description="100% Zero-Fee Local Windows 10 + MT5 Multi-Agent Mission Control HUD",
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
    app.state.next_autopilot_at = datetime.now(timezone.utc).timestamp() + float(engine.settings.autopilot_interval_seconds)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        html_path = STATIC_DIR / "index.html"
        return html_path.read_text(encoding="utf-8")

    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        if isinstance(engine.broker, PaperMT5Adapter):
            engine.broker.step_market()

        now_utc = datetime.now(timezone.utc)
        account = engine.broker.get_account_snapshot()
        positions = engine.broker.get_open_positions()
        enriched_positions = [_enrich_position_lifecycle(p) for p in positions]

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
        rem_sec = max(0.0, round(app.state.next_autopilot_at - now_utc.timestamp(), 1))

        base_slots = engine.settings.max_open_positions
        vip_gold_enabled = bool(engine.settings.gold_reserved_extra_slot)
        total_max_slots = base_slots + (1 if vip_gold_enabled else 0)

        return {
            "server_time_utc": now_utc.strftime("%H:%M:%S UTC"),
            "session_label": engine.scanner.classify_session(now_utc.hour),
            "trading_mode": engine.settings.trading_mode,
            "kill_switch_active": engine.risk.kill_switch_active,
            "autopilot": {
                "enabled": bool(app.state.autopilot_enabled),
                "interval_seconds": engine.settings.autopilot_interval_seconds,
                "seconds_until_next": rem_sec,
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
            "slot_policy": {
                "base_max_positions": base_slots,
                "gold_reserved_extra_slot": vip_gold_enabled,
                "total_max_with_gold": total_max_slots,
                "open_total": len(positions),
                "open_gold": sum(1 for p in positions if p.symbol == "XAUUSD"),
                "open_non_gold": sum(1 for p in positions if p.symbol != "XAUUSD"),
            },
            "risk_limits": {
                "risk_per_trade_pct": engine.settings.risk_per_trade_pct,
                "max_min_lot_risk_pct": engine.settings.max_min_lot_risk_pct,
                "gold_min_lot_max_risk_pct": engine.settings.gold_min_lot_max_risk_pct,
                "max_daily_loss_pct": engine.settings.max_daily_loss_pct,
                "max_total_drawdown_pct": engine.settings.max_total_drawdown_pct,
                "max_net_usd_risk_pct": engine.settings.max_net_usd_risk_pct,
                "max_open_positions": base_slots,
                "gold_reserved_extra_slot": vip_gold_enabled,
                "current_net_usd_risk_pct": net_usd_pct,
                "daily_drawdown_pct": account.daily_drawdown_pct,
                "total_drawdown_pct": account.total_drawdown_pct,
            },
            "telemetry": engine.telemetry.snapshot(),
            "account": account.model_dump(mode="json"),
            "open_positions": enriched_positions,
            "watchlist": watchlist,
            "recent_audits": engine.memory.get_recent_audits(limit=15),
            "gold_lessons": engine.memory.get_lessons_for_symbol("XAUUSD", limit=5),
        }

    @app.get("/api/candles/{symbol}")
    async def get_symbol_candles(symbol: str, limit: int = 85) -> dict[str, Any]:
        sym = symbol.upper()
        if sym not in engine.profiles:
            sym = "XAUUSD"
        prof = engine.profiles[sym]
        n = max(40, min(160, limit))

        def _fetch_chart_payload() -> dict[str, Any]:
            m5 = engine.broker.get_bars(sym, "M5", max(140, n + 55))
            h1 = engine.broker.get_bars(sym, "H1", 80)
            q = engine.broker.get_quote(sym)

            h1_ema20 = float(compute_ema(h1["close"], 20).iloc[-1])
            h1_ema50 = float(compute_ema(h1["close"], 50).iloc[-1])
            h1_close = float(h1["close"].iloc[-1])
            if h1_close > h1_ema20 >= h1_ema50:
                htf_bias = "BULLISH"
            elif h1_close < h1_ema20 <= h1_ema50:
                htf_bias = "BEARISH"
            else:
                htf_bias = "NEUTRAL"

            feat = detect_fvg_and_sweeps(m5).tail(n)
            candles = []
            for _, r in feat.iterrows():
                ts_val = pd.to_datetime(r["timestamp"], utc=True)
                p_low = r["prior_liq_low"]
                p_high = r["prior_liq_high"]
                candles.append({
                    "ts": ts_val.strftime("%H:%M"),
                    "iso": ts_val.isoformat(),
                    "o": round(float(r["open"]), 5),
                    "h": round(float(r["high"]), 5),
                    "l": round(float(r["low"]), 5),
                    "c": round(float(r["close"]), 5),
                    "ema20": round(float(r["ema_20"]), 5),
                    "ema50": round(float(r["ema_50"]), 5),
                    "vwap": round(float(r["vwap"]), 5),
                    "fvg_bull": bool(r["bullish_fvg"]),
                    "fvg_bear": bool(r["bearish_fvg"]),
                    "sweep_low": bool(pd.notna(p_low) and float(r["low"]) < float(p_low)),
                    "sweep_high": bool(pd.notna(p_high) and float(r["high"]) > float(p_high)),
                })

            last = feat.iloc[-1]
            open_pos = next((_enrich_position_lifecycle(p) for p in engine.broker.get_open_positions() if p.symbol == sym), None)
            recent_sym_audit = next(
                (a for a in engine.memory.get_recent_audits(limit=20) if a.get("setup", {}).get("symbol") == sym),
                None,
            )

            return {
                "symbol": sym,
                "broker_symbol": q.broker_symbol,
                "bid": q.bid,
                "ask": q.ask,
                "spread_points": round(prof.canonical_spread_points(q.spread_price), 1),
                "htf_bias": htf_bias,
                "rsi": round(float(last["rsi"]), 1),
                "adx": round(float(last["adx"]), 1),
                "atr": round(float(last["atr"]), 5),
                "vwap": round(float(last["vwap"]), 5),
                "candles": candles,
                "active_position": open_pos,
                "latest_audit": recent_sym_audit,
            }

        try:
            return await asyncio.to_thread(_fetch_chart_payload)
        except Exception as exc:
            return {
                "symbol": sym,
                "broker_symbol": engine.broker.mapper.to_broker(sym),
                "available": False,
                "error": str(exc),
                "candles": [],
            }

    @app.post("/api/scan")
    async def trigger_scan_cycle(auto_execute: bool = True) -> dict[str, Any]:
        async with scan_lock:
            out = await engine.run_single_cycle(auto_execute=auto_execute)
            app.state.next_autopilot_at = datetime.now(timezone.utc).timestamp() + float(engine.settings.autopilot_interval_seconds)
            return out

    @app.post("/api/autopilot")
    async def toggle_autopilot(req: AutopilotToggleRequest) -> dict[str, Any]:
        app.state.autopilot_enabled = bool(req.enabled)
        app.state.next_autopilot_at = datetime.now(timezone.utc).timestamp() + float(engine.settings.autopilot_interval_seconds)
        engine.telemetry.emit(
            "AUTOPILOT",
            "bull" if app.state.autopilot_enabled else "warn",
            "ALL",
            f"15s Gold-First Autopilot {'ENABLED' if app.state.autopilot_enabled else 'PAUSED'} by operator.",
        )
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
        engine.telemetry.emit(
            "OPERATOR",
            "warn",
            receipt.symbol or "MT5",
            f"Manual close requested for ticket #{ticket}: {receipt.message}",
        )
        return receipt.model_dump(mode="json")

    @app.post("/api/backtest")
    async def run_ablation_backtest(req: BacktestRequest) -> dict[str, Any]:
        sym = req.symbol.upper()
        if sym not in engine.profiles:
            sym = "XAUUSD"
        prof = engine.profiles[sym]
        bars_count = max(400, min(60000, req.bars))
        engine.telemetry.emit(
            "BACKTEST",
            "gold" if sym == "XAUUSD" else "info",
            sym,
            f"Starting Real-History Spread-Aware Ablation Backtest on {sym} ({bars_count:,} M5 bars)...",
        )

        def _compute_ablation() -> tuple[Any, Any]:
            start_eq = (
                req.starting_equity
                if req.starting_equity > 0
                else engine.broker.get_account_snapshot().equity
            )
            bundle = load_backtest_dataset(
                symbol=sym,
                bars=bars_count,
                settings=engine.settings,
                profile=prof,
                broker=engine.broker,
                source=req.source,
            )
            tester = SpreadAwareEventBacktester(
                settings=engine.settings,
                profile=prof,
                memory=engine.memory,
                tick_value_per_point=bundle.tick_value_per_point,
            )
            raw_q = tester.run(
                bundle.df,
                starting_equity=start_eq,
                use_agent_committee=False,
                data_source=bundle.data_source,
                is_real_history=bundle.is_real_history,
                cache_path=bundle.cache_path,
            )
            agent_f = tester.run(
                bundle.df,
                starting_equity=start_eq,
                use_agent_committee=True,
                data_source=bundle.data_source,
                is_real_history=bundle.is_real_history,
                cache_path=bundle.cache_path,
            )
            return raw_q, agent_f

        raw_quant, agent_filtered = await asyncio.to_thread(_compute_ablation)
        engine.telemetry.emit(
            "BACKTEST",
            "bull",
            sym,
            f"Backtest finished [{agent_filtered.data_source}]: Committee {agent_filtered.net_return_pct:+.2f}% vs Raw {raw_quant.net_return_pct:+.2f}%.",
        )
        return {
            "symbol": sym,
            "bars_tested": agent_filtered.bars_tested,
            "data_source": agent_filtered.data_source,
            "is_real_history": agent_filtered.is_real_history,
            "start_timestamp": agent_filtered.start_timestamp,
            "end_timestamp": agent_filtered.end_timestamp,
            "cache_path": agent_filtered.cache_path,
            "starting_equity": agent_filtered.starting_equity,
            "raw_quant_baseline": raw_quant.model_dump(mode="json"),
            "agent_committee_filtered": agent_filtered.model_dump(mode="json"),
        }

    return app


app = create_app()
