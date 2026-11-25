"""Main Autonomous Orchestrator Loop uniting Scanner -> Multi-Agent Brain -> Risk Firewall -> MT5."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import time
from typing import Any

from tradingagent.agents import MultiAgentCommittee
from tradingagent.broker import BrokerAdapter
from tradingagent.broker.mt5_windows import MT5WindowsAdapter
from tradingagent.broker.paper_mt5 import PaperMT5Adapter
from tradingagent.config import AppSettings, SymbolProfile, load_symbol_profiles
from tradingagent.memory import StateAndMemoryStore
from tradingagent.models import AgentAction, ExecutionReceipt, RiskCheckResult
from tradingagent.risk import DeterministicRiskFirewall
from tradingagent.scanner import GoldFirstScanner


class LiveTelemetryHub:
    """
    Zero-overhead real-time telemetry state tracking:
      - 6-stage visual pipeline progression (IDLE / WORKING / DONE / VETO / BLOCKED / EXECUTED)
      - Individual live status & latest reasoning for every specialist agent
      - Millisecond ring-buffered event stream ("Thinking Out Loud" mission control log)
    """

    def __init__(self) -> None:
        self.is_running: bool = False
        self.active_stage: int = 0
        self.active_symbol: str = "XAUUSD"
        self.headline: str = "Mission Control Online — Awaiting Gold-First Scan Cycle"
        self.last_cycle_ms: float = 0.0
        self.last_cycle_at: str | None = None
        self.cycles_run: int = 0
        self.events: deque[dict[str, Any]] = deque(maxlen=90)
        self.stages: list[dict[str, Any]] = self._default_stages()
        self.agents: dict[str, dict[str, Any]] = self._default_agents()
        self.emit(
            "SYSTEM",
            "info",
            "XAUUSD",
            "Autonomous Gold-First Multi-Agent Engine armed (2 base slots + 3rd VIP Gold-only slot active).",
        )

    def _default_stages(self) -> list[dict[str, Any]]:
        return [
            {
                "step": 1,
                "key": "be_guard",
                "title": "Position & BE Guard",
                "sub": "+1.0R Break-Even Lock",
                "status": "standby",
                "detail": "Monitoring open tickets",
                "ms": 0.0,
            },
            {
                "step": 2,
                "key": "scanner",
                "title": "Gold-First Scanner",
                "sub": "M5/H1 Sweep + FVG + VWAP",
                "status": "standby",
                "detail": "XAUUSD Priority #1",
                "ms": 0.0,
            },
            {
                "step": 3,
                "key": "specialists",
                "title": "Specialist Debate",
                "sub": "SMC · Macro · Trap Critic",
                "status": "standby",
                "detail": "3-Agent Peer Review",
                "ms": 0.0,
            },
            {
                "step": 4,
                "key": "head_trader",
                "title": "Head Trader Brain",
                "sub": "Ollama LLM / Rubric Verdict",
                "status": "standby",
                "detail": "A/A+ Conviction Filter",
                "ms": 0.0,
            },
            {
                "step": 5,
                "key": "risk_gate",
                "title": "Risk Firewall (2+1)",
                "sub": "Gold VIP 3rd Slot & Lot Math",
                "status": "standby",
                "detail": "Zero-Trust Python Gate",
                "ms": 0.0,
            },
            {
                "step": 6,
                "key": "mt5_router",
                "title": "MT5 Order Router",
                "sub": "Hard Server-Side SL/TP",
                "status": "standby",
                "detail": "Ready for Execution",
                "ms": 0.0,
            },
        ]

    def _default_agents(self) -> dict[str, dict[str, Any]]:
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "Intraday_BE_Guard": {
                "id": "Intraday_BE_Guard",
                "name": "Intraday BE & Trail Guard",
                "role": "Stage 1 · Active Trade Protector",
                "status": "STANDBY",
                "stance": "GUARDING",
                "confidence": 1.0,
                "symbol": "ALL",
                "task": "Watching open MT5 tickets to lock Break-Even at +1.0R.",
                "updated_at": now_iso,
            },
            "GoldFirst_Scanner": {
                "id": "GoldFirst_Scanner",
                "name": "Gold-First Quant Scanner",
                "role": "Stage 2 · M5/H1 Structure Detector",
                "status": "STANDBY",
                "stance": "SCAN_READY",
                "confidence": 1.0,
                "symbol": "XAUUSD",
                "task": "Prioritizing XAUUSD first, then NAS100, BTCUSD & FX Majors.",
                "updated_at": now_iso,
            },
            "SMC_Microstructure_Analyst": {
                "id": "SMC_Microstructure_Analyst",
                "name": "SMC Microstructure Analyst",
                "role": "Stage 3 · Sweep, FVG & VWAP",
                "status": "IDLE",
                "stance": "WAITING",
                "confidence": 0.0,
                "symbol": "XAUUSD",
                "task": "Awaiting candidate setup from Gold-First Scanner.",
                "updated_at": now_iso,
            },
            "Session_Macro_Analyst": {
                "id": "Session_Macro_Analyst",
                "name": "Session & Macro Analyst",
                "role": "Stage 3 · Killzones & H1 Trend",
                "status": "IDLE",
                "stance": "WAITING",
                "confidence": 0.0,
                "symbol": "XAUUSD",
                "task": "Monitoring London / NY Overlap killzone alignment.",
                "updated_at": now_iso,
            },
            "Adversarial_Trap_Critic": {
                "id": "Adversarial_Trap_Critic",
                "name": "Adversarial Trap Skeptic",
                "role": "Stage 3 · Fakeout & Spread Veto",
                "status": "IDLE",
                "stance": "VIGILANT",
                "confidence": 0.0,
                "symbol": "XAUUSD",
                "task": "Ready to veto RSI exhaustion traps, weak ADX breakouts & wide spreads.",
                "updated_at": now_iso,
            },
            "Head_Trader_Brain": {
                "id": "Head_Trader_Brain",
                "name": "Head Trader Synthesizer",
                "role": "Stage 4 · Final Committee Verdict",
                "status": "IDLE",
                "stance": "DISCIPLINED",
                "confidence": 0.0,
                "symbol": "XAUUSD",
                "task": "Synthesizes 3 specialists + settled reflection lessons.",
                "updated_at": now_iso,
            },
            "Deterministic_Risk_Gate": {
                "id": "Deterministic_Risk_Gate",
                "name": "Deterministic Risk Firewall",
                "role": "Stage 5 · 2+1 Gold VIP & Sizing",
                "status": "ARMED",
                "stance": "ENFORCING",
                "confidence": 1.0,
                "symbol": "ALL",
                "task": "2 base slots + 3rd VIP Gold-only slot, DD locks & exact broker lot math.",
                "updated_at": now_iso,
            },
        }

    def emit(self, agent: str, level: str, symbol: str, message: str) -> None:
        now = datetime.now(timezone.utc)
        self.events.appendleft(
            {
                "ts": now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}",
                "iso": now.isoformat(),
                "agent": agent,
                "level": level,
                "symbol": symbol,
                "message": message,
            }
        )

    def set_stage(self, step: int, status: str, detail: str, ms: float | None = None) -> None:
        self.active_stage = step
        idx = step - 1
        if 0 <= idx < len(self.stages):
            self.stages[idx]["status"] = status
            self.stages[idx]["detail"] = detail
            if ms is not None:
                self.stages[idx]["ms"] = round(ms, 1)

    def update_agent(
        self,
        agent_id: str,
        *,
        status: str,
        stance: str | None = None,
        confidence: float | None = None,
        symbol: str | None = None,
        task: str | None = None,
    ) -> None:
        card = self.agents.get(agent_id)
        if not card:
            return
        card["status"] = status
        if stance is not None:
            card["stance"] = stance
        if confidence is not None:
            card["confidence"] = round(float(confidence), 2)
        if symbol is not None:
            card["symbol"] = symbol
        if task is not None:
            card["task"] = task
        card["updated_at"] = datetime.now(timezone.utc).isoformat()

    def snapshot(self) -> dict[str, Any]:
        return {
            "is_running": self.is_running,
            "active_stage": self.active_stage,
            "active_symbol": self.active_symbol,
            "headline": self.headline,
            "last_cycle_ms": round(self.last_cycle_ms, 1),
            "last_cycle_at": self.last_cycle_at,
            "cycles_run": self.cycles_run,
            "stages": list(self.stages),
            "agents": list(self.agents.values()),
            "events": list(self.events)[:45],
        }


class TradingAgentEngine:
    """
    Coordinates the complete local trading pipeline:
      1. Sub-second management of open positions (Break-Even at +1R)
      2. Gold-first (`XAUUSD`) setup scan across priority tiers
      3. Multi-agent committee debate & settled reflection memory check ONLY when setup fires
      4. Deterministic pre-trade risk firewall & exact lot sizing (with 3rd VIP Gold-only slot)
      5. Server-side SL/TP order placement + SQLite immutable audit trail
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or AppSettings()
        self.profiles: dict[str, SymbolProfile] = load_symbol_profiles(self.settings.symbols_config_path)
        self.memory = StateAndMemoryStore(self.settings.sqlite_path)
        self.broker: BrokerAdapter = self._build_broker()
        self.scanner = GoldFirstScanner(self.settings, self.profiles)
        self.committee = MultiAgentCommittee(self.settings, self.memory)
        self.risk = DeterministicRiskFirewall(self.settings, self.profiles)
        self.telemetry = LiveTelemetryHub()
        self.broker.connect()

    def _build_broker(self) -> BrokerAdapter:
        if self.settings.trading_mode in ("mt5_demo", "mt5_live"):
            return MT5WindowsAdapter(self.settings, self.profiles)
        return PaperMT5Adapter(self.settings, self.profiles)

    async def run_single_cycle(self, auto_execute: bool = True) -> dict[str, Any]:
        cycle_t0 = time.perf_counter()
        self.telemetry.is_running = True
        self.telemetry.stages = self.telemetry._default_stages()

        try:
            if isinstance(self.broker, PaperMT5Adapter):
                self.broker.step_market()

            # Stage 1: Manage existing open positions deterministically
            t_s1 = time.perf_counter()
            self.telemetry.headline = "Stage 1/6 · Checking Open Tickets & +1.0R Break-Even Locks..."
            self.telemetry.set_stage(1, "working", "Checking SL/BE triggers")
            self.telemetry.update_agent("Intraday_BE_Guard", status="WORKING", task="Scanning open tickets for +1.0R excursion...")

            mgmt_actions = self.risk.manage_open_positions_intraday(self.broker)
            s1_ms = (time.perf_counter() - t_s1) * 1000.0
            for act in mgmt_actions:
                self.telemetry.emit("BE_GUARD", "bull", "PORTFOLIO", act)

            account = self.broker.get_account_snapshot()
            open_positions = self.broker.get_open_positions()
            self.telemetry.set_stage(
                1,
                "done",
                f"{len(mgmt_actions)} BE shift(s) · {len(open_positions)} open",
                ms=s1_ms,
            )
            self.telemetry.update_agent(
                "Intraday_BE_Guard",
                status="ACTIVE" if open_positions else "STANDBY",
                stance="PROTECTING" if open_positions else "CLEAR",
                task=(
                    mgmt_actions[0]
                    if mgmt_actions
                    else f"Guarding {len(open_positions)} active ticket(s) — server-side SL/TP verified."
                ),
            )

            # Stage 2: Scan XAUUSD first, then fallback priority tiers
            t_s2 = time.perf_counter()
            self.telemetry.active_symbol = "XAUUSD"
            self.telemetry.headline = "Stage 2/6 · Scanning XAUUSD (Tier 1) → Tier 2/3 M5 & H1 Structure..."
            self.telemetry.set_stage(2, "working", "Scanning XAUUSD + 5 pairs")
            self.telemetry.update_agent(
                "GoldFirst_Scanner",
                status="WORKING",
                symbol="XAUUSD",
                task="Computing H1 EMA20/50 bias + M5 Liquidity Sweeps, FVG & VWAP...",
            )

            scan_time = (
                datetime.now(timezone.utc).replace(hour=14)
                if self.settings.trading_mode == "paper"
                else datetime.now(timezone.utc)
            )
            candidates = self.scanner.scan_prioritized(self.broker, open_positions, now_utc=scan_time)
            s2_ms = (time.perf_counter() - t_s2) * 1000.0

            if not candidates:
                self.telemetry.set_stage(2, "done", "No A-grade M5 setup right now", ms=s2_ms)
                self.telemetry.update_agent(
                    "GoldFirst_Scanner",
                    status="STANDBY",
                    stance="PATIENT",
                    symbol="XAUUSD",
                    task=f"Scanned {len(self.profiles)} symbols ({s2_ms:.0f}ms) — no high-probability setup on current M5 bar.",
                )
                self.telemetry.emit(
                    "SCANNER",
                    "info",
                    "XAUUSD",
                    f"Gold-First scan complete ({s2_ms:.0f}ms): market consolidating / awaiting clean sweep or pullback.",
                )
            else:
                cand_syms = ", ".join(f"{c.symbol} ({c.side.value} {c.quant_score:.0f})" for c in candidates)
                self.telemetry.set_stage(2, "done", f"{len(candidates)} candidate(s): {candidates[0].symbol}", ms=s2_ms)
                self.telemetry.update_agent(
                    "GoldFirst_Scanner",
                    status="APPROVED",
                    stance=candidates[0].side.value,
                    confidence=min(0.99, candidates[0].quant_score / 100.0),
                    symbol=candidates[0].symbol,
                    task=f"Detected {len(candidates)} setup(s): {cand_syms}",
                )
                self.telemetry.emit(
                    "SCANNER",
                    "gold" if candidates[0].symbol == "XAUUSD" else "info",
                    candidates[0].symbol,
                    f"Triggered {len(candidates)} candidate(s): {cand_syms}",
                )

            evaluated_results: list[dict[str, Any]] = []
            gold_executed = False

            for cand in candidates:
                if gold_executed and cand.symbol != "XAUUSD":
                    self.telemetry.emit(
                        "SCANNER",
                        "gold",
                        cand.symbol,
                        f"Skipping {cand.symbol} fallback — Tier-1 Gold (XAUUSD) already executed this cycle.",
                    )
                    continue

                self.telemetry.active_symbol = cand.symbol
                # Stage 3 & 4: Specialist Debate + Head Trader Synthesis
                t_s3 = time.perf_counter()
                self.telemetry.headline = f"Stage 3-4/6 · Specialist Trio & Head Trader debating {cand.symbol} {cand.side.value}..."
                self.telemetry.set_stage(3, "working", f"Debating {cand.symbol} {cand.side.value}")
                self.telemetry.set_stage(4, "working", f"Synthesizing {cand.symbol}")
                for sp_id in ("SMC_Microstructure_Analyst", "Session_Macro_Analyst", "Adversarial_Trap_Critic", "Head_Trader_Brain"):
                    self.telemetry.update_agent(
                        sp_id,
                        status="WORKING",
                        symbol=cand.symbol,
                        task=f"Evaluating {cand.symbol} {cand.side.value} ({cand.setup_type.value})...",
                    )

                decision = await self.committee.evaluate_candidate(cand)
                s34_ms = (time.perf_counter() - t_s3) * 1000.0

                # Update each specialist card from actual opinions
                any_veto = False
                for op in decision.opinions:
                    is_veto = op.stance == "VETO"
                    if is_veto:
                        any_veto = True
                    msg_txt = (op.risk_flags[0] if op.risk_flags else (op.key_arguments[0] if op.key_arguments else op.stance))
                    self.telemetry.update_agent(
                        op.agent_name,
                        status="VETO" if is_veto else "APPROVED",
                        stance=op.stance,
                        confidence=op.confidence,
                        symbol=cand.symbol,
                        task=msg_txt,
                    )
                    self.telemetry.emit(
                        op.agent_name.split("_")[0].upper(),
                        "warn" if is_veto else ("bull" if op.stance == "BULLISH" else "info"),
                        cand.symbol,
                        f"{op.stance} ({op.confidence * 100:.0f}%): {msg_txt}",
                    )

                self.telemetry.set_stage(
                    3,
                    "veto" if any_veto else "done",
                    f"{cand.symbol}: {'Critic VETO' if any_veto else '3/3 Reviewed'}",
                    ms=round(s34_ms * 0.35, 1),
                )

                should_exec = decision.action in (AgentAction.EXECUTE_LONG, AgentAction.EXECUTE_SHORT)
                self.telemetry.set_stage(
                    4,
                    "done" if should_exec else "veto",
                    f"{decision.action.value} ({decision.grade.value})",
                    ms=decision.latency_ms,
                )
                self.telemetry.update_agent(
                    "Head_Trader_Brain",
                    status="APPROVED" if should_exec else "VETO",
                    stance=f"{decision.action.value} [{decision.grade.value}]",
                    confidence=decision.confidence,
                    symbol=cand.symbol,
                    task=f"[{decision.llm_mode_used}] {decision.thesis_summary}",
                )
                self.telemetry.emit(
                    "HEAD_TRADER",
                    "bull" if should_exec else "warn",
                    cand.symbol,
                    f"{decision.action.value} ({decision.grade.value} · {decision.latency_ms:.0f}ms): {decision.thesis_summary}",
                )

                # Stage 5: Deterministic Pre-Trade Risk Firewall
                t_s5 = time.perf_counter()
                self.telemetry.headline = f"Stage 5/6 · Risk Firewall validating {cand.symbol} lots & 2+1 Gold VIP caps..."
                self.telemetry.set_stage(5, "working", f"Sizing {cand.symbol}")
                try:
                    quote = self.broker.get_quote(cand.symbol)
                except Exception as exc:
                    self.telemetry.set_stage(5, "veto", f"Quote unavailable: {cand.symbol}")
                    self.telemetry.emit("RISK_GATE", "warn", cand.symbol, f"Broker quote error: {exc}")
                    continue

                risk_res: RiskCheckResult = self.risk.validate_and_size(
                    setup=cand,
                    account=account,
                    quote=quote,
                    open_positions=self.broker.get_open_positions(),
                )
                s5_ms = (time.perf_counter() - t_s5) * 1000.0

                if risk_res.approved:
                    self.telemetry.set_stage(
                        5,
                        "done",
                        f"Approved {risk_res.calculated_lots:.2f}L (${risk_res.risk_usd:.2f})",
                        ms=s5_ms,
                    )
                    self.telemetry.update_agent(
                        "Deterministic_Risk_Gate",
                        status="APPROVED",
                        stance=f"{risk_res.calculated_lots:.2f} LOTS OK",
                        confidence=1.0,
                        symbol=cand.symbol,
                        task=f"Sized {risk_res.calculated_lots:.2f} lots (${risk_res.risk_usd:.2f} = {risk_res.risk_pct:.2f}% risk, Net R:R {risk_res.net_rr_after_spread:.2f}).",
                    )
                else:
                    first_rej = risk_res.rejection_reasons[0]
                    self.telemetry.set_stage(5, "veto", first_rej.split(":")[0], ms=s5_ms)
                    self.telemetry.update_agent(
                        "Deterministic_Risk_Gate",
                        status="BLOCKED",
                        stance="FIREWALL_BLOCK",
                        confidence=1.0,
                        symbol=cand.symbol,
                        task=first_rej,
                    )
                    self.telemetry.emit("RISK_GATE", "warn", cand.symbol, f"Blocked {cand.symbol}: {first_rej}")

                # Stage 6: Broker Order Placement
                receipt: ExecutionReceipt | None = None
                if should_exec and risk_res.approved and auto_execute:
                    t_s6 = time.perf_counter()
                    self.telemetry.headline = f"Stage 6/6 · Routing {cand.symbol} {cand.side.value} ({risk_res.calculated_lots:.2f} lots) to MT5..."
                    self.telemetry.set_stage(6, "working", f"Sending {cand.symbol} to MT5")
                    receipt = self.broker.place_market_order(
                        canonical_symbol=cand.symbol,
                        side=cand.side,
                        volume=risk_res.calculated_lots,
                        sl=risk_res.adjusted_sl,
                        tp=risk_res.adjusted_tp,
                        comment=f"TA_{decision.grade.value}_{cand.setup_type.value[:10]}",
                        initial_risk_usd=risk_res.risk_usd,
                        setup_type=cand.setup_type.value,
                    )
                    s6_ms = (time.perf_counter() - t_s6) * 1000.0
                    if receipt.success:
                        self.risk.record_trade_timestamp(cand.symbol)
                        if cand.symbol == "XAUUSD":
                            gold_executed = True
                        self.telemetry.set_stage(
                            6,
                            "done",
                            f"FILLED #{receipt.ticket} @{receipt.fill_price}",
                            ms=s6_ms,
                        )
                        self.telemetry.emit(
                            "MT5_EXEC",
                            "exec",
                            cand.symbol,
                            f"EXECUTED #{receipt.ticket}: {cand.side.value} {receipt.volume}L {cand.symbol} @ {receipt.fill_price} (SL {receipt.sl} | TP {receipt.tp})",
                        )
                    else:
                        self.telemetry.set_stage(6, "veto", f"MT5 Reject ({receipt.retcode})", ms=s6_ms)
                        self.telemetry.emit(
                            "MT5_EXEC",
                            "bear",
                            cand.symbol,
                            f"MT5 Order Rejected ({receipt.retcode}): {receipt.message}",
                        )
                elif not auto_execute and should_exec and risk_res.approved:
                    self.telemetry.set_stage(6, "standby", "Dry-Run (Execution Skipped)")
                    self.telemetry.emit(
                        "MT5_EXEC",
                        "info",
                        cand.symbol,
                        f"Dry-run validated {cand.symbol} {cand.side.value} ({risk_res.calculated_lots:.2f}L ready).",
                    )

                self.memory.record_audit_cycle(
                    setup_dict=cand.model_dump(mode="json"),
                    decision=decision,
                    risk=risk_res,
                    receipt=receipt,
                )

                evaluated_results.append({
                    "setup": cand.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                    "risk": risk_res.model_dump(mode="json") if risk_res else None,
                    "receipt": receipt.model_dump(mode="json") if receipt else None,
                })

            total_ms = (time.perf_counter() - cycle_t0) * 1000.0
            self.telemetry.last_cycle_ms = total_ms
            self.telemetry.last_cycle_at = datetime.now(timezone.utc).isoformat()
            self.telemetry.cycles_run += 1
            self.telemetry.headline = (
                f"Cycle #{self.telemetry.cycles_run} Complete ({total_ms:.0f}ms) — "
                f"{len(candidates)} setup(s) evaluated · Monitoring MT5 Live"
            )

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trading_mode": self.settings.trading_mode,
                "management_actions": mgmt_actions,
                "candidates_found": len(candidates),
                "evaluations": evaluated_results,
            }
        finally:
            self.telemetry.is_running = False
            self.telemetry.active_stage = 0

    def trigger_emergency_kill_switch(self) -> dict[str, Any]:
        self.risk.kill_switch_active = True
        receipts = self.broker.close_all_positions(reason="EMERGENCY_KILL_SWITCH")
        self.telemetry.headline = "🛑 EMERGENCY KILL SWITCH ACTIVE — All Positions Liquidated"
        self.telemetry.emit(
            "KILL_SWITCH",
            "bear",
            "ALL",
            f"Operator triggered EMERGENCY KILL SWITCH — closed {len(receipts)} open position(s).",
        )
        return {
            "kill_switch_active": True,
            "closed_positions": [r.model_dump(mode="json") for r in receipts],
        }

    def reset_kill_switch(self) -> dict[str, Any]:
        self.risk.kill_switch_active = False
        self.telemetry.headline = "System Re-Armed — Kill Switch Cleared"
        self.telemetry.emit("SYSTEM", "bull", "ALL", "Operator cleared Kill Switch — trading engine re-armed.")
        return {"kill_switch_active": False}
