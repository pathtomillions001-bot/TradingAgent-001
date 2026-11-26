"""Layer 2: Zero-Fee Multi-Agent Trading Committee (Ollama Local LLM + Deterministic Institutional Rubric)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import time
import httpx

from tradingagent.config import AppSettings
from tradingagent.memory import StateAndMemoryStore
from tradingagent.models import (
    AgentAction,
    CandidateSetup,
    CommitteeDecision,
    SetupGrade,
    Side,
    SpecialistOpinion,
)


class MultiAgentCommittee:
    """
    Combines TauricResearch/TradingAgents adversarial role specialization with QuantHarness
    short-horizon microstructure & regime focus — tuned specifically for a 16GB RAM Windows 10 laptop:
      - Specialist 1: SMC & Microstructure Analyst (M5/H1 structure, VWAP, FVG, Sweep, RVOL)
      - Specialist 2: Session, Macro & Regime Analyst (Killzones, H1 trend, ADX & Efficiency Ratio)
      - Specialist 3: Adversarial Trap & Spread Skeptic (Vetoes extended chases, weak RVOL, chop & counter-H1 traps)
      - Head Trader Synthesizer: Weighs all 3 specialists + Settled Memory Lessons -> Typed Verdict
    """

    def __init__(self, settings: AppSettings, memory: StateAndMemoryStore) -> None:
        self.settings = settings
        self.memory = memory

    async def check_ollama_health(self) -> tuple[bool, list[str]]:
        if not self.settings.llm_enabled or self.settings.llm_provider != "ollama":
            return False, []
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.settings.ollama_base_url.rstrip('/')}/api/tags")
                if resp.status_code == 200:
                    models = [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
                    return True, models
        except Exception:
            pass
        return False, []

    def resolve_ollama_model(self, installed_models: list[str]) -> str:
        """Picks configured deep model if installed, otherwise quick model (e.g. qwen3:4b), else first installed."""
        if not installed_models:
            return self.settings.ollama_deep_model
        if self.settings.ollama_deep_model in installed_models:
            return self.settings.ollama_deep_model
        if self.settings.ollama_quick_model in installed_models:
            return self.settings.ollama_quick_model
        return installed_models[0]

    def _build_heuristic_opinions(self, setup: CandidateSetup) -> list[SpecialistOpinion]:
        """Deterministic baseline specialist analysis (always runs first so numbers are never hallucinated)."""
        expected_stance = "BULLISH" if setup.side == Side.LONG else "BEARISH"

        # 1. Microstructure, RVOL & SMC Specialist
        smc_args = list(setup.notes)
        smc_flags: list[str] = []
        smc_conf = min(0.95, setup.quant_score / 100.0)

        favorable_ext = setup.vwap_distance_atr if setup.side == Side.LONG else -setup.vwap_distance_atr
        is_blowoff = favorable_ext > 6.5 and (
            (setup.side == Side.LONG and setup.rsi > 65.0) or (setup.side == Side.SHORT and setup.rsi < 35.0)
        )
        if is_blowoff:
            smc_flags.append(f"Parabolic blow-off extension ({setup.vwap_distance_atr:+.2f} ATR from VWAP, RSI={setup.rsi:.1f})")
            smc_conf -= 0.16
        elif favorable_ext >= 0.0:
            smc_args.append(f"Trading on institutional side of Session VWAP ({setup.vwap_distance_atr:+.2f} ATR)")
            smc_conf += 0.03
        else:
            smc_args.append(f"Mean-reversion discount to Session VWAP ({setup.vwap_distance_atr:+.2f} ATR)")

        if setup.rvol >= 1.15:
            smc_args.append(f"Strong institutional volume participation (RVOL {setup.rvol:.2f}x)")
            smc_conf += 0.05
        elif setup.rvol < 0.82:
            smc_flags.append(f"Sub-average tick volume (RVOL {setup.rvol:.2f}x)")
            smc_conf -= 0.12

        op_smc = SpecialistOpinion(
            agent_name="SMC_Microstructure_Analyst",
            stance=expected_stance,
            confidence=round(min(0.96, max(0.30, smc_conf)), 2),
            key_arguments=smc_args,
            risk_flags=smc_flags,
        )

        # 2. Macro, Regime & Session Specialist
        macro_args: list[str] = [f"Detected market regime: {setup.regime} (ER={setup.efficiency_ratio:.2f})"]
        macro_flags: list[str] = []
        macro_stance = expected_stance
        macro_conf = 0.76

        if setup.session_label in ("LONDON_SESSION", "NY_OVERLAP_PRIME"):
            macro_args.append(f"Peak institutional liquidity killzone: {setup.session_label}")
            macro_conf += 0.09
        else:
            macro_flags.append(f"Late/secondary session liquidity ({setup.session_label})")
            macro_conf -= 0.12

        if (setup.side == Side.LONG and setup.htf_bias == "BULLISH") or (
            setup.side == Side.SHORT and setup.htf_bias == "BEARISH"
        ):
            macro_args.append(f"Aligned with H1 {setup.htf_bias} macro trend")
            macro_conf += 0.09
        elif setup.htf_bias == "NEUTRAL":
            macro_args.append("H1 trend transitional — relying on structural wick stop + Chandelier trail")
            macro_conf -= 0.04
        else:
            macro_flags.append(f"Counter-trend entry against H1 {setup.htf_bias} bias")
            macro_conf -= 0.24
            macro_stance = "NEUTRAL"

        op_macro = SpecialistOpinion(
            agent_name="Session_Macro_Analyst",
            stance=macro_stance,
            confidence=round(min(0.96, max(0.25, macro_conf)), 2),
            key_arguments=macro_args,
            risk_flags=macro_flags,
        )

        # 3. Adversarial Trap, Regime & Spread Skeptic (Hard Veto Power)
        trap_args: list[str] = []
        trap_flags: list[str] = []
        trap_stance = "NEUTRAL"
        trap_conf = 0.68

        is_counter_h1 = (setup.side == Side.LONG and setup.htf_bias == "BEARISH") or (
            setup.side == Side.SHORT and setup.htf_bias == "BULLISH"
        )

        if setup.spread_to_sl_ratio > 0.09:
            trap_flags.append(f"High spread drag ({setup.spread_to_sl_ratio * 100:.1f}% of SL) erodes expectancy")
            trap_stance = "VETO"
            trap_conf = 0.86
        elif (setup.side == Side.LONG and setup.rsi > 69.5) or (setup.side == Side.SHORT and setup.rsi < 30.5):
            trap_flags.append(f"Momentum exhaustion trap (M5 RSI={setup.rsi:.1f}) — refusing late chase")
            trap_stance = "VETO"
            trap_conf = 0.88
        elif is_blowoff:
            trap_flags.append(f"Parabolic blow-off ({favorable_ext:.1f} ATR beyond VWAP + RSI={setup.rsi:.1f})")
            trap_stance = "VETO"
            trap_conf = 0.85
        elif is_counter_h1 and setup.quant_score < 85.0:
            trap_flags.append(f"Counter-H1 ({setup.htf_bias}) trade without extreme capitulation score")
            trap_stance = "VETO"
            trap_conf = 0.87
        elif setup.rvol < 0.80 and setup.setup_type.value != "LIQUIDITY_SWEEP_FVG":
            trap_flags.append(f"Thin volume (RVOL {setup.rvol:.2f}x) — high false-breakout probability")
            trap_stance = "VETO"
            trap_conf = 0.82
        elif (
            setup.setup_type.value == "VWAP_EMA_PULLBACK"
            and setup.efficiency_ratio < 0.24
            and setup.adx < 21.0
        ):
            trap_flags.append(f"Pullback in non-persistent chop (ER={setup.efficiency_ratio:.2f}, ADX={setup.adx:.1f})")
            trap_stance = "VETO"
            trap_conf = 0.84
        else:
            trap_args.append(
                f"Cleared trap audit (RSI={setup.rsi:.1f}, RVOL={setup.rvol:.2f}x, Spread/SL={setup.spread_to_sl_ratio * 100:.1f}%)"
            )

        op_trap = SpecialistOpinion(
            agent_name="Adversarial_Trap_Critic",
            stance=trap_stance,
            confidence=trap_conf,
            key_arguments=trap_args,
            risk_flags=trap_flags,
        )

        return [op_smc, op_macro, op_trap]

    async def _call_ollama_head_trader(
        self,
        setup: CandidateSetup,
        opinions: list[SpecialistOpinion],
        lessons: list[str],
        model_name: str,
    ) -> dict | None:
        prompt = {
            "symbol": setup.symbol,
            "priority_tier": setup.tier,
            "proposed_side": setup.side.value,
            "setup_type": setup.setup_type.value,
            "market_regime": setup.regime,
            "quant_score": setup.quant_score,
            "htf_bias": setup.htf_bias,
            "session": setup.session_label,
            "rvol": setup.rvol,
            "efficiency_ratio": setup.efficiency_ratio,
            "rr_ratio": setup.rr_ratio,
            "spread_to_sl_ratio": setup.spread_to_sl_ratio,
            "exit_engine": "3-Stage Profit-Plus BE (+1.15R->+0.20R) + Chandelier ATR Trailing Stop",
            "specialist_opinions": [op.model_dump() for op in opinions],
            "past_settled_lessons": lessons,
        }
        system_msg = (
            "You are the Head Intraday Gold & FX Trader on a disciplined quantitative prop desk. "
            "Review the regime-classified setup, the 3 specialist opinions, and past lessons. "
            "Output ONLY valid JSON with keys: "
            '{"action": "EXECUTE_LONG"|"EXECUTE_SHORT"|"NO_TRADE", '
            '"grade": "A+"|"A"|"B"|"SKIP", '
            '"confidence": 0.0-1.0, '
            '"thesis_summary": "concise 1-2 sentence rationale"}. '
            "STRICT RULE: Always output NO_TRADE if Adversarial_Trap_Critic issued a VETO or grade is below A."
        )
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 220},
        }
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
            resp = await client.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "{}")
            return json.loads(content)

    def _evaluate_deterministic(
        self,
        setup: CandidateSetup,
        opinions: list[SpecialistOpinion],
        lessons: list[str],
        t0: float,
    ) -> CommitteeDecision:
        """Zero-Fee Deterministic Institutional Rubric (used when Ollama isn't running or during fast backtests)."""
        has_veto = any(op.stance == "VETO" for op in opinions)
        avg_conf = sum(op.confidence for op in opinions[:2]) / 2.0

        if has_veto or setup.quant_score < 74.0 or avg_conf < 0.68:
            action = AgentAction.NO_TRADE
            grade = SetupGrade.B if setup.quant_score >= 68.0 else SetupGrade.SKIP
            veto_notes = [f for op in opinions for f in op.risk_flags]
            reason = veto_notes[0] if veto_notes else "Setup below high-conviction A-grade threshold (74+)."
            summary = f"Committee stood aside (NO_TRADE): {reason}"
        else:
            action = AgentAction.EXECUTE_LONG if setup.side == Side.LONG else AgentAction.EXECUTE_SHORT
            grade = SetupGrade.A_PLUS if (setup.quant_score >= 86.0 and avg_conf >= 0.80) else SetupGrade.A
            summary = (
                f"{grade.value} {setup.symbol} {setup.side.value} [{setup.regime}] approved ({setup.setup_type.value}): "
                f"Score={setup.quant_score:.0f}, H1={setup.htf_bias}, RVOL={setup.rvol:.2f}x, Chandelier Trail armed."
            )

        return CommitteeDecision(
            setup_id=setup.setup_id,
            timestamp=datetime.now(timezone.utc),
            symbol=setup.symbol,
            action=action,
            grade=grade,
            confidence=round(avg_conf, 2),
            llm_mode_used="deterministic_committee_rubric",
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            opinions=opinions,
            thesis_summary=summary,
            lessons_injected=len(lessons),
        )

    async def evaluate_candidate(self, setup: CandidateSetup) -> CommitteeDecision:
        t0 = time.perf_counter()
        opinions = self._build_heuristic_opinions(setup)
        lessons = self.memory.get_lessons_for_symbol(setup.symbol, limit=4)
        has_hard_veto = any(op.stance == "VETO" for op in opinions)

        # If Adversarial Trap Critic issued a hard quantitative VETO, short-circuit immediately in <1ms!
        if has_hard_veto:
            return self._evaluate_deterministic(setup, opinions, lessons, t0)

        ollama_online, installed_models = await self.check_ollama_health()
        if ollama_online and installed_models:
            active_model = self.resolve_ollama_model(installed_models)
            try:
                raw = await self._call_ollama_head_trader(setup, opinions, lessons, model_name=active_model)
                if raw:
                    act_str = str(raw.get("action", "REVIEW")).upper()
                    action = AgentAction(act_str) if act_str in AgentAction.__members__ else AgentAction.REVIEW
                    grade_str = str(raw.get("grade", "SKIP")).upper()
                    grade_map = {"A+": SetupGrade.A_PLUS, "A": SetupGrade.A, "B": SetupGrade.B, "SKIP": SetupGrade.SKIP}
                    grade = grade_map.get(grade_str, SetupGrade.SKIP)
                    conf = float(min(1.0, max(0.0, float(raw.get("confidence", 0.5)))))
                    summary = str(raw.get("thesis_summary", "Ollama Head Trader synthesis completed."))

                    if action == AgentAction.EXECUTE_LONG and setup.side != Side.LONG:
                        action = AgentAction.NO_TRADE
                    if action == AgentAction.EXECUTE_SHORT and setup.side != Side.SHORT:
                        action = AgentAction.NO_TRADE

                    return CommitteeDecision(
                        setup_id=setup.setup_id,
                        timestamp=datetime.now(timezone.utc),
                        symbol=setup.symbol,
                        action=action,
                        grade=grade,
                        confidence=round(conf, 2),
                        llm_mode_used=f"ollama:{active_model}",
                        latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                        opinions=opinions,
                        thesis_summary=summary,
                        lessons_injected=len(lessons),
                    )
            except Exception:
                pass

        return self._evaluate_deterministic(setup, opinions, lessons, t0)

    def evaluate_candidate_sync(self, setup: CandidateSetup) -> CommitteeDecision:
        """Safe synchronous entrypoint that works both from CLI and inside FastAPI's running event loop."""
        if not self.settings.llm_enabled or self.settings.llm_provider == "heuristic":
            t0 = time.perf_counter()
            opinions = self._build_heuristic_opinions(setup)
            lessons = self.memory.get_lessons_for_symbol(setup.symbol, limit=4)
            return self._evaluate_deterministic(setup, opinions, lessons, t0)

        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(self.evaluate_candidate(setup))).result()
        except RuntimeError:
            return asyncio.run(self.evaluate_candidate(setup))
