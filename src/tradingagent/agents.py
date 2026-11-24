"""Layer 2: Zero-Fee Multi-Agent Trading Committee (Ollama Local LLM + Deterministic Fallback)."""

from __future__ import annotations

import asyncio
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
    short-horizon microstructure focus — tuned specifically for a 16GB RAM Windows 10 laptop:
      - Specialist 1: SMC & Microstructure Analyst (M5/H1 structure, VWAP, FVG, Sweep)
      - Specialist 2: Session, Macro & Cross-Asset Regime Analyst (Killzones, USD beta, ADX/RSI)
      - Specialist 3: Adversarial Trap & Spread Skeptic ("Why could this Gold setup fail?")
      - Head Trader Synthesizer: Weighs all 3 specialists + Settled Memory Lessons -> Typed Verdict

    100% Zero-Fee Execution:
      - Queries local Ollama (`http://127.0.0.1:11434/api/chat` with `format: "json"`) when running
      - Gracefully falls back to a deterministic institutional multi-agent scoring rubric if Ollama
        is offline or disabled — ensuring zero crashes and <10ms backtest compatibility!
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
                    models = [m.get("name", "") for m in resp.json().get("models", [])]
                    return True, models
        except Exception:
            pass
        return False, []

    def _build_heuristic_opinions(self, setup: CandidateSetup) -> list[SpecialistOpinion]:
        """Deterministic baseline specialist analysis (always runs first so numbers are never hallucinated)."""
        expected_stance = "BULLISH" if setup.side == Side.LONG else "BEARISH"

        # 1. Microstructure & SMC Specialist
        smc_args = list(setup.notes)
        smc_flags: list[str] = []
        smc_conf = min(0.95, setup.quant_score / 100.0)
        if abs(setup.vwap_distance_atr) > 2.6:
            smc_flags.append(f"Price extended {setup.vwap_distance_atr:+.2f} ATR from session VWAP")
            smc_conf -= 0.12
        else:
            smc_args.append(f"Healthy VWAP distance ({setup.vwap_distance_atr:+.2f} ATR)")

        op_smc = SpecialistOpinion(
            agent_name="SMC_Microstructure_Analyst",
            stance=expected_stance,
            confidence=round(max(0.3, smc_conf), 2),
            key_arguments=smc_args,
            risk_flags=smc_flags,
        )

        # 2. Macro & Session Regime Specialist
        macro_args: list[str] = []
        macro_flags: list[str] = []
        macro_stance = expected_stance
        macro_conf = 0.76

        if setup.session_label in ("LONDON_SESSION", "NY_OVERLAP_PRIME"):
            macro_args.append(f"Active institutional liquidity window: {setup.session_label}")
            macro_conf += 0.10
        else:
            macro_flags.append(f"Off-peak liquidity session ({setup.session_label})")
            macro_conf -= 0.14

        if (setup.side == Side.LONG and setup.htf_bias == "BULLISH") or (
            setup.side == Side.SHORT and setup.htf_bias == "BEARISH"
        ):
            macro_args.append(f"H1 macro trend aligned ({setup.htf_bias})")
            macro_conf += 0.08
        elif setup.htf_bias == "NEUTRAL":
            macro_args.append("H1 trend transitional/neutral — rely on tight M5 invalidation")
        else:
            macro_flags.append(f"Counter-trend scalp against H1 {setup.htf_bias} bias")
            macro_conf -= 0.18
            macro_stance = "NEUTRAL"

        op_macro = SpecialistOpinion(
            agent_name="Session_Macro_Analyst",
            stance=macro_stance,
            confidence=round(min(0.95, max(0.25, macro_conf)), 2),
            key_arguments=macro_args,
            risk_flags=macro_flags,
        )

        # 3. Adversarial Trap & Execution Critic
        trap_args: list[str] = []
        trap_flags: list[str] = []
        trap_stance = "NEUTRAL"
        trap_conf = 0.65

        if setup.spread_to_sl_ratio <= 0.08:
            trap_args.append(f"Tight spread friction ({setup.spread_to_sl_ratio * 100:.1f}% of SL)")
        else:
            trap_flags.append(f"Moderate spread drag ({setup.spread_to_sl_ratio * 100:.1f}% of SL)")

        if (setup.side == Side.LONG and setup.rsi > 73) or (setup.side == Side.SHORT and setup.rsi < 27):
            trap_flags.append(f"Momentum exhaustion risk (M5 RSI={setup.rsi:.1f}) — inducement trap warning")
            trap_stance = "VETO"
            trap_conf = 0.85
        elif setup.adx < 15 and setup.setup_type.value == "SESSION_ORB_BREAKOUT":
            trap_flags.append(f"Weak ADX ({setup.adx:.1f}) increases fakeout probability on breakout")
            trap_stance = "VETO"
            trap_conf = 0.80
        else:
            trap_args.append(f"No exhaustion trap detected (RSI={setup.rsi:.1f}, ADX={setup.adx:.1f})")

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
    ) -> dict | None:
        prompt = {
            "symbol": setup.symbol,
            "priority_tier": setup.tier,
            "proposed_side": setup.side.value,
            "setup_type": setup.setup_type.value,
            "quant_score": setup.quant_score,
            "htf_bias": setup.htf_bias,
            "session": setup.session_label,
            "rr_ratio": setup.rr_ratio,
            "spread_to_sl_ratio": setup.spread_to_sl_ratio,
            "specialist_opinions": [op.model_dump() for op in opinions],
            "past_settled_lessons": lessons,
        }
        system_msg = (
            "You are the Head Intraday Gold & FX Trader on a disciplined prop desk. "
            "Review the computed quantitative setup, the 3 specialist opinions, and past lessons. "
            "Output ONLY valid JSON with keys: "
            '{"action": "EXECUTE_LONG"|"EXECUTE_SHORT"|"NO_TRADE", '
            '"grade": "A+"|"A"|"B"|"SKIP", '
            '"confidence": 0.0-1.0, '
            '"thesis_summary": "concise 1-2 sentence rationale"}. '
            "Remember: NO_TRADE is preferred if Adversarial_Trap_Critic issued a VETO or grade is below A."
        )
        payload = {
            "model": self.settings.ollama_deep_model,
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

    async def evaluate_candidate(self, setup: CandidateSetup) -> CommitteeDecision:
        t0 = time.perf_counter()
        opinions = self._build_heuristic_opinions(setup)
        lessons = self.memory.get_lessons_for_symbol(setup.symbol, limit=4)

        ollama_online, installed_models = await self.check_ollama_health()
        if ollama_online and installed_models:
            try:
                raw = await self._call_ollama_head_trader(setup, opinions, lessons)
                if raw:
                    act_str = str(raw.get("action", "REVIEW")).upper()
                    action = AgentAction(act_str) if act_str in AgentAction.__members__ else AgentAction.REVIEW
                    grade_str = str(raw.get("grade", "SKIP")).upper()
                    grade_map = {"A+": SetupGrade.A_PLUS, "A": SetupGrade.A, "B": SetupGrade.B, "SKIP": SetupGrade.SKIP}
                    grade = grade_map.get(grade_str, SetupGrade.SKIP)
                    conf = float(min(1.0, max(0.0, float(raw.get("confidence", 0.5)))))
                    summary = str(raw.get("thesis_summary", "Ollama Head Trader synthesis completed."))

                    # Enforce consistency: never allow LONG setup to flip to SHORT without scanner trigger
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
                        llm_mode_used=f"ollama:{self.settings.ollama_deep_model}",
                        latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                        opinions=opinions,
                        thesis_summary=summary,
                        lessons_injected=len(lessons),
                    )
            except Exception:
                pass

        # Zero-Fee Deterministic Institutional Rubric (used when Ollama isn't started yet or during fast backtests)
        has_veto = any(op.stance == "VETO" for op in opinions)
        avg_conf = sum(op.confidence for op in opinions[:2]) / 2.0

        if has_veto or setup.quant_score < 68.0 or avg_conf < 0.62:
            action = AgentAction.NO_TRADE
            grade = SetupGrade.B if setup.quant_score >= 64.0 else SetupGrade.SKIP
            veto_notes = [f for op in opinions for f in op.risk_flags]
            reason = veto_notes[0] if veto_notes else "Setup below A-grade institutional threshold."
            summary = f"Committee stood aside (NO_TRADE): {reason}"
        else:
            action = AgentAction.EXECUTE_LONG if setup.side == Side.LONG else AgentAction.EXECUTE_SHORT
            grade = SetupGrade.A_PLUS if (setup.quant_score >= 84.0 and avg_conf >= 0.78) else SetupGrade.A
            summary = (
                f"{grade.value} {setup.symbol} {setup.side.value} approved ({setup.setup_type.value}): "
                f"QuantScore={setup.quant_score:.0f}/100, H1={setup.htf_bias}, Net R:R={setup.rr_ratio:.2f}."
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

    def evaluate_candidate_sync(self, setup: CandidateSetup) -> CommitteeDecision:
        return asyncio.run(self.evaluate_candidate(setup))
