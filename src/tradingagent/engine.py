"""Main Autonomous Orchestrator Loop uniting Scanner -> Multi-Agent Brain -> Risk Firewall -> MT5."""

from __future__ import annotations

from datetime import datetime, timezone
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


class TradingAgentEngine:
    """
    Coordinates the complete local trading pipeline:
      1. Sub-second management of open positions (Break-Even at +1R)
      2. Gold-first (`XAUUSD`) setup scan across priority tiers
      3. Multi-agent committee debate & settled reflection memory check ONLY when setup fires
      4. Deterministic pre-trade risk firewall & exact lot sizing
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
        self.broker.connect()

    def _build_broker(self) -> BrokerAdapter:
        if self.settings.trading_mode in ("mt5_demo", "mt5_live"):
            return MT5WindowsAdapter(self.settings, self.profiles)
        return PaperMT5Adapter(self.settings, self.profiles)

    async def run_single_cycle(self, auto_execute: bool = True) -> dict[str, Any]:
        if isinstance(self.broker, PaperMT5Adapter):
            self.broker.step_market()

        # 1. Manage existing open positions deterministically
        mgmt_actions = self.risk.manage_open_positions_intraday(self.broker)

        # 2. Fetch account & open positions
        account = self.broker.get_account_snapshot()
        open_positions = self.broker.get_open_positions()

        # 3. Scan XAUUSD first, then fallback priority tiers
        # Use prime NY overlap hour when in paper mode if off-peak so live simulation stays illustrative
        scan_time = (
            datetime.now(timezone.utc).replace(hour=14)
            if self.settings.trading_mode == "paper"
            else datetime.now(timezone.utc)
        )
        candidates = self.scanner.scan_prioritized(self.broker, open_positions, now_utc=scan_time)

        evaluated_results: list[dict[str, Any]] = []
        for cand in candidates:
            decision = await self.committee.evaluate_candidate(cand)
            risk_res: RiskCheckResult | None = None
            receipt: ExecutionReceipt | None = None

            should_exec = decision.action in (AgentAction.EXECUTE_LONG, AgentAction.EXECUTE_SHORT)
            quote = self.broker.get_quote(cand.symbol)
            risk_res = self.risk.validate_and_size(
                setup=cand,
                account=account,
                quote=quote,
                open_positions=self.broker.get_open_positions(),
            )

            if should_exec and risk_res.approved and auto_execute:
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
                if receipt.success:
                    self.risk.record_trade_timestamp(cand.symbol)

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

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trading_mode": self.settings.trading_mode,
            "management_actions": mgmt_actions,
            "candidates_found": len(candidates),
            "evaluations": evaluated_results,
        }

    def trigger_emergency_kill_switch(self) -> dict[str, Any]:
        self.risk.kill_switch_active = True
        receipts = self.broker.close_all_positions(reason="EMERGENCY_KILL_SWITCH")
        return {
            "kill_switch_active": True,
            "closed_positions": [r.model_dump(mode="json") for r in receipts],
        }

    def reset_kill_switch(self) -> dict[str, Any]:
        self.risk.kill_switch_active = False
        return {"kill_switch_active": False}
