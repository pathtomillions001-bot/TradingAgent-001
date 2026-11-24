"""Typed domain schemas shared across Scanner, Agents, Risk Firewall, and MT5 Broker."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SetupType(str, Enum):
    LIQUIDITY_SWEEP_FVG = "LIQUIDITY_SWEEP_FVG"
    SESSION_ORB_BREAKOUT = "SESSION_ORB_BREAKOUT"
    VWAP_EMA_PULLBACK = "VWAP_EMA_PULLBACK"


class AgentAction(str, Enum):
    EXECUTE_LONG = "EXECUTE_LONG"
    EXECUTE_SHORT = "EXECUTE_SHORT"
    NO_TRADE = "NO_TRADE"
    REVIEW = "REVIEW"


class SetupGrade(str, Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"
    SKIP = "SKIP"


class QuoteTick(BaseModel):
    symbol: str
    broker_symbol: str
    timestamp: datetime
    bid: float
    ask: float
    point: float
    tick_value: float = Field(description="Value in account currency of 1 point move for 1.0 lot")
    tick_size: float = Field(description="Minimum price change step (usually == point)")
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    stops_level_points: int = 0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_price(self) -> float:
        return max(0.0, self.ask - self.bid)

    @property
    def spread_points(self) -> float:
        return self.spread_price / self.point if self.point > 0 else 0.0


class AccountSnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    login: int = 100001
    server: str = "Local-Paper-MT5"
    currency: str = "USD"
    balance: float = 10000.0
    equity: float = 10000.0
    margin: float = 0.0
    free_margin: float = 10000.0
    leverage: int = 100
    day_starting_equity: float = 10000.0
    peak_equity: float = 10000.0
    realized_pnl_today: float = 0.0

    @property
    def daily_drawdown_pct(self) -> float:
        if self.day_starting_equity <= 0:
            return 0.0
        dd = (self.day_starting_equity - min(self.equity, self.day_starting_equity)) / self.day_starting_equity
        return round(dd * 100.0, 4)

    @property
    def total_drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        dd = (self.peak_equity - min(self.equity, self.peak_equity)) / self.peak_equity
        return round(dd * 100.0, 4)


class OpenPosition(BaseModel):
    ticket: int
    symbol: str
    broker_symbol: str
    side: Side
    volume: float
    entry_price: float
    current_price: float
    sl: float
    tp: float
    initial_risk_usd: float
    unrealized_pnl: float
    opened_at: datetime
    setup_type: str = "MANUAL_OR_AGENT"
    be_moved: bool = False
    partial_taken: bool = False
    magic: int = 2026001


class CandidateSetup(BaseModel):
    setup_id: str
    timestamp: datetime
    symbol: str
    tier: int
    side: Side
    setup_type: SetupType
    timeframe: str = "M5"
    htf_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    adx: float
    rsi: float
    vwap_distance_atr: float
    spread_points: float
    spread_to_sl_ratio: float
    rr_ratio: float
    quant_score: float = Field(ge=0.0, le=100.0)
    session_label: str
    notes: list[str] = Field(default_factory=list)


class SpecialistOpinion(BaseModel):
    agent_name: str
    stance: Literal["BULLISH", "BEARISH", "NEUTRAL", "VETO"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_arguments: list[str]
    risk_flags: list[str] = Field(default_factory=list)


class CommitteeDecision(BaseModel):
    setup_id: str
    timestamp: datetime
    symbol: str
    action: AgentAction
    grade: SetupGrade
    confidence: float = Field(ge=0.0, le=1.0)
    llm_mode_used: str
    latency_ms: float
    opinions: list[SpecialistOpinion]
    thesis_summary: str
    lessons_injected: int = 0


class RiskCheckResult(BaseModel):
    approved: bool
    symbol: str
    side: Side
    calculated_lots: float = 0.0
    risk_usd: float = 0.0
    risk_pct: float = 0.0
    net_rr_after_spread: float = 0.0
    projected_net_usd_risk_pct: float = 0.0
    adjusted_sl: float = 0.0
    adjusted_tp: float = 0.0
    rejection_reasons: list[str] = Field(default_factory=list)
    passed_checks: list[str] = Field(default_factory=list)


class ExecutionReceipt(BaseModel):
    success: bool
    ticket: int | None = None
    symbol: str
    side: Side
    volume: float
    fill_price: float
    sl: float
    tp: float
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
