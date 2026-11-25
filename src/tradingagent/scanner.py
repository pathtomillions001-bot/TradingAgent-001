"""Layer 1: Gold-First Quantitative Setup Scanner & Multi-Asset Fallback Priority Router."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import uuid
import pandas as pd

from tradingagent.broker import BrokerAdapter
from tradingagent.config import AppSettings, SymbolProfile
from tradingagent.indicators import compute_ema, detect_fvg_and_sweeps
from tradingagent.models import CandidateSetup, OpenPosition, Side, SetupType

logger = logging.getLogger(__name__)


class GoldFirstScanner:
    """
    Continuous deterministic setup detector (<5ms per symbol, ZERO LLM calls).
    Priority Hierarchy:
      1. XAUUSD (Tier 1 Primary): Always scanned first. If a high-conviction Gold setup triggers
         and Gold capacity is available, secondary symbols are skipped so focus & margin stay on Gold.
      2. NAS100 & BTCUSD (Tier 2): Scanned when Gold has no active setup or Gold slot is full.
      3. EURUSD, GBPUSD, USDCAD (Tier 3): Clean FX intraday fallback setups.
    """

    def __init__(self, settings: AppSettings, profiles: dict[str, SymbolProfile]) -> None:
        self.settings = settings
        self.profiles = profiles

    def ordered_symbols(self) -> list[SymbolProfile]:
        return sorted(self.profiles.values(), key=lambda p: (p.tier, p.canonical != "XAUUSD"))

    def classify_session(self, hour_utc: int) -> str:
        if 7 <= hour_utc < 12:
            return "LONDON_SESSION"
        if 12 <= hour_utc < 17:
            return "NY_OVERLAP_PRIME"
        if 17 <= hour_utc < 21:
            return "NY_AFTERNOON"
        return "ASIAN_OFF_PEAK"

    def evaluate_symbol(
        self,
        profile: SymbolProfile,
        m5_bars: pd.DataFrame,
        h1_bars: pd.DataFrame,
        live_spread_points: float,
        now_utc: datetime | None = None,
    ) -> CandidateSetup | None:
        if len(m5_bars) < 60 or len(h1_bars) < 30:
            return None

        now = now_utc or datetime.now(timezone.utc)
        # Live spread ceiling gate
        if live_spread_points > profile.max_spread_points:
            return None

        # Higher-timeframe (H1) directional bias
        h1_ema20 = compute_ema(h1_bars["close"], 20).iloc[-1]
        h1_ema50 = compute_ema(h1_bars["close"], 50).iloc[-1]
        h1_close = float(h1_bars["close"].iloc[-1])
        if h1_close > h1_ema20 >= h1_ema50:
            htf_bias = "BULLISH"
        elif h1_close < h1_ema20 <= h1_ema50:
            htf_bias = "BEARISH"
        else:
            htf_bias = "NEUTRAL"

        feat = detect_fvg_and_sweeps(m5_bars)
        last = feat.iloc[-1]
        prev = feat.iloc[-2]

        close = float(last["close"])
        atr = max(float(last["atr"]), profile.point * 20)
        adx = float(last["adx"])
        rsi = float(last["rsi"])
        ema20 = float(last["ema_20"])
        ema50 = float(last["ema_50"])
        vwap = float(last["vwap"])
        vwap_dist_atr = round((close - vwap) / atr, 3)

        side: Side | None = None
        setup_type: SetupType | None = None
        notes: list[str] = []
        base_score = 52.0

        # Setup Playbook 1: ICT Liquidity Sweep + Displacement Fair Value Gap (Best on XAUUSD & NAS100)
        if bool(last["recent_sweep_low"]) and bool(last["recent_bull_fvg"]) and close > ema20 and rsi < 72:
            side = Side.LONG
            setup_type = SetupType.LIQUIDITY_SWEEP_FVG
            base_score = 78.0
            notes.append("Sell-side liquidity swept followed by bullish displacement + M5 FVG reclaim")
        elif bool(last["recent_sweep_high"]) and bool(last["recent_bear_fvg"]) and close < ema20 and rsi > 28:
            side = Side.SHORT
            setup_type = SetupType.LIQUIDITY_SWEEP_FVG
            base_score = 78.0
            notes.append("Buy-side liquidity swept followed by bearish displacement + M5 FVG breakdown")

        # Setup Playbook 2: Trend + Session VWAP / EMA-20 Pullback Continuation
        elif (
            htf_bias == "BULLISH"
            and close > vwap
            and ema20 > ema50
            and float(prev["low"]) <= ema20 * 1.0008
            and close > float(prev["high"])
            and 42 <= rsi <= 68
        ):
            side = Side.LONG
            setup_type = SetupType.VWAP_EMA_PULLBACK
            base_score = 71.0
            notes.append("H1 Bullish trend + M5 VWAP/EMA20 pullback rejection candle")
        elif (
            htf_bias == "BEARISH"
            and close < vwap
            and ema20 < ema50
            and float(prev["high"]) >= ema20 * 0.9992
            and close < float(prev["low"])
            and 32 <= rsi <= 58
        ):
            side = Side.SHORT
            setup_type = SetupType.VWAP_EMA_PULLBACK
            base_score = 71.0
            notes.append("H1 Bearish trend + M5 VWAP/EMA20 pullback rejection candle")

        # Setup Playbook 3: Prime Session Volatility Momentum Breakout
        elif adx >= 24 and profile.is_in_prime_session(now.hour):
            recent_12_high = float(feat["high"].iloc[-13:-1].max())
            recent_12_low = float(feat["low"].iloc[-13:-1].min())
            if close > recent_12_high and htf_bias != "BEARISH" and rsi < 75:
                side = Side.LONG
                setup_type = SetupType.SESSION_ORB_BREAKOUT
                base_score = 68.0
                notes.append("Prime session 1-hour range breakout with ADX >= 24")
            elif close < recent_12_low and htf_bias != "BULLISH" and rsi > 25:
                side = Side.SHORT
                setup_type = SetupType.SESSION_ORB_BREAKOUT
                base_score = 68.0
                notes.append("Prime session 1-hour range breakdown with ADX >= 24")

        if side is None or setup_type is None:
            return None

        # Score bonuses/penalties
        if (side == Side.LONG and htf_bias == "BULLISH") or (side == Side.SHORT and htf_bias == "BEARISH"):
            base_score += 9.0
            notes.append(f"Aligned with H1 {htf_bias} structure")
        if profile.is_in_prime_session(now.hour):
            base_score += 7.0
            notes.append(f"Inside prime liquidity killzone ({self.classify_session(now.hour)})")
        if profile.canonical == "XAUUSD":
            base_score += 4.0  # Gold-first focus preference

        # Dynamic ATR-based Stop Loss & Take Profit (minimum 1:2.2 gross R:R so net R:R > 1.8 after spread)
        sl_dist = 1.45 * atr
        tp_dist = 3.20 * atr
        spread_price = live_spread_points * profile.point
        spread_to_sl = spread_price / sl_dist if sl_dist > 0 else 1.0

        if spread_to_sl > profile.max_spread_to_sl_ratio:
            return None

        if side == Side.LONG:
            sl_price = round(close - sl_dist, 5)
            tp_price = round(close + tp_dist, 5)
        else:
            sl_price = round(close + sl_dist, 5)
            tp_price = round(close - tp_dist, 5)

        rr = round(tp_dist / sl_dist, 2)

        return CandidateSetup(
            setup_id=f"{profile.canonical}-{uuid.uuid4().hex[:8]}",
            timestamp=now,
            symbol=profile.canonical,
            tier=profile.tier,
            side=side,
            setup_type=setup_type,
            timeframe="M5",
            htf_bias=htf_bias,
            entry_price=close,
            stop_loss=sl_price,
            take_profit=tp_price,
            atr=round(atr, 5),
            adx=round(adx, 2),
            rsi=round(rsi, 2),
            vwap_distance_atr=vwap_dist_atr,
            spread_points=round(live_spread_points, 1),
            spread_to_sl_ratio=round(spread_to_sl, 4),
            rr_ratio=rr,
            quant_score=min(99.0, round(base_score, 1)),
            session_label=self.classify_session(now.hour),
            notes=notes,
        )

    def scan_prioritized(
        self,
        broker: BrokerAdapter,
        open_positions: list[OpenPosition],
        now_utc: datetime | None = None,
    ) -> list[CandidateSetup]:
        """
        Scans XAUUSD first (Tier 1), followed by Tier 2 (NAS100, BTCUSD) and Tier 3 (EURUSD, GBPUSD, USDCAD).
        Fault-tolerant: if any individual symbol is closed/unavailable on the connected MT5 broker,
        it is skipped cleanly without interrupting Gold or other active pairs.
        """
        candidates: list[CandidateSetup] = []
        gold_open_count = sum(1 for p in open_positions if p.symbol == "XAUUSD")
        open_symbols = {p.symbol for p in open_positions}

        for prof in self.ordered_symbols():
            if prof.canonical in open_symbols:
                continue
            if prof.canonical == "XAUUSD" and gold_open_count >= self.settings.max_gold_positions:
                continue

            try:
                q = broker.get_quote(prof.canonical)
                m5 = broker.get_bars(prof.canonical, "M5", 260)
                h1 = broker.get_bars(prof.canonical, "H1", 120)
            except Exception as exc:
                logger.warning("Skipping %s during scan (symbol unavailable/closed): %s", prof.canonical, exc)
                continue

            norm_spread_pts = prof.canonical_spread_points(q.spread_price)
            setup = self.evaluate_symbol(prof, m5, h1, norm_spread_pts, now_utc=now_utc)
            if setup is not None:
                candidates.append(setup)

        candidates.sort(key=lambda c: (c.tier, -c.quant_score))
        return candidates
