"""Layer 1: Regime-Adaptive Gold-First Quantitative Strategy Engine & Priority Router."""

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


def classify_session_label(hour_utc: int) -> str:
    if 7 <= hour_utc < 12:
        return "LONDON_SESSION"
    if 12 <= hour_utc < 17:
        return "NY_OVERLAP_PRIME"
    if 17 <= hour_utc < 21:
        return "NY_AFTERNOON"
    return "ASIAN_OFF_PEAK"


def evaluate_regime_signal_at_index(
    profile: SymbolProfile,
    feat: pd.DataFrame,
    idx: int,
    htf_bias: str,
    live_spread_points: float,
    eval_ts: datetime,
    entry_price_override: float | None = None,
    setup_id_prefix: str | None = None,
) -> CandidateSetup | None:
    """
    Single Source of Truth Regime-Adaptive Signal Engine shared 1:1 between Live MT5 Scanning
    and Historical Backtesting (zero drift between backtest and live execution):
      1. Hard Killzone & Volatility Chop Gate (stands aside in Asian drift & low-ATR sawtooth chop)
      2. Playbook A — Judas Swing Liquidity Sweep + Displacement FVG Reclaim (`LIQUIDITY_SWEEP_FVG`)
      3. Playbook B — Asian Session Range / Bollinger-Keltner Squeeze + RVOL Breakout (`SESSION_ORB_BREAKOUT`)
      4. Playbook C — Regime-Confirmed ATR-Normalized EMA20/VWAP Continuation (`VWAP_EMA_PULLBACK`)
      5. Structural 6-bar Swing-Wick Stop-Loss + 3.20R Ceiling TP (managed by 3-Stage Chandelier Trail)
    """
    if idx < 25 or idx >= len(feat):
        return None

    # 1. Hard Live Spread & Prime Killzone Gate
    if live_spread_points > profile.max_spread_points:
        return None
    if not profile.is_in_prime_session(int(eval_ts.hour)):
        return None

    row = feat.iloc[idx]
    prev = feat.iloc[idx - 1]

    open_p = float(row["open"])
    close = float(row["close"])
    atr = max(float(row["atr"]), profile.point * 20.0)
    atr_ratio = float(row["atr_ratio"])
    adx = float(row["adx"])
    rsi = float(row["rsi"])
    ema20 = float(row["ema_20"])
    ema50 = float(row["ema_50"])
    vwap = float(row["vwap"])
    rvol = float(row["rvol"])
    er = float(row["efficiency_ratio"])
    body_ratio = float(row["body_ratio"])
    body_atr = float(row["body_atr"])
    recent_squeeze = bool(row["recent_squeeze"])
    swing_low_6 = float(row["swing_low_6"])
    swing_high_6 = float(row["swing_high_6"])
    vwap_dist_atr = round((close - vwap) / atr, 3)

    # Dead low-volatility random-walk chop gate
    if atr_ratio < 0.72 and adx < 17.5 and not recent_squeeze:
        return None

    # Quantitative Regime Classification
    if adx >= 22.5 and er >= 0.27 and atr_ratio >= 0.85:
        regime = "TREND_EXPANSION"
    elif recent_squeeze and atr_ratio >= 0.88:
        regime = "SQUEEZE_BREAKOUT"
    else:
        regime = "LIQUIDITY_ROTATION"

    side: Side | None = None
    setup_type: SetupType | None = None
    notes: list[str] = []
    score = 60.0

    # -------------------------------------------------------------------------
    # PLAYBOOK 1: Judas Liquidity Sweep + Displacement FVG Reclaim (Sniper Reversal/Continuation)
    # -------------------------------------------------------------------------
    fresh_sw_low = bool(row["fresh_sweep_low"])
    fresh_sw_high = bool(row["fresh_sweep_high"])
    rec_bull_fvg = bool(row["recent_bull_fvg"])
    rec_bear_fvg = bool(row["recent_bear_fvg"])

    if (
        fresh_sw_low
        and rec_bull_fvg
        and close > open_p
        and close > ema20
        and (close - swing_low_6) <= 2.30 * atr
        and 26.0 <= rsi <= 72.0
    ):
        side = Side.LONG
        setup_type = SetupType.LIQUIDITY_SWEEP_FVG
        score = 80.0 if htf_bias != "BEARISH" else 73.0
        notes.append(f"[{regime}] Fresh 20-bar sell-side sweep + bullish FVG reclaim (RVOL {rvol:.2f}x)")
    elif (
        fresh_sw_high
        and rec_bear_fvg
        and close < open_p
        and close < ema20
        and (swing_high_6 - close) <= 2.30 * atr
        and 28.0 <= rsi <= 74.0
    ):
        side = Side.SHORT
        setup_type = SetupType.LIQUIDITY_SWEEP_FVG
        score = 80.0 if htf_bias != "BULLISH" else 73.0
        notes.append(f"[{regime}] Fresh 20-bar buy-side sweep + bearish FVG breakdown (RVOL {rvol:.2f}x)")

    # -------------------------------------------------------------------------
    # PLAYBOOK 2: Asian Session Range / Bollinger-Keltner Squeeze + RVOL Breakout
    # -------------------------------------------------------------------------
    if side is None and 7 <= int(eval_ts.hour) <= 17:
        asian_hi = row["asian_high"]
        asian_lo = row["asian_low"]
        prior_14_hi = float(feat["high"].iloc[max(0, idx - 14):idx].max())
        prior_14_lo = float(feat["low"].iloc[max(0, idx - 14):idx].min())

        bull_asian_brk = (
            pd.notna(asian_hi)
            and float(prev["close"]) <= float(asian_hi)
            and close > float(asian_hi)
            and (close - float(asian_hi)) <= 0.85 * atr
        )
        bear_asian_brk = (
            pd.notna(asian_lo)
            and float(prev["close"]) >= float(asian_lo)
            and close < float(asian_lo)
            and (float(asian_lo) - close) <= 0.85 * atr
        )
        bull_sqz_brk = recent_squeeze and close > prior_14_hi and adx >= 20.0
        bear_sqz_brk = recent_squeeze and close < prior_14_lo and adx >= 20.0

        if (
            (bull_asian_brk or bull_sqz_brk)
            and htf_bias != "BEARISH"
            and close > vwap
            and ema20 >= ema50
            and rvol >= 1.00
            and body_ratio >= 0.48
            and body_atr >= 0.36
            and rsi <= 72.0
        ):
            side = Side.LONG
            setup_type = SetupType.SESSION_ORB_BREAKOUT
            score = 81.0 if (htf_bias == "BULLISH" and rvol >= 1.20) else 72.0
            brk_lbl = "Asian Session High" if bull_asian_brk else "BB-Keltner Volatility Squeeze"
            notes.append(f"[{regime}] {brk_lbl} breakout with RVOL {rvol:.2f}x & H1 {htf_bias}")
        elif (
            (bear_asian_brk or bear_sqz_brk)
            and htf_bias != "BULLISH"
            and close < vwap
            and ema20 <= ema50
            and rvol >= 1.00
            and body_ratio >= 0.48
            and body_atr >= 0.36
            and rsi >= 28.0
        ):
            side = Side.SHORT
            setup_type = SetupType.SESSION_ORB_BREAKOUT
            score = 81.0 if (htf_bias == "BEARISH" and rvol >= 1.20) else 72.0
            brk_lbl = "Asian Session Low" if bear_asian_brk else "BB-Keltner Volatility Squeeze"
            notes.append(f"[{regime}] {brk_lbl} breakdown with RVOL {rvol:.2f}x & H1 {htf_bias}")

    # -------------------------------------------------------------------------
    # PLAYBOOK 3: Regime-Confirmed ATR-Normalized VWAP / EMA-20 Trend Pullback
    # -------------------------------------------------------------------------
    if side is None and regime == "TREND_EXPANSION":
        prev_low = float(prev["low"])
        prev_high = float(prev["high"])
        if (
            htf_bias == "BULLISH"
            and ema20 > ema50
            and close > vwap
            and prev_low <= (ema20 + 0.14 * atr)
            and prev_low >= (ema50 - 0.25 * atr)
            and close > open_p
            and close > prev_high
            and body_atr >= 0.28
            and body_ratio >= 0.45
            and 42.0 <= rsi <= 71.0
        ):
            side = Side.LONG
            setup_type = SetupType.VWAP_EMA_PULLBACK
            score = 77.0
            notes.append(
                f"[TREND_EXPANSION] ATR-normalized EMA20 pullback engulfing (ER={er:.2f}, ADX={adx:.1f})"
            )
        elif (
            htf_bias == "BEARISH"
            and ema20 < ema50
            and close < vwap
            and prev_high >= (ema20 - 0.14 * atr)
            and prev_high <= (ema50 + 0.25 * atr)
            and close < open_p
            and close < prev_low
            and body_atr >= 0.28
            and body_ratio >= 0.45
            and 29.0 <= rsi <= 58.0
        ):
            side = Side.SHORT
            setup_type = SetupType.VWAP_EMA_PULLBACK
            score = 77.0
            notes.append(
                f"[TREND_EXPANSION] ATR-normalized EMA20 pullback breakdown (ER={er:.2f}, ADX={adx:.1f})"
            )

    if side is None or setup_type is None:
        return None

    # Quality Confluence Scoring (no blind 24/7 symbol inflation)
    if (side == Side.LONG and htf_bias == "BULLISH") or (side == Side.SHORT and htf_bias == "BEARISH"):
        score += 6.0
        notes.append(f"Aligned with H1 {htf_bias} macro trend")
    if rvol >= 1.35:
        score += 4.0
        notes.append(f"Elevated institutional volume ({rvol:.2f}x 20-bar avg)")
    if er >= 0.36 and adx >= 25.0:
        score += 3.0
    if 12 <= int(eval_ts.hour) < 17:
        score += 3.0

    # Structural 6-bar Swing-Wick Stop-Loss + 3.20R Ceiling TP (paired with 3-Stage Chandelier Trailing)
    entry_ref = float(entry_price_override) if entry_price_override is not None else close
    if side == Side.LONG:
        raw_wick_dist = (entry_ref - swing_low_6) + 0.25 * atr
    else:
        raw_wick_dist = (swing_high_6 - entry_ref) + 0.25 * atr

    sl_dist = min(2.05 * atr, max(1.30 * atr, raw_wick_dist))
    tp_dist = round(3.20 * sl_dist, 6)

    spread_price = live_spread_points * profile.point
    spread_to_sl = spread_price / sl_dist if sl_dist > 0 else 1.0
    if spread_to_sl > profile.max_spread_to_sl_ratio:
        return None

    if side == Side.LONG:
        sl_price = round(entry_ref - sl_dist, 5)
        tp_price = round(entry_ref + tp_dist, 5)
    else:
        sl_price = round(entry_ref + sl_dist, 5)
        tp_price = round(entry_ref - tp_dist, 5)

    rr = round(tp_dist / sl_dist, 2)
    sid = f"{setup_id_prefix}" if setup_id_prefix else f"{profile.canonical}-{uuid.uuid4().hex[:8]}"

    return CandidateSetup(
        setup_id=sid,
        timestamp=eval_ts,
        symbol=profile.canonical,
        tier=profile.tier,
        side=side,
        setup_type=setup_type,
        timeframe="M5",
        htf_bias=htf_bias,
        entry_price=entry_ref,
        stop_loss=sl_price,
        take_profit=tp_price,
        atr=round(atr, 5),
        adx=round(adx, 2),
        rsi=round(rsi, 2),
        vwap_distance_atr=vwap_dist_atr,
        spread_points=round(live_spread_points, 1),
        spread_to_sl_ratio=round(spread_to_sl, 4),
        rr_ratio=rr,
        quant_score=min(99.0, round(score, 1)),
        session_label=classify_session_label(int(eval_ts.hour)),
        regime=regime,
        rvol=round(rvol, 2),
        efficiency_ratio=round(er, 3),
        notes=notes,
    )


class GoldFirstScanner:
    """
    Continuous deterministic setup detector (<5ms per symbol, ZERO LLM calls).
    Priority Hierarchy:
      1. XAUUSD (Tier 1 Primary): Always scanned first, with dedicated 3rd VIP position slot.
      2. NAS100 & BTCUSD (Tier 2): Scanned when Gold has no active setup or Gold slot is full.
      3. EURUSD, GBPUSD, USDCAD (Tier 3): Clean FX intraday fallback setups.
    """

    def __init__(self, settings: AppSettings, profiles: dict[str, SymbolProfile]) -> None:
        self.settings = settings
        self.profiles = profiles

    def ordered_symbols(self) -> list[SymbolProfile]:
        return sorted(self.profiles.values(), key=lambda p: (p.tier, p.canonical != "XAUUSD"))

    def classify_session(self, hour_utc: int) -> str:
        return classify_session_label(hour_utc)

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

        # Higher-timeframe (H1) directional bias using confirmed closed H1 bar (`iloc[-2]` when available)
        h1_idx = -2 if len(h1_bars) >= 35 else -1
        h1_ema20 = float(compute_ema(h1_bars["close"], 20).iloc[h1_idx])
        h1_ema50 = float(compute_ema(h1_bars["close"], 50).iloc[h1_idx])
        h1_close = float(h1_bars["close"].iloc[h1_idx])
        if h1_close > h1_ema20 >= h1_ema50:
            htf_bias = "BULLISH"
        elif h1_close < h1_ema20 <= h1_ema50:
            htf_bias = "BEARISH"
        else:
            htf_bias = "NEUTRAL"

        feat = detect_fvg_and_sweeps(m5_bars)
        # Evaluate on the most recently CLOSED M5 bar (`len - 2`) to eliminate mid-candle intrabar wick fakeouts
        closed_idx = len(feat) - 2
        return evaluate_regime_signal_at_index(
            profile=profile,
            feat=feat,
            idx=closed_idx,
            htf_bias=htf_bias,
            live_spread_points=live_spread_points,
            eval_ts=now,
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
