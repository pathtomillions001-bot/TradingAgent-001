"""Spread, Slippage & Session-Aware Backtesting + Agent Ablation Engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from pydantic import BaseModel

from tradingagent.config import AppSettings, SymbolProfile
from tradingagent.indicators import compute_ema, detect_fvg_and_sweeps
from tradingagent.memory import StateAndMemoryStore
from tradingagent.models import AgentAction, CandidateSetup, QuoteTick, Side, SetupType
from tradingagent.risk import DeterministicRiskFirewall
from tradingagent.agents import MultiAgentCommittee


class BacktestSummary(BaseModel):
    symbol: str
    bars_tested: int
    mode: str
    starting_equity: float
    ending_equity: float
    net_return_pct: float
    win_rate_pct: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    wins: int
    losses: int
    avg_r_multiple: float
    spread_cost_usd: float
    equity_curve: list[dict]
    sample_trades: list[dict]


def generate_realistic_intraday_dataset(
    symbol: str = "XAUUSD",
    n_bars: int = 2500,
    seed: int = 101,
) -> pd.DataFrame:
    """
    Generates realistic multi-session M5 bars with London/NY volatility expansion, Asian consolidation,
    liquidity sweeps, trend legs, and variable session bid/ask spreads for offline laptop verification
    before or alongside MT5 CSV/Parquet export.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(n_bars)]

    base_map = {
        "XAUUSD": (2620.0, 1.55, 17.0),
        "NAS100": (20400.0, 17.0, 130.0),
        "BTCUSD": (91000.0, 135.0, 1700.0),
        "EURUSD": (1.07200, 0.00048, 9.0),
        "GBPUSD": (1.28500, 0.00060, 11.0),
        "USDCAD": (1.38400, 0.00050, 12.0),
    }
    base_price, base_vol, base_spread = base_map.get(symbol, (100.0, 0.5, 10.0))

    closes = np.zeros(n_bars)
    opens = np.zeros(n_bars)
    highs = np.zeros(n_bars)
    lows = np.zeros(n_bars)
    spreads = np.zeros(n_bars)
    volumes = np.zeros(n_bars, dtype=int)

    price = base_price
    regime_drift = 0.0

    for i, ts in enumerate(timestamps):
        hr = ts.hour
        in_killzone = (7 <= hr < 11) or (12 <= hr < 17)
        sess_mult = 1.35 if in_killzone else 0.55

        if i % 72 == 0:
            regime_drift = float(rng.choice([-1.0, -0.45, 0.0, 0.55, 1.1])) * base_vol * 0.22

        # Periodic London/NY stop-hunt sweep followed by institutional impulse continuation
        if in_killzone and i > 30 and i % 85 == 0:
            sweep_dir = 1.0 if regime_drift >= 0 else -1.0
            op = price
            # Wick sweeps liquidity opposite to true move, then closes strongly in trend direction
            cl = op + sweep_dir * base_vol * 1.9
            hi = max(op, cl) + (base_vol * 0.4 if sweep_dir > 0 else base_vol * 2.6)
            lo = min(op, cl) - (base_vol * 2.6 if sweep_dir > 0 else base_vol * 0.4)
            price = cl
        elif in_killzone and i > 31 and (i - 1) % 85 == 0:
            # Displacement candle creating clean FVG
            sweep_dir = 1.0 if regime_drift >= 0 else -1.0
            op = price
            cl = op + sweep_dir * base_vol * 2.3
            hi = max(op, cl) + base_vol * 0.35
            lo = min(op, cl) - base_vol * 0.20
            price = cl
        else:
            op = price
            step = regime_drift * sess_mult + float(rng.normal(0.0, base_vol * sess_mult * 0.72))
            cl = op + step
            hi = max(op, cl) + abs(float(rng.normal(0.0, base_vol * 0.38)))
            lo = min(op, cl) - abs(float(rng.normal(0.0, base_vol * 0.38)))
            price = cl

        opens[i] = round(op, 5)
        closes[i] = round(cl, 5)
        highs[i] = round(hi, 5)
        lows[i] = round(lo, 5)

        # Rollover hour (21-22 UTC) widens spread realistically
        sp_mult = 2.1 if hr in (21, 22) else (0.92 if in_killzone else 1.18)
        spreads[i] = round(max(4.0, float(rng.normal(base_spread * sp_mult, base_spread * 0.1))), 1)
        volumes[i] = int(rng.integers(350, 1900) if in_killzone else rng.integers(90, 480))

    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, utc=True),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "spread": spreads,
    })


@dataclass
class _SimTrade:
    side: Side
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    sl: float
    tp: float
    initial_sl: float
    lots: float
    risk_usd: float
    spread_cost_usd: float
    setup_type: str
    be_moved: bool = False


class SpreadAwareEventBacktester:
    """
    Tick/Bar-conservative event-driven backtester that enforces:
      - Zero look-ahead bias (signals evaluated on closed bar `i`, entered at bar `i+1` open + half-spread)
      - Pessimistic intra-bar ambiguity resolution: if BOTH SL and TP are touched inside the same bar,
        SL is assumed hit FIRST (prevents inflated fake scalping win rates!)
      - Break-even management at +1.0R
      - Ablation comparison: `quant_only` vs `agent_filtered`
    """

    def __init__(
        self,
        settings: AppSettings,
        profile: SymbolProfile,
        memory: StateAndMemoryStore,
        tick_value_per_point: float = 1.0,
    ) -> None:
        self.settings = settings
        self.profile = profile
        self.memory = memory
        self.tick_value = tick_value_per_point

    def run(
        self,
        m5_df: pd.DataFrame,
        starting_equity: float = 10000.0,
        use_agent_committee: bool = True,
    ) -> BacktestSummary:
        feat = detect_fvg_and_sweeps(m5_df)
        # Compute non-lookahead H1 proxy EMA on M5 (12x M5 bars = 1 hour)
        feat["h1_ema20"] = compute_ema(feat["close"], 20 * 12)
        feat["h1_ema50"] = compute_ema(feat["close"], 50 * 12)

        # Force fast offline committee mode during multi-thousand bar backtests
        bt_settings = self.settings.model_copy(update={"llm_enabled": False})
        committee = MultiAgentCommittee(bt_settings, self.memory)
        risk_engine = DeterministicRiskFirewall(bt_settings, {self.profile.canonical: self.profile})

        equity = starting_equity
        peak_equity = starting_equity
        max_dd_pct = 0.0
        active_trade: _SimTrade | None = None
        closed_trades: list[dict] = []
        equity_curve: list[dict] = []
        total_spread_cost = 0.0
        last_exit_idx = -99

        n = len(feat)
        for i in range(220, n - 1):
            row = feat.iloc[i]
            next_bar = feat.iloc[i + 1]
            ts: pd.Timestamp = row["timestamp"]
            spread_pts = float(row["spread"])
            half_sp = (spread_pts * self.profile.point) / 2.0

            # 1. Manage open position on bar `i` (conservative SL-first intra-bar rule)
            if active_trade is not None:
                bar_high_bid = float(row["high"]) - half_sp
                bar_low_bid = float(row["low"]) - half_sp
                bar_high_ask = float(row["high"]) + half_sp
                bar_low_ask = float(row["low"]) + half_sp

                exit_price: float | None = None
                exit_reason = ""

                if active_trade.side == Side.LONG:
                    # Pessimistic check: SL checked BEFORE TP on same bar
                    if bar_low_bid <= active_trade.sl:
                        exit_price = active_trade.sl
                        exit_reason = "BE_STOP" if active_trade.be_moved else "SL"
                    elif bar_high_bid >= active_trade.tp:
                        exit_price = active_trade.tp
                        exit_reason = "TP"
                    else:
                        # Check +1.0R Break-Even shift for subsequent bars
                        init_r = abs(active_trade.entry_price - active_trade.initial_sl)
                        if not active_trade.be_moved and init_r > 0 and (bar_high_bid - active_trade.entry_price) >= init_r:
                            active_trade.sl = round(active_trade.entry_price + spread_pts * self.profile.point, 5)
                            active_trade.be_moved = True
                else:
                    if bar_high_ask >= active_trade.sl:
                        exit_price = active_trade.sl
                        exit_reason = "BE_STOP" if active_trade.be_moved else "SL"
                    elif bar_low_ask <= active_trade.tp:
                        exit_price = active_trade.tp
                        exit_reason = "TP"
                    else:
                        init_r = abs(active_trade.initial_sl - active_trade.entry_price)
                        if not active_trade.be_moved and init_r > 0 and (active_trade.entry_price - bar_low_ask) >= init_r:
                            active_trade.sl = round(active_trade.entry_price - spread_pts * self.profile.point, 5)
                            active_trade.be_moved = True

                if exit_price is not None:
                    diff = (
                        (exit_price - active_trade.entry_price)
                        if active_trade.side == Side.LONG
                        else (active_trade.entry_price - exit_price)
                    )
                    pnl = round((diff / self.profile.point) * self.tick_value * active_trade.lots, 2)
                    r_mult = round(pnl / active_trade.risk_usd, 2) if active_trade.risk_usd > 0 else 0.0
                    equity = round(equity + pnl, 2)
                    peak_equity = max(peak_equity, equity)
                    dd = ((peak_equity - equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
                    max_dd_pct = max(max_dd_pct, dd)

                    closed_trades.append({
                        "entry_time": active_trade.entry_time.isoformat(),
                        "exit_time": ts.isoformat(),
                        "side": active_trade.side.value,
                        "setup_type": active_trade.setup_type,
                        "lots": active_trade.lots,
                        "entry": active_trade.entry_price,
                        "exit": exit_price,
                        "pnl_usd": pnl,
                        "r_multiple": r_mult,
                        "reason": exit_reason,
                    })
                    active_trade = None
                    last_exit_idx = i

            if i % 20 == 0:
                equity_curve.append({"timestamp": ts.isoformat(), "equity": equity})

            # 2. If flat and cooldown elapsed, check signal on closed bar `i` for entry at `i+1` open
            if active_trade is not None or (i - last_exit_idx) < max(3, bt_settings.cooldown_minutes_per_symbol // 5):
                continue

            if spread_pts > self.profile.max_spread_points:
                continue

            close = float(row["close"])
            atr = max(float(row["atr"]), self.profile.point * 20)
            rsi = float(row["rsi"])
            adx = float(row["adx"])
            ema20 = float(row["ema_20"])
            ema50 = float(row["ema_50"])
            vwap = float(row["vwap"])

            if close > float(row["h1_ema20"]) >= float(row["h1_ema50"]):
                htf_bias = "BULLISH"
            elif close < float(row["h1_ema20"]) <= float(row["h1_ema50"]):
                htf_bias = "BEARISH"
            else:
                htf_bias = "NEUTRAL"

            side: Side | None = None
            stype: SetupType | None = None
            score = 55.0
            notes: list[str] = []

            if bool(row["recent_sweep_low"]) and bool(row["bullish_fvg"]) and close > ema20 and rsi < 74:
                side = Side.LONG
                stype = SetupType.LIQUIDITY_SWEEP_FVG
                score = 78.0
                notes.append("Sweep low + Bullish FVG")
            elif bool(row["recent_sweep_high"]) and bool(row["bearish_fvg"]) and close < ema20 and rsi > 26:
                side = Side.SHORT
                stype = SetupType.LIQUIDITY_SWEEP_FVG
                score = 78.0
                notes.append("Sweep high + Bearish FVG")
            elif (
                htf_bias == "BULLISH"
                and close > vwap
                and ema20 > ema50
                and float(feat["low"].iloc[i - 1]) <= ema20 * 1.0005
                and close > float(feat["high"].iloc[i - 1])
                and 44 <= rsi <= 67
            ):
                side = Side.LONG
                stype = SetupType.VWAP_EMA_PULLBACK
                score = 70.0
                notes.append("VWAP/EMA20 bullish continuation")
            elif (
                htf_bias == "BEARISH"
                and close < vwap
                and ema20 < ema50
                and float(feat["high"].iloc[i - 1]) >= ema20 * 0.9995
                and close < float(feat["low"].iloc[i - 1])
                and 33 <= rsi <= 56
            ):
                side = Side.SHORT
                stype = SetupType.VWAP_EMA_PULLBACK
                score = 70.0
                notes.append("VWAP/EMA20 bearish continuation")

            if side is None or stype is None:
                continue

            in_prime = self.profile.is_in_prime_session(int(ts.hour))
            if (side == Side.LONG and htf_bias == "BULLISH") or (side == Side.SHORT and htf_bias == "BEARISH"):
                score += 9.0
            if in_prime:
                score += 7.0
            if self.profile.canonical == "XAUUSD":
                score += 4.0

            sess_lbl = "NY_OVERLAP_PRIME" if 12 <= ts.hour < 17 else ("LONDON_SESSION" if 7 <= ts.hour < 12 else "ASIAN_OFF_PEAK")
            next_open_mid = float(next_bar["open"])
            entry_fill = round(next_open_mid + half_sp, 5) if side == Side.LONG else round(next_open_mid - half_sp, 5)

            sl_dist = 1.45 * atr
            tp_dist = 3.20 * atr
            sp_ratio = (spread_pts * self.profile.point) / sl_dist
            if sp_ratio > self.profile.max_spread_to_sl_ratio:
                continue

            sl_price = round(entry_fill - sl_dist, 5) if side == Side.LONG else round(entry_fill + sl_dist, 5)
            tp_price = round(entry_fill + tp_dist, 5) if side == Side.LONG else round(entry_fill - tp_dist, 5)

            cand = CandidateSetup(
                setup_id=f"BT-{i}",
                timestamp=ts.to_pydatetime(),
                symbol=self.profile.canonical,
                tier=self.profile.tier,
                side=side,
                setup_type=stype,
                timeframe="M5",
                htf_bias=htf_bias,
                entry_price=entry_fill,
                stop_loss=sl_price,
                take_profit=tp_price,
                atr=round(atr, 5),
                adx=round(adx, 2),
                rsi=round(rsi, 2),
                vwap_distance_atr=round((close - vwap) / atr, 3),
                spread_points=spread_pts,
                spread_to_sl_ratio=round(sp_ratio, 4),
                rr_ratio=round(tp_dist / sl_dist, 2),
                quant_score=min(99.0, score),
                session_label=sess_lbl,
                notes=notes,
            )

            if use_agent_committee:
                verdict = committee.evaluate_candidate_sync(cand)
                if verdict.action not in (AgentAction.EXECUTE_LONG, AgentAction.EXECUTE_SHORT):
                    continue

            sim_quote = QuoteTick(
                symbol=self.profile.canonical,
                broker_symbol=self.profile.canonical,
                timestamp=ts.to_pydatetime(),
                bid=round(next_open_mid - half_sp, 5),
                ask=round(next_open_mid + half_sp, 5),
                point=self.profile.point,
                tick_value=self.tick_value,
                tick_size=self.profile.point,
            )
            lots, risk_usd = risk_engine.calculate_exact_lots(
                equity=equity,
                risk_pct=bt_settings.risk_per_trade_pct,
                sl_distance_price=sl_dist,
                quote=sim_quote,
            )
            if lots <= 0:
                continue

            sp_cost = round(spread_pts * self.tick_value * lots, 2)
            total_spread_cost += sp_cost
            active_trade = _SimTrade(
                side=side,
                entry_idx=i + 1,
                entry_time=next_bar["timestamp"],
                entry_price=entry_fill,
                sl=sl_price,
                tp=tp_price,
                initial_sl=sl_price,
                lots=lots,
                risk_usd=risk_usd,
                spread_cost_usd=sp_cost,
                setup_type=stype.value,
            )

        equity_curve.append({"timestamp": feat.iloc[-1]["timestamp"].isoformat(), "equity": equity})

        wins = [t for t in closed_trades if t["pnl_usd"] > 0]
        losses = [t for t in closed_trades if t["pnl_usd"] <= 0]
        gross_win = sum(t["pnl_usd"] for t in wins)
        gross_loss = abs(sum(t["pnl_usd"] for t in losses))
        pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else (9.99 if gross_win > 0 else 0.0)
        win_rate = round((len(wins) / len(closed_trades)) * 100.0, 2) if closed_trades else 0.0
        avg_r = round(float(np.mean([t["r_multiple"] for t in closed_trades])), 2) if closed_trades else 0.0

        if len(closed_trades) >= 3:
            rets = np.array([t["pnl_usd"] / starting_equity for t in closed_trades])
            std = float(np.std(rets, ddof=1))
            # Annualize assuming ~252 trading days
            trades_per_year = max(20.0, len(closed_trades) * (252.0 * 288.0 / max(1, n)))
            sharpe = round(float((np.mean(rets) / std) * np.sqrt(trades_per_year)), 2) if std > 1e-9 else 0.0
        else:
            sharpe = 0.0

        return BacktestSummary(
            symbol=self.profile.canonical,
            bars_tested=n,
            mode="agent_committee_filtered" if use_agent_committee else "raw_quant_baseline",
            starting_equity=starting_equity,
            ending_equity=equity,
            net_return_pct=round(((equity - starting_equity) / starting_equity) * 100.0, 2),
            win_rate_pct=win_rate,
            profit_factor=pf,
            sharpe_ratio=sharpe,
            max_drawdown_pct=round(max_dd_pct, 2),
            total_trades=len(closed_trades),
            wins=len(wins),
            losses=len(losses),
            avg_r_multiple=avg_r,
            spread_cost_usd=round(total_spread_cost, 2),
            equity_curve=equity_curve,
            sample_trades=closed_trades[-15:],
        )
