"""Fast Vectorized Technical, Regime, Session Range, VWAP & Smart-Money-Concepts (SMC) Indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().bfill()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = compute_atr(df, period=period).replace(0.0, np.nan)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean().fillna(20.0)


def compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    """Computes daily UTC session anchored VWAP."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].clip(lower=1.0)
    pv = typical * vol
    dates = pd.to_datetime(df["timestamp"], utc=True).dt.date
    cum_pv = pv.groupby(dates).cumsum()
    cum_vol = vol.groupby(dates).cumsum()
    return cum_pv / cum_vol


def detect_fvg_and_sweeps(df: pd.DataFrame, lookback_liquidity: int = 20) -> pd.DataFrame:
    """
    Vectorized Institutional Regime + Smart Money Concepts (ICT/SMC) + Volatility Squeeze Engine:
      - EMA 20 / 50 / 200, ATR(14), ATR Expansion Ratio, ADX(14), RSI(14), Session VWAP
      - Kaufman Efficiency Ratio (`efficiency_ratio`, 20-bar Hurst/persistence proxy)
      - Relative Tick Volume (`rvol` vs 20-bar mean)
      - Bollinger(20, 2.0) inside Keltner(20, 1.5*ATR) Volatility Squeeze (`bb_squeeze`, `recent_squeeze`)
      - Daily 00:00–07:00 UTC Asian Session High/Low (`asian_high`, `asian_low`)
      - 6-bar Structural Swing High/Low (`swing_high_6`, `swing_low_6`) for wick-anchored stops
      - Fresh 6-bar Liquidity Sweeps & 3-bar Displacement Fair Value Gaps (FVG)
    """
    out = df.copy()
    ts_utc = pd.to_datetime(out["timestamp"], utc=True)

    out["ema_20"] = compute_ema(out["close"], 20)
    out["ema_50"] = compute_ema(out["close"], 50)
    out["ema_200"] = compute_ema(out["close"], 200)
    out["atr"] = compute_atr(out, 14)
    out["atr_sma50"] = out["atr"].rolling(50, min_periods=10).mean().bfill()
    out["atr_ratio"] = (out["atr"] / out["atr_sma50"].replace(0.0, np.nan)).fillna(1.0)

    out["rsi"] = compute_rsi(out["close"], 14)
    out["adx"] = compute_adx(out, 14)
    out["vwap"] = compute_session_vwap(out)

    # Relative Volume (RVOL) — critical breakout & sweep institutional participation filter
    vol_clean = out["volume"].clip(lower=1.0)
    vol_sma20 = vol_clean.rolling(20, min_periods=5).mean().bfill()
    out["rvol"] = (vol_clean / vol_sma20.replace(0.0, np.nan)).fillna(1.0)

    # Directional Persistence / Efficiency Ratio (fast vectorized Hurst proxy in [0, 1])
    net_move_20 = (out["close"] - out["close"].shift(20)).abs()
    path_move_20 = out["close"].diff().abs().rolling(20, min_periods=5).sum().replace(0.0, np.nan)
    out["efficiency_ratio"] = (net_move_20 / path_move_20).fillna(0.35).clip(0.0, 1.0)

    # Candle body displacement metrics
    bar_range = (out["high"] - out["low"]).replace(0.0, np.nan)
    body_abs = (out["close"] - out["open"]).abs()
    out["body_ratio"] = (body_abs / bar_range).fillna(0.0)
    out["body_atr"] = (body_abs / out["atr"].replace(0.0, np.nan)).fillna(0.0)

    # Bollinger Bands (20, 2.0) inside Keltner Channels (20, 1.5*ATR) Volatility Squeeze
    bb_std = out["close"].rolling(20, min_periods=10).std().fillna(0.0)
    out["bb_squeeze"] = (2.0 * bb_std) <= (1.5 * out["atr"])
    out["recent_squeeze"] = out["bb_squeeze"].shift(1).rolling(8, min_periods=1).max().fillna(0).astype(bool)

    # Daily 00:00–07:00 UTC Asian Session High & Low mapped onto the rest of each UTC day
    utc_dates = ts_utc.dt.date
    asian_mask = ts_utc.dt.hour < 7
    if asian_mask.any():
        asian_df = out.loc[asian_mask]
        a_dates = utc_dates.loc[asian_mask]
        a_high_map = asian_df["high"].groupby(a_dates).max()
        a_low_map = asian_df["low"].groupby(a_dates).min()
        out["asian_high"] = utc_dates.map(a_high_map).astype(float)
        out["asian_low"] = utc_dates.map(a_low_map).astype(float)
    else:
        out["asian_high"] = np.nan
        out["asian_low"] = np.nan

    # Structural swing wicks (for wick-anchored stop placement)
    out["swing_low_6"] = out["low"].rolling(6, min_periods=1).min()
    out["swing_high_6"] = out["high"].rolling(6, min_periods=1).max()

    prior_liq_low = out["low"].shift(1).rolling(lookback_liquidity, min_periods=5).min()
    prior_liq_high = out["high"].shift(1).rolling(lookback_liquidity, min_periods=5).max()
    out["prior_liq_low"] = prior_liq_low
    out["prior_liq_high"] = prior_liq_high

    # 3-candle Fair Value Gaps (FVG) with institutional displacement middle candle
    body_mid = (out["close"].shift(1) - out["open"].shift(1)).abs()
    out["bullish_fvg"] = (out["low"] > out["high"].shift(2)) & (out["close"].shift(1) > out["open"].shift(1)) & (
        body_mid > 0.38 * out["atr"]
    )
    out["bearish_fvg"] = (out["high"] < out["low"].shift(2)) & (out["close"].shift(1) < out["open"].shift(1)) & (
        body_mid > 0.38 * out["atr"]
    )

    # Sweeps: keep 12-bar fields for backward compatibility + tight 6-bar fresh sweep fields for sniper entries
    raw_sweep_low = out["low"] < prior_liq_low
    raw_sweep_high = out["high"] > prior_liq_high
    out["recent_sweep_low"] = raw_sweep_low.rolling(12, min_periods=1).max().astype(bool)
    out["recent_sweep_high"] = raw_sweep_high.rolling(12, min_periods=1).max().astype(bool)
    out["fresh_sweep_low"] = raw_sweep_low.rolling(6, min_periods=1).max().astype(bool)
    out["fresh_sweep_high"] = raw_sweep_high.rolling(6, min_periods=1).max().astype(bool)
    out["recent_bull_fvg"] = out["bullish_fvg"].rolling(5, min_periods=1).max().astype(bool)
    out["recent_bear_fvg"] = out["bearish_fvg"].rolling(5, min_periods=1).max().astype(bool)

    return out
