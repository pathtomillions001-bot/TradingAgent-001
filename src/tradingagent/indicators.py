"""Fast Vectorized Technical, Session VWAP & Smart-Money-Concepts (SMC) Indicators."""

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
    Vectorized Smart Money Concepts (ICT/SMC) detector:
    - Bullish FVG: low[i] > high[i-2] with strong bullish middle candle
    - Bearish FVG: high[i] < low[i-2] with strong bearish middle candle
    - Liquidity Sweep Low: low pierces rolling 20-bar prior low & closes back above or displaces strongly
    - Liquidity Sweep High: high pierces rolling 20-bar prior high & closes back below or displaces strongly
    """
    out = df.copy()
    out["ema_20"] = compute_ema(out["close"], 20)
    out["ema_50"] = compute_ema(out["close"], 50)
    out["ema_200"] = compute_ema(out["close"], 200)
    out["atr"] = compute_atr(out, 14)
    out["rsi"] = compute_rsi(out["close"], 14)
    out["adx"] = compute_adx(out, 14)
    out["vwap"] = compute_session_vwap(out)

    prior_liq_low = out["low"].shift(1).rolling(lookback_liquidity, min_periods=5).min()
    prior_liq_high = out["high"].shift(1).rolling(lookback_liquidity, min_periods=5).max()

    out["prior_liq_low"] = prior_liq_low
    out["prior_liq_high"] = prior_liq_high

    # 3-candle Fair Value Gaps (FVG)
    body_mid = (out["close"].shift(1) - out["open"].shift(1)).abs()
    out["bullish_fvg"] = (out["low"] > out["high"].shift(2)) & (out["close"].shift(1) > out["open"].shift(1)) & (
        body_mid > 0.35 * out["atr"]
    )
    out["bearish_fvg"] = (out["high"] < out["low"].shift(2)) & (out["close"].shift(1) < out["open"].shift(1)) & (
        body_mid > 0.35 * out["atr"]
    )

    # Recent sweep within last 12 bars
    raw_sweep_low = out["low"] < prior_liq_low
    raw_sweep_high = out["high"] > prior_liq_high
    out["recent_sweep_low"] = raw_sweep_low.rolling(12, min_periods=1).max().astype(bool)
    out["recent_sweep_high"] = raw_sweep_high.rolling(12, min_periods=1).max().astype(bool)
    out["recent_bull_fvg"] = out["bullish_fvg"].rolling(10, min_periods=1).max().astype(bool)
    out["recent_bear_fvg"] = out["bearish_fvg"].rolling(10, min_periods=1).max().astype(bool)

    return out
