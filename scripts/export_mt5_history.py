"""Export your real MT5 Broker's Historical M1/M5 Bars to Parquet/DuckDB for Local Backtesting."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from tradingagent.broker.mt5_windows import MT5WindowsAdapter
from tradingagent.config import AppSettings, load_symbol_profiles


def export_broker_history(timeframe: str = "M5", bars: int = 20000) -> list[Path]:
    settings = AppSettings(trading_mode="mt5_demo")
    profiles = load_symbol_profiles(settings.symbols_config_path)
    adapter = MT5WindowsAdapter(settings, profiles)
    if not adapter.connect():
        raise RuntimeError("Could not connect to local Windows MT5 terminal. Open MT5 and enable Algo Trading.")

    out_dir = Path("data/catalog")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    try:
        for sym in profiles:
            df: pd.DataFrame = adapter.get_bars(sym, timeframe=timeframe, count=bars)
            path = out_dir / f"{sym}_{timeframe}.parquet"
            df.to_parquet(path, index=False)
            saved.append(path)
            print(f"Saved {len(df):,} bars for {sym} -> {path}")
    finally:
        adapter.disconnect()

    return saved


if __name__ == "__main__":
    export_broker_history()
