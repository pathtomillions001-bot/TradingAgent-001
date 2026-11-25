"""Export & Normalize Your Real MT5 Broker's Historical M5/M1 Bars to Parquet on Disk (`data/catalog/`)."""

from __future__ import annotations

from pathlib import Path

from tradingagent.backtest import load_backtest_dataset
from tradingagent.broker.mt5_windows import MT5WindowsAdapter
from tradingagent.config import AppSettings, load_symbol_profiles


def export_broker_history(timeframe: str = "M5", bars: int = 30000) -> list[Path]:
    settings = AppSettings()
    if settings.trading_mode == "paper":
        settings = settings.model_copy(update={"trading_mode": "mt5_demo"})
    profiles = load_symbol_profiles(settings.symbols_config_path)
    adapter = MT5WindowsAdapter(settings, profiles)
    if not adapter.connect():
        raise RuntimeError("Could not connect to local Windows MT5 terminal. Open MT5 and enable Algo Trading.")

    saved: list[Path] = []
    try:
        for sym, prof in profiles.items():
            try:
                bundle = load_backtest_dataset(
                    symbol=sym,
                    bars=bars,
                    settings=settings,
                    profile=prof,
                    broker=adapter,
                    source="mt5",
                    timeframe=timeframe,
                )
                p = Path(bundle.cache_path)
                saved.append(p)
                print(
                    f"Saved {len(bundle.df):,} real {timeframe} bars for {sym} "
                    f"({adapter.mapper.to_broker(sym)}, tick_val=${bundle.tick_value_per_point:.4f}/pt) -> {p}"
                )
            except Exception as exc:
                print(f"Skipped {sym} ({adapter.mapper.to_broker(sym)}): {exc}")
    finally:
        adapter.disconnect()

    return saved


if __name__ == "__main__":
    export_broker_history()
