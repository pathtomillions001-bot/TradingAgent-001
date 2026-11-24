"""Comprehensive Unit & Integration Tests for TradingAgent-001."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from fastapi.testclient import TestClient

from tradingagent.backtest import SpreadAwareEventBacktester, generate_realistic_intraday_dataset
from tradingagent.broker import SymbolMapper
from tradingagent.config import AppSettings, load_symbol_profiles
from tradingagent.engine import TradingAgentEngine
from tradingagent.memory import StateAndMemoryStore
from tradingagent.models import (
    AccountSnapshot,
    CandidateSetup,
    OpenPosition,
    QuoteTick,
    SetupType,
    Side,
)
from tradingagent.risk import DeterministicRiskFirewall
from tradingagent.web import create_app


def _make_test_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        trading_mode="paper",
        llm_enabled=False,
        sqlite_path=tmp_path / "test_state.sqlite3",
        duckdb_path=tmp_path / "test_market.duckdb",
    )


def test_symbol_auto_mapper_discovers_broker_suffixes() -> None:
    profiles = load_symbol_profiles()
    mapper = SymbolMapper(profiles)
    broker_list = ["XAUUSD.pro", "USTEC", "BTCUSDm", "EURUSD.pro", "GBPUSD.pro", "USDCAD.pro"]
    discovered = mapper.discover(broker_list)

    assert discovered["XAUUSD"] == "XAUUSD.pro"
    assert discovered["NAS100"] == "USTEC"
    assert discovered["BTCUSD"] == "BTCUSDm"
    assert mapper.to_canonical("USTEC") == "NAS100"

    # Exact JustMarkets-Demo symbol list regression test
    jm_mapper = SymbolMapper(profiles)
    jm_symbols = [
        "EURUSD.m", "GBPUSD.m", "USDCAD.m", "XAUUSD.m", "XAUAUD.m",
        "XAUEUR.m", "BTCUSD.m", "BTCEUR.m", "UK100.std", "US100.std",
    ]
    jm_discovered = jm_mapper.discover(jm_symbols)
    assert jm_discovered["XAUUSD"] == "XAUUSD.m"
    assert jm_discovered["NAS100"] == "US100.std"
    assert jm_discovered["BTCUSD"] == "BTCUSD.m"
    assert jm_discovered["EURUSD"] == "EURUSD.m"
    assert jm_discovered["GBPUSD"] == "GBPUSD.m"
    assert jm_discovered["USDCAD"] == "USDCAD.m"


def test_blank_env_fields_and_micro_account_lot_sizing(tmp_path: Path) -> None:
    # Blank MT5_LOGIN="" and UTF-8 BOM in .env must never raise ValidationError
    env_path = tmp_path / ".env"
    env_path.write_bytes(b"\xef\xbb\xbfTRADING_MODE=paper\nMT5_LOGIN=\nMT5_PASSWORD=\nMT5_SERVER=\n")
    s = AppSettings(_env_file=str(env_path), sqlite_path=tmp_path / "s.db", duckdb_path=tmp_path / "d.db")
    assert s.mt5_login is None
    assert s.mt5_password is None

    # JustMarkets $374.38 Demo Account micro-lot sizing check
    profiles = load_symbol_profiles()
    firewall = DeterministicRiskFirewall(s, profiles)
    gold_3digit_quote = QuoteTick(
        symbol="XAUUSD",
        broker_symbol="XAUUSD.m",
        timestamp=datetime.now(timezone.utc),
        bid=2910.000,
        ask=2910.220,
        point=0.001,       # 3-decimal broker quote
        tick_value=0.10,   # $0.10 per 0.001 move on 1.0 lot ($100 per $1.00 move)
        tick_size=0.001,
    )
    # Canonical spread must normalize $0.220 -> 22.0 canonical points (not 220!)
    assert round(profiles["XAUUSD"].canonical_spread_points(gold_3digit_quote.spread_price), 1) == 22.0

    # $4.20 SL on 0.01 lot = $4.20 risk = 1.12% of $374.38 (<= 1.25% micro cap -> approved at 0.01 lot)
    lots_ok, risk_ok = firewall.calculate_exact_lots(374.38, 0.75, 4.20, gold_3digit_quote)
    assert lots_ok == 0.01
    assert risk_ok == 4.20

    # $7.50 SL on 0.01 lot = $7.50 risk = 2.00% of $374.38 (> 1.25% micro cap -> safely blocked!)
    lots_blocked, risk_blocked = firewall.calculate_exact_lots(374.38, 0.75, 7.50, gold_3digit_quote)
    assert lots_blocked == 0.0
    assert risk_blocked == 7.50


def test_exact_lot_sizing_and_net_usd_correlation_guard(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path)
    profiles = load_symbol_profiles()
    firewall = DeterministicRiskFirewall(settings, profiles)

    account = AccountSnapshot(equity=10000.0, balance=10000.0, free_margin=10000.0, day_starting_equity=10000.0)
    gold_quote = QuoteTick(
        symbol="XAUUSD",
        broker_symbol="XAUUSDm",
        timestamp=datetime.now(timezone.utc),
        bid=2680.00,
        ask=2680.20,
        point=0.01,
        tick_value=1.00,  # $1 per $0.01 point on 1.0 lot (100 oz)
        tick_size=0.01,
    )

    # Risk 0.75% of $10,000 = $75. SL distance = $2.50 (250 points) -> $250/lot -> 0.30 lots exact
    lots, risk_usd = firewall.calculate_exact_lots(10000.0, 0.75, 2.50, gold_quote)
    assert lots == 0.30
    assert risk_usd == 75.00

    # Now simulate already holding LONG XAUUSD (-0.75% Short USD) + LONG EURUSD (-0.40% Short USD) = -1.15% Short USD
    existing_positions = [
        OpenPosition(
            ticket=1,
            symbol="XAUUSD",
            broker_symbol="XAUUSDm",
            side=Side.LONG,
            volume=0.30,
            entry_price=2680.20,
            current_price=2681.00,
            sl=2677.70,
            tp=2686.00,
            initial_risk_usd=75.0,
            unrealized_pnl=24.0,
            opened_at=datetime.now(timezone.utc),
        ),
        OpenPosition(
            ticket=2,
            symbol="EURUSD",
            broker_symbol="EURUSDm",
            side=Side.LONG,
            volume=0.20,
            entry_price=1.06500,
            current_price=1.06520,
            sl=1.06300,
            tp=1.06950,
            initial_risk_usd=40.0,
            unrealized_pnl=4.0,
            opened_at=datetime.now(timezone.utc),
        ),
    ]

    # Increase max_open_positions so we specifically test the USD correlation guard on GBPUSD LONG (-0.75% more Short USD => -1.90% > 1.25% cap!)
    firewall.settings.max_open_positions = 5
    gbp_quote = QuoteTick(
        symbol="GBPUSD",
        broker_symbol="GBPUSDm",
        timestamp=datetime.now(timezone.utc),
        bid=1.27800,
        ask=1.27812,
        point=0.00001,
        tick_value=1.00,
        tick_size=0.00001,
    )
    gbp_setup = CandidateSetup(
        setup_id="TEST-GBP",
        timestamp=datetime.now(timezone.utc),
        symbol="GBPUSD",
        tier=3,
        side=Side.LONG,
        setup_type=SetupType.VWAP_EMA_PULLBACK,
        htf_bias="BULLISH",
        entry_price=1.27812,
        stop_loss=1.27612,
        take_profit=1.28252,
        atr=0.0012,
        adx=26.0,
        rsi=55.0,
        vwap_distance_atr=0.4,
        spread_points=12.0,
        spread_to_sl_ratio=0.06,
        rr_ratio=2.2,
        quant_score=80.0,
        session_label="NY_OVERLAP_PRIME",
    )

    res = firewall.validate_and_size(gbp_setup, account, gbp_quote, existing_positions)
    assert not res.approved
    assert any("USD_CORRELATION_CAP" in r for r in res.rejection_reasons)


def test_daily_drawdown_lock_blocks_new_orders(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path)
    profiles = load_symbol_profiles()
    firewall = DeterministicRiskFirewall(settings, profiles)

    # 3.0% daily drawdown exceeds 2.5% max_daily_loss_pct
    account = AccountSnapshot(
        equity=9700.0,
        balance=9700.0,
        free_margin=9700.0,
        day_starting_equity=10000.0,
        peak_equity=10000.0,
    )
    gold_quote = QuoteTick(
        symbol="XAUUSD",
        broker_symbol="XAUUSDm",
        timestamp=datetime.now(timezone.utc),
        bid=2680.00,
        ask=2680.18,
        point=0.01,
        tick_value=1.00,
        tick_size=0.01,
    )
    setup = CandidateSetup(
        setup_id="TEST-DD",
        timestamp=datetime.now(timezone.utc),
        symbol="XAUUSD",
        tier=1,
        side=Side.LONG,
        setup_type=SetupType.LIQUIDITY_SWEEP_FVG,
        htf_bias="BULLISH",
        entry_price=2680.18,
        stop_loss=2677.68,
        take_profit=2685.80,
        atr=1.7,
        adx=28.0,
        rsi=54.0,
        vwap_distance_atr=0.3,
        spread_points=18.0,
        spread_to_sl_ratio=0.072,
        rr_ratio=2.25,
        quant_score=88.0,
        session_label="NY_OVERLAP_PRIME",
    )
    res = firewall.validate_and_size(setup, account, gold_quote, [])
    assert not res.approved
    assert any("DAILY_LOSS_LOCK" in r for r in res.rejection_reasons)


def test_end_to_end_engine_cycle_and_kill_switch(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path)
    engine = TradingAgentEngine(settings=settings)
    import asyncio

    cycle_out = asyncio.run(engine.run_single_cycle(auto_execute=True))
    assert cycle_out["trading_mode"] == "paper"
    assert cycle_out["candidates_found"] >= 1

    # Verify Gold-first priority: first evaluated candidate should be XAUUSD
    first_eval = cycle_out["evaluations"][0]
    assert first_eval["setup"]["symbol"] == "XAUUSD"

    # Trigger emergency kill switch and verify all positions close
    kill_out = engine.trigger_emergency_kill_switch()
    assert kill_out["kill_switch_active"] is True
    assert len(engine.broker.get_open_positions()) == 0


def test_spread_aware_ablation_backtester_and_api(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path)
    profiles = load_symbol_profiles()
    mem = StateAndMemoryStore(settings.sqlite_path)
    df = generate_realistic_intraday_dataset("XAUUSD", n_bars=900, seed=101)
    tester = SpreadAwareEventBacktester(settings, profiles["XAUUSD"], mem, tick_value_per_point=1.0)

    raw_bt = tester.run(df, use_agent_committee=False)
    agent_bt = tester.run(df, use_agent_committee=True)
    assert raw_bt.bars_tested == 900
    assert agent_bt.total_trades <= raw_bt.total_trades

    app = create_app(settings=settings)
    client = TestClient(app)
    r = client.get("/api/status")
    assert r.status_code == 200
    assert r.json()["watchlist"][0]["symbol"] == "XAUUSD"

    # Regression test: /api/backtest inside running async event loop must succeed (HTTP 200)
    bt_resp = client.post("/api/backtest", json={"symbol": "XAUUSD", "bars": 600, "starting_equity": 10000})
    assert bt_resp.status_code == 200
    assert bt_resp.json()["bars_tested"] == 600

    # Regression test: if one symbol raises RuntimeError on broker, /api/status & /api/scan stay 200 OK
    orig_get_quote = app.state.engine.broker.get_quote

    def flaky_quote(sym: str) -> QuoteTick:
        if sym == "NAS100":
            raise RuntimeError("Failed to fetch MT5 tick/info for NAS100: (-4, 'Terminal: Not found')")
        return orig_get_quote(sym)

    app.state.engine.broker.get_quote = flaky_quote
    r_degraded = client.get("/api/status")
    assert r_degraded.status_code == 200
    nas_entry = next(w for w in r_degraded.json()["watchlist"] if w["symbol"] == "NAS100")
    assert nas_entry["available"] is False
    gold_entry = next(w for w in r_degraded.json()["watchlist"] if w["symbol"] == "XAUUSD")
    assert gold_entry["available"] is True

    scan_degraded = client.post("/api/scan?auto_execute=false")
    assert scan_degraded.status_code == 200
