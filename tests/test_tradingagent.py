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

    # Tier-1 Gold 0.01 min-lot allowance on $360.23 JustMarkets Demo equity:
    # Exact London session scenario: $6.16 SL on 0.01 lot = $6.16 risk (1.71% <= 3.25% Gold min-lot cap) -> APPROVED!
    lots_gold_london, risk_gold_london = firewall.calculate_exact_lots(360.23, 0.75, 6.16, gold_3digit_quote)
    assert lots_gold_london == 0.01
    assert risk_gold_london == 6.16

    # Extreme news spike ($15.00 SL on Gold = 4.16% > 3.25% Gold cap) -> STILL SAFELY BLOCKED!
    lots_gold_spike, risk_gold_spike = firewall.calculate_exact_lots(360.23, 0.75, 15.00, gold_3digit_quote)
    assert lots_gold_spike == 0.0
    assert risk_gold_spike == 15.00

    # Cross-asset contract awareness on the EXACT SAME $360.23 account:
    # EURUSD 12-pip SL (0.00120) -> $1.20/lot*0.01 = $1.20 per 0.01 lot -> 0.75% target ($2.70) sizes to 0.02 lots ($2.40)!
    eur_quote = QuoteTick(
        symbol="EURUSD",
        broker_symbol="EURUSD.m",
        timestamp=datetime.now(timezone.utc),
        bid=1.07200,
        ask=1.07210,
        point=0.00001,
        tick_value=1.00,  # $1 per point on 1.00 standard lot (100,000 EUR) => $10/pip on 1.0 lot, $0.10/pip on 0.01 lot
        tick_size=0.00001,
    )
    lots_eur_small, risk_eur_small = firewall.calculate_exact_lots(360.23, 0.75, 0.00120, eur_quote)
    assert lots_eur_small == 0.02
    assert risk_eur_small == 2.40

    # On $10,000 equity (0.75% = $75 risk): EURUSD 12-pip SL = 0.62 lots ($74.40) vs XAUUSD $6.16 SL = 0.12 lots ($73.92)
    lots_eur_10k, _ = firewall.calculate_exact_lots(10000.0, 0.75, 0.00120, eur_quote)
    lots_gold_10k, _ = firewall.calculate_exact_lots(10000.0, 0.75, 6.16, gold_3digit_quote)
    assert lots_eur_10k == 0.62
    assert lots_gold_10k == 0.12

    # Full validate_and_size check for Solo Gold 0.01 min-lot + automatic USD Correlation Lock until Break-Even!
    small_acct = AccountSnapshot(
        equity=360.23, balance=360.23, free_margin=360.23, day_starting_equity=360.23, peak_equity=360.23, leverage=500
    )
    gold_short_setup = CandidateSetup(
        setup_id="TEST-GOLD-MINLOT",
        timestamp=datetime.now(timezone.utc),
        symbol="XAUUSD",
        tier=1,
        side=Side.SHORT,
        setup_type=SetupType.VWAP_EMA_PULLBACK,
        htf_bias="BEARISH",
        entry_price=2910.000,
        stop_loss=2916.160,
        take_profit=2896.380,
        atr=4.25,
        adx=27.0,
        rsi=44.0,
        vwap_distance_atr=-0.4,
        spread_points=22.0,
        spread_to_sl_ratio=0.035,
        rr_ratio=2.21,
        quant_score=91.0,
        session_label="LONDON_SESSION",
    )
    gold_res = firewall.validate_and_size(gold_short_setup, small_acct, gold_3digit_quote, [])
    assert gold_res.approved is True
    assert gold_res.calculated_lots == 0.01
    assert gold_res.risk_usd == 6.16

    # While Gold 0.01 SHORT (+1.71% Long USD) is open before Break-Even, stacking EURUSD SHORT (+0.67% Long USD) is LOCKED!
    open_gold_pos = OpenPosition(
        ticket=99,
        symbol="XAUUSD",
        broker_symbol="XAUUSD.m",
        side=Side.SHORT,
        volume=0.01,
        entry_price=2910.000,
        current_price=2909.500,
        sl=2916.160,
        tp=2896.380,
        initial_risk_usd=6.16,
        unrealized_pnl=0.50,
        opened_at=datetime.now(timezone.utc),
        be_moved=False,
    )
    eur_short_setup = CandidateSetup(
        setup_id="TEST-EUR-STACK",
        timestamp=datetime.now(timezone.utc),
        symbol="EURUSD",
        tier=3,
        side=Side.SHORT,
        setup_type=SetupType.VWAP_EMA_PULLBACK,
        htf_bias="BEARISH",
        entry_price=1.07200,
        stop_loss=1.07320,
        take_profit=1.06930,
        atr=0.0008,
        adx=25.0,
        rsi=45.0,
        vwap_distance_atr=-0.3,
        spread_points=10.0,
        spread_to_sl_ratio=0.08,
        rr_ratio=2.25,
        quant_score=82.0,
        session_label="LONDON_SESSION",
    )
    eur_locked = firewall.validate_and_size(eur_short_setup, small_acct, eur_quote, [open_gold_pos])
    assert eur_locked.approved is False
    assert any("USD_CORRELATION_CAP" in r for r in eur_locked.rejection_reasons)

    # Once Gold moves SL to Break-Even (be_moved=True), EURUSD SHORT at 0.02 lots ($2.40 = 0.67%) is UNLOCKED!
    open_gold_pos.be_moved = True
    eur_unlocked = firewall.validate_and_size(eur_short_setup, small_acct, eur_quote, [open_gold_pos])
    assert eur_unlocked.approved is True
    assert eur_unlocked.calculated_lots == 0.02


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
    with TestClient(app) as client:
        r = client.get("/api/status")
        assert r.status_code == 200
        status_json = r.json()
        assert status_json["watchlist"][0]["symbol"] == "XAUUSD"
        assert "unit_label" in status_json["watchlist"][0]
        assert status_json["autopilot"]["enabled"] is False

        # Toggle built-in Autopilot on & off via API
        ap_on = client.post("/api/autopilot", json={"enabled": True})
        assert ap_on.status_code == 200
        assert ap_on.json()["enabled"] is True
        ap_off = client.post("/api/autopilot", json={"enabled": False})
        assert ap_off.status_code == 200
        assert ap_off.json()["enabled"] is False

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
