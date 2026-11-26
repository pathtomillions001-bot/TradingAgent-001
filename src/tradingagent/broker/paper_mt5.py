"""Local MT5-Fidelity Paper Simulator (zero external dependencies, realistic bid/ask spreads & margin)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

from tradingagent.broker import BrokerAdapter, SymbolMapper
from tradingagent.config import AppSettings, SymbolProfile
from tradingagent.models import (
    AccountSnapshot,
    ExecutionReceipt,
    OpenPosition,
    QuoteTick,
    Side,
)


DEFAULT_PRICES: dict[str, tuple[float, float, float, float]] = {
    # canonical: (base_mid, m5_vol_price, typical_spread_points, tick_value_per_point_1lot)
    "XAUUSD": (2685.50, 1.85, 18.0, 1.00),      # $0.01 point -> 100 oz * $0.01 = $1.00/point
    "NAS100": (20850.00, 19.50, 140.0, 0.01),   # 0.01 point, contract_size=1 -> $0.01/point ($1/index pt)
    "BTCUSD": (94500.00, 145.00, 1800.0, 0.01), # 0.01 point, contract_size=1 -> $0.01/point ($1/$1 move)
    "EURUSD": (1.06450, 0.00055, 9.0, 1.00),    # 0.00001 point * 100,000 = $1.00/point ($10/pip)
    "GBPUSD": (1.27820, 0.00068, 12.0, 1.00),   # 0.00001 point * 100,000 = $1.00/point
    "USDCAD": (1.39120, 0.00058, 13.0, 0.72),   # ~$0.72/point in USD
}


class PaperMT5Adapter(BrokerAdapter):
    """
    Simulates an ECN/Raw-Spread MT5 account locally with realistic intraday volatility regimes,
    liquidity sweeps, bid/ask spreads, margin accounting, and automatic SL/TP fills.
    """

    def __init__(
        self,
        settings: AppSettings,
        profiles: dict[str, SymbolProfile],
        starting_balance: float = 10000.0,
        seed: int = 42,
    ) -> None:
        self.settings = settings
        self.profiles = profiles
        self.mapper = SymbolMapper(profiles)
        self.rng = np.random.default_rng(seed)

        self._connected = False
        self.balance = float(starting_balance)
        self.day_starting_equity = float(starting_balance)
        self.peak_equity = float(starting_balance)
        self.realized_pnl_today = 0.0
        self.leverage = 100

        self._next_ticket = 500101
        self._positions: dict[int, OpenPosition] = {}
        self.closed_history: list[dict] = []
        self._bars: dict[str, pd.DataFrame] = {}

        self._seed_synthetic_market()

    def _seed_synthetic_market(self) -> None:
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        n_bars = 500
        timestamps = [now - timedelta(minutes=5 * (n_bars - 1 - i)) for i in range(n_bars)]

        for sym, prof in self.profiles.items():
            base_mid, vol, spread_pts, _ = DEFAULT_PRICES.get(sym, (100.0, 0.5, 10.0, 1.0))
            # Create realistic multi-regime intraday series (trend + pullback + liquidity sweep)
            t = np.linspace(0, 8 * np.pi, n_bars)
            macro_wave = np.sin(t / 2.2) * (vol * 7.5) + np.cos(t * 1.1) * (vol * 3.2)
            noise = self.rng.normal(0.0, vol * 0.65, size=n_bars).cumsum() * 0.25
            closes = base_mid + macro_wave + noise

            # Inject a clean fresh 5-bar sell-side liquidity sweep + FVG reclaim on XAUUSD closed bar [-2]
            if sym == "XAUUSD":
                recent_base = float(np.mean(closes[-22:-8]))
                closes[-8:] = recent_base + np.array([0.2, 0.0, -0.3, -1.1, -0.4, 0.55, 0.95, 1.05]) * vol

            opens = np.roll(closes, 1)
            opens[0] = closes[0] - vol * 0.1
            wick_up = np.abs(self.rng.normal(0.0, vol * 0.45, size=n_bars))
            wick_dn = np.abs(self.rng.normal(0.0, vol * 0.45, size=n_bars))
            if sym == "XAUUSD":
                wick_dn[-6:] = vol * 0.08
                wick_up[-4] = 0.02
                wick_dn[-2] = 0.02

            highs = np.maximum(opens, closes) + wick_up
            lows = np.minimum(opens, closes) - wick_dn
            if sym == "XAUUSD":
                # Guarantee [-5] cleanly sweeps prior 20-bar liquidity low while keeping stop <= 2.0*ATR
                prior_20_min = float(np.min(lows[-25:-5]))
                lows[-5] = round(prior_20_min - vol * 0.15, 5)
                closes[-2] = round(lows[-5] + vol * 1.65, 5)
                opens[-2] = round(closes[-2] - vol * 0.45, 5)
                highs[-2] = round(closes[-2] + vol * 0.12, 5)
                lows[-2] = round(max(opens[-2] - vol * 0.05, highs[-4] + vol * 0.08), 5)
            volumes = self.rng.integers(180, 1450, size=n_bars)
            if sym == "XAUUSD":
                volumes[-4:-1] = 1950  # Institutional RVOL >= 1.35x on displacement reclaim
            spreads = np.clip(
                self.rng.normal(spread_pts, spread_pts * 0.12, size=n_bars),
                spread_pts * 0.6,
                spread_pts * 1.6,
            ).round(0)

            self._bars[sym] = pd.DataFrame({
                "timestamp": pd.to_datetime(timestamps, utc=True),
                "open": np.round(opens, 5),
                "high": np.round(highs, 5),
                "low": np.round(lows, 5),
                "close": np.round(closes, 5),
                "volume": volumes,
                "spread": spreads,
            })

    def step_market(self) -> None:
        """Advance live synthetic tick/bar slightly and evaluate server-side SL/TP fills."""
        now = datetime.now(timezone.utc)
        for sym, df in self._bars.items():
            _, vol, spread_pts, _ = DEFAULT_PRICES.get(sym, (100.0, 0.5, 10.0, 1.0))
            last_close = float(df["close"].iloc[-1])
            delta = float(self.rng.normal(0.02 * vol, vol * 0.18))
            new_close = round(last_close + delta, 5)
            idx = df.index[-1]
            df.loc[idx, "close"] = new_close
            df.loc[idx, "high"] = max(float(df.loc[idx, "high"]), new_close)
            df.loc[idx, "low"] = min(float(df.loc[idx, "low"]), new_close)
            df.loc[idx, "volume"] = int(df.loc[idx, "volume"]) + int(self.rng.integers(5, 35))
            df.loc[idx, "spread"] = round(
                float(np.clip(self.rng.normal(spread_pts, spread_pts * 0.08), spread_pts * 0.7, spread_pts * 1.35)),
                0,
            )
            df.loc[idx, "timestamp"] = pd.Timestamp(now)

        self._update_open_positions()

    def _update_open_positions(self) -> None:
        to_close: list[tuple[int, str]] = []
        for ticket, pos in self._positions.items():
            q = self.get_quote(pos.symbol)
            mark = q.bid if pos.side == Side.LONG else q.ask
            pos.current_price = mark
            price_diff = (mark - pos.entry_price) if pos.side == Side.LONG else (pos.entry_price - mark)
            points_moved = price_diff / q.point
            pos.unrealized_pnl = round(points_moved * q.tick_value * pos.volume, 2)

            if pos.side == Side.LONG:
                if mark <= pos.sl:
                    to_close.append((ticket, "SL_HIT"))
                elif mark >= pos.tp:
                    to_close.append((ticket, "TP_HIT"))
            else:
                if mark >= pos.sl:
                    to_close.append((ticket, "SL_HIT"))
                elif mark <= pos.tp:
                    to_close.append((ticket, "TP_HIT"))

        for ticket, reason in to_close:
            self.close_position(ticket, reason=reason)

    def connect(self) -> bool:
        self._connected = True
        simulated_broker_syms = [f"{sym}m" for sym in self.profiles]
        self.mapper.discover(simulated_broker_syms)
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def list_available_symbols(self) -> list[str]:
        return [f"{sym}m" for sym in self.profiles]

    def get_account_snapshot(self) -> AccountSnapshot:
        self._update_open_positions()
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        equity = round(self.balance + unrealized, 2)
        self.peak_equity = max(self.peak_equity, equity)

        used_margin = 0.0
        for p in self._positions.values():
            prof = self.profiles[p.symbol]
            notional = p.entry_price * prof.contract_size * p.volume
            used_margin += notional / self.leverage
        used_margin = round(used_margin, 2)

        return AccountSnapshot(
            timestamp=datetime.now(timezone.utc),
            login=100001,
            server="Local-Paper-MT5-Sim",
            currency="USD",
            balance=round(self.balance, 2),
            equity=equity,
            margin=used_margin,
            free_margin=round(max(0.0, equity - used_margin), 2),
            leverage=self.leverage,
            day_starting_equity=self.day_starting_equity,
            peak_equity=self.peak_equity,
            realized_pnl_today=round(self.realized_pnl_today, 2),
        )

    def get_quote(self, canonical_symbol: str) -> QuoteTick:
        prof = self.profiles[canonical_symbol]
        df = self._bars[canonical_symbol]
        mid = float(df["close"].iloc[-1])
        spread_pts = float(df["spread"].iloc[-1])
        half_spread = (spread_pts * prof.point) / 2.0
        _, _, _, tick_val = DEFAULT_PRICES.get(canonical_symbol, (100.0, 0.5, 10.0, 1.0))

        return QuoteTick(
            symbol=canonical_symbol,
            broker_symbol=self.mapper.to_broker(canonical_symbol),
            timestamp=datetime.now(timezone.utc),
            bid=round(mid - half_spread, 5),
            ask=round(mid + half_spread, 5),
            point=prof.point,
            tick_value=tick_val,
            tick_size=prof.point,
            volume_min=0.01,
            volume_max=50.0,
            volume_step=0.01,
            stops_level_points=5,
        )

    def get_bars(self, canonical_symbol: str, timeframe: str = "M5", count: int = 300) -> pd.DataFrame:
        base_df = self._bars[canonical_symbol].copy()
        tf = timeframe.upper()
        if tf == "M5":
            return base_df.tail(count).reset_index(drop=True)

        rule_map = {"M1": "1min", "M15": "15min", "H1": "1h", "H4": "4h"}
        rule = rule_map.get(tf, "15min")
        resampled = (
            base_df.set_index("timestamp")
            .resample(rule)
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "spread": "mean",
            })
            .dropna()
            .reset_index()
        )
        return resampled.tail(count).reset_index(drop=True)

    def get_open_positions(self) -> list[OpenPosition]:
        self._update_open_positions()
        return list(self._positions.values())

    def place_market_order(
        self,
        canonical_symbol: str,
        side: Side,
        volume: float,
        sl: float,
        tp: float,
        comment: str = "TA001",
        initial_risk_usd: float = 0.0,
        setup_type: str = "AGENT",
    ) -> ExecutionReceipt:
        if sl <= 0 or tp <= 0:
            return ExecutionReceipt(
                success=False,
                symbol=canonical_symbol,
                side=side,
                volume=volume,
                fill_price=0.0,
                sl=sl,
                tp=tp,
                message="HARD_SAFETY_REJECT: Server-side SL and TP are mandatory.",
            )

        q = self.get_quote(canonical_symbol)
        fill = q.ask if side == Side.LONG else q.bid
        ticket = self._next_ticket
        self._next_ticket += 1

        pos = OpenPosition(
            ticket=ticket,
            symbol=canonical_symbol,
            broker_symbol=q.broker_symbol,
            side=side,
            volume=round(volume, 2),
            entry_price=fill,
            current_price=fill,
            sl=sl,
            tp=tp,
            initial_risk_usd=initial_risk_usd,
            unrealized_pnl=0.0,
            opened_at=datetime.now(timezone.utc),
            setup_type=setup_type,
            magic=self.settings.mt5_magic_number,
        )
        self._positions[ticket] = pos
        return ExecutionReceipt(
            success=True,
            ticket=ticket,
            symbol=canonical_symbol,
            side=side,
            volume=pos.volume,
            fill_price=fill,
            sl=sl,
            tp=tp,
            message="PAPER_MT5_FILLED",
        )

    def modify_position_sl_tp(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        pos = self._positions.get(ticket)
        if not pos:
            return False
        pos.sl = new_sl
        pos.tp = new_tp
        return True

    def close_position(self, ticket: int, volume: float | None = None, reason: str = "MANUAL") -> ExecutionReceipt:
        pos = self._positions.get(ticket)
        if not pos:
            return ExecutionReceipt(
                success=False,
                ticket=ticket,
                symbol="UNKNOWN",
                side=Side.LONG,
                volume=0.0,
                fill_price=0.0,
                sl=0.0,
                tp=0.0,
                message=f"Ticket {ticket} not found",
            )
        q = self.get_quote(pos.symbol)
        exit_price = q.bid if pos.side == Side.LONG else q.ask
        close_vol = round(min(pos.volume, volume) if volume else pos.volume, 2)

        price_diff = (exit_price - pos.entry_price) if pos.side == Side.LONG else (pos.entry_price - exit_price)
        pnl = round((price_diff / q.point) * q.tick_value * close_vol, 2)

        self.balance = round(self.balance + pnl, 2)
        self.realized_pnl_today = round(self.realized_pnl_today + pnl, 2)

        if close_vol >= pos.volume - 1e-6:
            del self._positions[ticket]
        else:
            pos.volume = round(pos.volume - close_vol, 2)

        self.closed_history.append({
            "ticket": ticket,
            "symbol": pos.symbol,
            "side": pos.side.value,
            "volume": close_vol,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        })

        return ExecutionReceipt(
            success=True,
            ticket=ticket,
            symbol=pos.symbol,
            side=pos.side,
            volume=close_vol,
            fill_price=exit_price,
            sl=pos.sl,
            tp=pos.tp,
            message=f"CLOSED_{reason} (PnL=${pnl:+.2f})",
        )

    def close_all_positions(self, reason: str = "KILL_SWITCH") -> list[ExecutionReceipt]:
        tickets = list(self._positions.keys())
        return [self.close_position(t, reason=reason) for t in tickets]
