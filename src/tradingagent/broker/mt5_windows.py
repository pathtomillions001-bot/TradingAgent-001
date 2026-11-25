"""Windows 10 Native MetaTrader 5 IPC Adapter (zero-fee official MetaTrader5 package)."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
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

logger = logging.getLogger(__name__)


class MT5WindowsAdapter(BrokerAdapter):
    """
    Connects directly to the local Windows 10 MetaTrader 5 terminal via Named Pipes (<2ms latency).
    Every order ALWAYS includes broker server-side hard SL and TP so account risk is bounded even if
    the laptop loses Wi-Fi or battery.
    """

    def __init__(self, settings: AppSettings, profiles: dict[str, SymbolProfile]) -> None:
        self.settings = settings
        self.profiles = profiles
        self.mapper = SymbolMapper(profiles)
        self._mt5 = None
        self._connected = False
        self._day_start_equity: float | None = None
        self._peak_equity: float = 0.0
        self._day_ordinal: int = -1

    def _import_mt5(self):
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5  # type: ignore
                self._mt5 = mt5
            except ImportError as exc:
                raise RuntimeError(
                    "MetaTrader5 package is only available on Windows. "
                    "On your Windows 10 laptop, run: pip install MetaTrader5"
                ) from exc
        return self._mt5

    def connect(self) -> bool:
        mt5 = self._import_mt5()
        init_kwargs: dict = {}
        if self.settings.mt5_terminal_path:
            init_kwargs["path"] = self.settings.mt5_terminal_path
        if self.settings.mt5_login:
            init_kwargs["login"] = int(self.settings.mt5_login)
        if self.settings.mt5_password:
            init_kwargs["password"] = self.settings.mt5_password
        if self.settings.mt5_server:
            init_kwargs["server"] = self.settings.mt5_server

        ok = mt5.initialize(**init_kwargs) if init_kwargs else mt5.initialize()
        if not ok:
            logger.error("MT5 initialize() failed: %s", mt5.last_error())
            self._connected = False
            return False

        self._connected = True
        symbols = self.list_available_symbols()
        mapped = self.mapper.discover(symbols)
        for canonical, broker_sym in mapped.items():
            mt5.symbol_select(broker_sym, True)
        logger.info("Connected to MT5 terminal. Mapped symbols: %s", mapped)
        return True

    def disconnect(self) -> None:
        if self._mt5 is not None and self._connected:
            self._mt5.shutdown()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def list_available_symbols(self) -> list[str]:
        mt5 = self._import_mt5()
        raw = mt5.symbols_get()
        if not raw:
            return []
        return [s.name for s in raw]

    def get_account_snapshot(self) -> AccountSnapshot:
        mt5 = self._import_mt5()
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"Failed to read MT5 account_info: {mt5.last_error()}")

        now = datetime.now(timezone.utc)
        equity = float(info.equity)
        if self._day_ordinal != now.toordinal() or self._day_start_equity is None:
            self._day_ordinal = now.toordinal()
            self._day_start_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

        return AccountSnapshot(
            timestamp=now,
            login=int(info.login),
            server=str(info.server),
            currency=str(info.currency),
            balance=float(info.balance),
            equity=equity,
            margin=float(info.margin),
            free_margin=float(info.margin_free),
            leverage=int(info.leverage),
            day_starting_equity=self._day_start_equity,
            peak_equity=self._peak_equity,
            realized_pnl_today=round(equity - self._day_start_equity, 2),
        )

    def get_quote(self, canonical_symbol: str) -> QuoteTick:
        mt5 = self._import_mt5()
        broker_sym = self.mapper.to_broker(canonical_symbol)
        mt5.symbol_select(broker_sym, True)
        tick = mt5.symbol_info_tick(broker_sym)
        info = mt5.symbol_info(broker_sym)
        if tick is None or info is None:
            raise RuntimeError(f"Failed to fetch MT5 tick/info for {broker_sym}: {mt5.last_error()}")

        point = float(info.point)
        tick_size = float(info.trade_tick_size) if info.trade_tick_size > 0 else point
        tick_val = float(info.trade_tick_value)
        # Normalize tick_value to 1 point move for 1.0 lot
        point_val = tick_val * (point / tick_size) if tick_size > 0 else tick_val

        return QuoteTick(
            symbol=canonical_symbol,
            broker_symbol=broker_sym,
            timestamp=datetime.fromtimestamp(int(tick.time), tz=timezone.utc),
            bid=float(tick.bid),
            ask=float(tick.ask),
            point=point,
            tick_value=point_val,
            tick_size=tick_size,
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            stops_level_points=int(info.trade_stops_level),
        )

    def get_bars(self, canonical_symbol: str, timeframe: str = "M5", count: int = 300) -> pd.DataFrame:
        mt5 = self._import_mt5()
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
        }
        mt5_tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M5)
        broker_sym = self.mapper.to_broker(canonical_symbol)
        rates = mt5.copy_rates_from_pos(broker_sym, mt5_tf, 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No bars returned for {broker_sym} ({timeframe}): {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["timestamp", "open", "high", "low", "close", "volume", "spread"]].reset_index(drop=True)

    def get_open_positions(self) -> list[OpenPosition]:
        mt5 = self._import_mt5()
        positions = mt5.positions_get()
        if not positions:
            return []
        result: list[OpenPosition] = []
        for p in positions:
            if int(p.magic) != int(self.settings.mt5_magic_number):
                continue
            canonical = self.mapper.to_canonical(str(p.symbol))
            side = Side.LONG if int(p.type) == mt5.ORDER_TYPE_BUY else Side.SHORT
            result.append(
                OpenPosition(
                    ticket=int(p.ticket),
                    symbol=canonical,
                    broker_symbol=str(p.symbol),
                    side=side,
                    volume=float(p.volume),
                    entry_price=float(p.price_open),
                    current_price=float(p.price_current),
                    sl=float(p.sl),
                    tp=float(p.tp),
                    initial_risk_usd=0.0,
                    unrealized_pnl=float(p.profit),
                    opened_at=datetime.fromtimestamp(int(p.time), tz=timezone.utc),
                    magic=int(p.magic),
                )
            )
        return result

    def _normalize_price(self, broker_sym: str, price: float) -> float:
        """Aligns price/SL/TP to broker symbol digits and trade_tick_size so MT5 order_check never rejects rounding."""
        mt5 = self._import_mt5()
        info = mt5.symbol_info(broker_sym)
        if info is None or price <= 0:
            return round(float(price), 5)
        tick_size = float(info.trade_tick_size) if info.trade_tick_size > 0 else float(info.point)
        digits = int(info.digits)
        if tick_size > 0:
            snapped = round(float(price) / tick_size) * tick_size
            return round(snapped, digits)
        return round(float(price), digits)

    def _resolve_filling_mode(self, broker_sym: str) -> int:
        mt5 = self._import_mt5()
        info = mt5.symbol_info(broker_sym)
        if info is None:
            return mt5.ORDER_FILLING_IOC
        filling = int(info.filling_mode)
        if filling & 1:
            return mt5.ORDER_FILLING_FOK
        if filling & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

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
        mt5 = self._import_mt5()
        if sl <= 0 or tp <= 0:
            return ExecutionReceipt(
                success=False,
                symbol=canonical_symbol,
                side=side,
                volume=volume,
                fill_price=0.0,
                sl=sl,
                tp=tp,
                message="HARD_SAFETY_REJECT: Server-side SL and TP are mandatory on every MT5 order.",
            )

        broker_sym = self.mapper.to_broker(canonical_symbol)
        quote = self.get_quote(canonical_symbol)
        order_type = mt5.ORDER_TYPE_BUY if side == Side.LONG else mt5.ORDER_TYPE_SELL
        price = self._normalize_price(broker_sym, quote.ask if side == Side.LONG else quote.bid)
        norm_sl = self._normalize_price(broker_sym, sl)
        norm_tp = self._normalize_price(broker_sym, tp)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_sym,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "sl": float(norm_sl),
            "tp": float(norm_tp),
            "deviation": 20,
            "magic": int(self.settings.mt5_magic_number),
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._resolve_filling_mode(broker_sym),
        }

        # Preflight order_check before sending real/demo order
        check = mt5.order_check(request)
        if check is None or int(check.retcode) != 0:
            code = getattr(check, "retcode", -1)
            cmt = getattr(check, "comment", str(mt5.last_error()))
            return ExecutionReceipt(
                success=False,
                symbol=canonical_symbol,
                side=side,
                volume=volume,
                fill_price=price,
                sl=norm_sl,
                tp=norm_tp,
                message=f"MT5 order_check rejected (retcode={code}): {cmt}",
            )

        res = mt5.order_send(request)
        if res is None or int(res.retcode) != mt5.TRADE_RETCODE_DONE:
            code = getattr(res, "retcode", -1)
            cmt = getattr(res, "comment", str(mt5.last_error()))
            return ExecutionReceipt(
                success=False,
                symbol=canonical_symbol,
                side=side,
                volume=volume,
                fill_price=price,
                sl=norm_sl,
                tp=norm_tp,
                message=f"MT5 order_send failed (retcode={code}): {cmt}",
            )

        return ExecutionReceipt(
            success=True,
            ticket=int(res.order),
            symbol=canonical_symbol,
            side=side,
            volume=float(res.volume),
            fill_price=float(res.price),
            sl=norm_sl,
            tp=norm_tp,
            message="MT5_FILLED",
        )

    def modify_position_sl_tp(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        mt5 = self._import_mt5()
        pos_list = mt5.positions_get(ticket=int(ticket))
        if not pos_list:
            return False
        pos = pos_list[0]
        broker_sym = str(pos.symbol)
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol": broker_sym,
            "sl": self._normalize_price(broker_sym, new_sl),
            "tp": self._normalize_price(broker_sym, new_tp),
            "magic": int(self.settings.mt5_magic_number),
        }
        res = mt5.order_send(request)
        return bool(res is not None and int(res.retcode) == mt5.TRADE_RETCODE_DONE)

    def close_position(self, ticket: int, volume: float | None = None, reason: str = "MANUAL") -> ExecutionReceipt:
        mt5 = self._import_mt5()
        pos_list = mt5.positions_get(ticket=int(ticket))
        if not pos_list:
            return ExecutionReceipt(
                success=False,
                ticket=ticket,
                symbol="UNKNOWN",
                side=Side.LONG,
                volume=0.0,
                fill_price=0.0,
                sl=0.0,
                tp=0.0,
                message=f"Position {ticket} not found",
            )
        pos = pos_list[0]
        broker_sym = str(pos.symbol)
        canonical = self.mapper.to_canonical(broker_sym)
        close_vol = float(volume) if volume else float(pos.volume)
        is_buy_pos = int(pos.type) == mt5.ORDER_TYPE_BUY
        close_type = mt5.ORDER_TYPE_SELL if is_buy_pos else mt5.ORDER_TYPE_BUY
        quote = self.get_quote(canonical)
        close_price = quote.bid if is_buy_pos else quote.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": int(ticket),
            "symbol": broker_sym,
            "volume": close_vol,
            "type": close_type,
            "price": float(close_price),
            "deviation": 25,
            "magic": int(self.settings.mt5_magic_number),
            "comment": f"CLS_{reason}"[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._resolve_filling_mode(broker_sym),
        }
        res = mt5.order_send(request)
        ok = bool(res is not None and int(res.retcode) == mt5.TRADE_RETCODE_DONE)
        return ExecutionReceipt(
            success=ok,
            ticket=ticket,
            symbol=canonical,
            side=Side.LONG if is_buy_pos else Side.SHORT,
            volume=close_vol,
            fill_price=float(getattr(res, "price", close_price)),
            sl=float(pos.sl),
            tp=float(pos.tp),
            message="CLOSED" if ok else f"Close failed: {getattr(res, 'comment', '')}",
        )

    def close_all_positions(self, reason: str = "KILL_SWITCH") -> list[ExecutionReceipt]:
        receipts: list[ExecutionReceipt] = []
        for pos in self.get_open_positions():
            receipts.append(self.close_position(pos.ticket, reason=reason))
        return receipts
