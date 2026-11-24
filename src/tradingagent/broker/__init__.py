"""Broker Adapter Protocol & Broker Symbol Auto-Mapper."""

from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd

from tradingagent.config import SymbolProfile
from tradingagent.models import (
    AccountSnapshot,
    ExecutionReceipt,
    OpenPosition,
    QuoteTick,
    Side,
)


class SymbolMapper:
    """Maps canonical names (XAUUSD, NAS100, etc.) to broker-specific symbols (XAUUSD.m, US100.std, USTEC, etc.)."""

    _ALLOWED_SUFFIXES = {
        "m", "i", "c", "a", "b", "r", "pro", "raw", "std", "ecn", "cash", "micro", "mini", "zero", "stp"
    }

    def __init__(self, profiles: dict[str, SymbolProfile]) -> None:
        self.profiles = profiles
        self.canonical_to_broker: dict[str, str] = {}
        self.broker_to_canonical: dict[str, str] = {}

    @classmethod
    def _has_valid_broker_suffix(cls, broker_sym: str, base_alias: str) -> bool:
        b_up = broker_sym.upper()
        a_up = base_alias.upper()
        if not b_up.startswith(a_up):
            return False
        rem = broker_sym[len(base_alias):]
        if not rem:
            return True
        if rem[0] in "._-#":
            return True
        return rem.lower() in cls._ALLOWED_SUFFIXES

    def discover(self, available_broker_symbols: list[str]) -> dict[str, str]:
        available_set = set(available_broker_symbols)
        lower_map = {s.lower(): s for s in available_broker_symbols}

        for canonical, prof in self.profiles.items():
            matched = None
            candidates = [canonical] + [a for a in prof.aliases if a != canonical]

            # 1. Exact alias match
            for alias in candidates:
                if alias in available_set:
                    matched = alias
                    break
            # 2. Case-insensitive alias match
            if not matched:
                for alias in candidates:
                    if alias.lower() in lower_map:
                        matched = lower_map[alias.lower()]
                        break
            # 3. Safe broker suffix match across canonical + all aliases (e.g. XAUUSD.m, US100.std, USTEC.pro)
            if not matched:
                for alias in candidates:
                    for b_sym in available_broker_symbols:
                        if self._has_valid_broker_suffix(b_sym, alias):
                            matched = b_sym
                            break
                    if matched:
                        break

            if matched:
                self.canonical_to_broker[canonical] = matched
                self.broker_to_canonical[matched] = canonical

        return dict(self.canonical_to_broker)

    def is_mapped(self, canonical: str) -> bool:
        return canonical in self.canonical_to_broker

    def to_broker(self, canonical: str) -> str:
        return self.canonical_to_broker.get(canonical, canonical)

    def to_canonical(self, broker_symbol: str) -> str:
        return self.broker_to_canonical.get(broker_symbol, broker_symbol)


class BrokerAdapter(ABC):
    """Unified interface for Native Windows MetaTrader5 IPC and Local Paper Simulator."""

    @abstractmethod
    def connect(self) -> bool:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        ...

    @abstractmethod
    def list_available_symbols(self) -> list[str]:
        ...

    @abstractmethod
    def get_account_snapshot(self) -> AccountSnapshot:
        ...

    @abstractmethod
    def get_quote(self, canonical_symbol: str) -> QuoteTick:
        ...

    @abstractmethod
    def get_bars(self, canonical_symbol: str, timeframe: str = "M5", count: int = 300) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_open_positions(self) -> list[OpenPosition]:
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def modify_position_sl_tp(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        ...

    @abstractmethod
    def close_position(self, ticket: int, volume: float | None = None, reason: str = "MANUAL") -> ExecutionReceipt:
        ...

    @abstractmethod
    def close_all_positions(self, reason: str = "KILL_SWITCH") -> list[ExecutionReceipt]:
        ...
