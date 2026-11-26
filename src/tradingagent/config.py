"""Configuration & Symbol Registry for TradingAgent-001."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class SymbolProfile(BaseModel):
    canonical: str
    tier: int
    aliases: list[str]
    asset_class: Literal["metal", "index", "crypto", "forex"]
    usd_direction_sign: int = Field(
        description="-1 if Long = Short USD (e.g. XAUUSD, EURUSD), +1 if Long = Long USD (USDCAD), 0 if neutral"
    )
    contract_size: float
    point: float
    pip_size: float
    max_spread_points: float
    max_spread_to_sl_ratio: float = 0.12
    prime_sessions_utc: list[tuple[int, int]]

    def is_in_prime_session(self, hour_utc: int) -> bool:
        for start_h, end_h in self.prime_sessions_utc:
            if start_h <= hour_utc < end_h:
                return True
        return False

    def canonical_spread_points(self, spread_price: float) -> float:
        """Normalizes broker spread (in price units) into canonical profile points regardless of 2 vs 3 decimal MT5 quotes."""
        if self.point <= 0:
            return 0.0
        return max(0.0, spread_price) / self.point


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8-sig",
        extra="ignore",
    )

    trading_mode: Literal["paper", "mt5_demo", "mt5_live"] = "paper"

    # MT5 Native Windows IPC settings
    mt5_terminal_path: str = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    mt5_login: int | None = None
    mt5_password: str | None = None
    mt5_server: str | None = None
    mt5_magic_number: int = 2026001

    @field_validator("mt5_login", mode="before")
    @classmethod
    def _empty_str_to_none_int(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("mt5_password", "mt5_server", mode="before")
    @classmethod
    def _empty_str_to_none_str(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    # Zero-Fee Local Ollama / OpenAI-compatible LLM settings (tuned for 16GB RAM Windows 10)
    llm_enabled: bool = True
    llm_provider: Literal["ollama", "openai_compat", "heuristic"] = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_quick_model: str = "qwen3:4b"
    ollama_deep_model: str = "qwen3:8b"
    llm_timeout_seconds: float = 25.0

    openai_compat_base_url: str = ""
    openai_compat_api_key: str = ""

    # Deterministic Risk Firewall limits
    risk_per_trade_pct: float = 0.75
    max_min_lot_risk_pct: float = 1.25
    gold_min_lot_max_risk_pct: float = 3.25
    max_daily_loss_pct: float = 2.50
    max_total_drawdown_pct: float = 6.00
    max_open_positions: int = 2
    gold_reserved_extra_slot: bool = True
    max_gold_positions: int = 1
    max_net_usd_risk_pct: float = 1.25
    min_risk_reward_ratio: float = 1.80
    cooldown_minutes_per_symbol: int = 15
    loss_cooldown_minutes: int = 45

    # 3-Stage Ratchet & Chandelier ATR Trailing Stop Parameters
    be_trigger_r: float = 1.15
    be_lock_r: float = 0.20
    chandelier_start_r: float = 1.55
    chandelier_atr_mult: float = 1.65

    # Local Persistence Paths
    sqlite_path: Path = ROOT_DIR / "data" / "state" / "tradingagent.sqlite3"
    duckdb_path: Path = ROOT_DIR / "data" / "catalog" / "market_data.duckdb"
    symbols_config_path: Path = ROOT_DIR / "config" / "symbols.yaml"

    # Dashboard & Built-In Autonomous Loop
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000
    autopilot_enabled: bool = False
    autopilot_interval_seconds: int = 15


def load_symbol_profiles(config_path: Path | None = None) -> dict[str, SymbolProfile]:
    path = config_path or (ROOT_DIR / "config" / "symbols.yaml")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    profiles: dict[str, SymbolProfile] = {}
    tier_map = {
        "tier_1_primary": 1,
        "tier_2_high_beta": 2,
        "tier_3_fx_majors": 3,
    }
    for key, tier_num in tier_map.items():
        for entry in raw.get("priority_tiers", {}).get(key, []):
            prof = SymbolProfile(
                canonical=entry["canonical"],
                tier=tier_num,
                aliases=entry["aliases"],
                asset_class=entry["asset_class"],
                usd_direction_sign=entry["usd_direction_sign"],
                contract_size=float(entry["contract_size"]),
                point=float(entry["point"]),
                pip_size=float(entry["pip_size"]),
                max_spread_points=float(entry["max_spread_points"]),
                max_spread_to_sl_ratio=float(entry.get("max_spread_to_sl_ratio", 0.12)),
                prime_sessions_utc=[tuple(x) for x in entry["prime_sessions_utc"]],
            )
            profiles[prof.canonical] = prof
    return profiles
