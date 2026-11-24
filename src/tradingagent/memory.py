"""SQLite Persistence & TradingAgents-Inspired Settled Reflection Memory Log."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from tradingagent.models import CommitteeDecision, ExecutionReceipt, RiskCheckResult


class StateAndMemoryStore:
    """
    Local zero-fee SQLite store holding:
      1. Full audit log of every candidate setup, committee vote, risk check, and MT5 execution receipt
      2. Settled Decision Reflection Memory (inspired by TauricResearch/TradingAgents):
         Records outcome of each trade (+R / -R, regime, session, setup type) and generates concise
         symbol-specific lessons injected into future Head-Trader prompts so the agent learns locally!
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS decision_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setup_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    setup_type TEXT NOT NULL,
                    quant_score REAL NOT NULL,
                    agent_action TEXT NOT NULL,
                    agent_grade TEXT NOT NULL,
                    risk_approved INTEGER NOT NULL,
                    ticket INTEGER,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reflection_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    setup_type TEXT NOT NULL,
                    session_label TEXT NOT NULL,
                    outcome_r REAL NOT NULL,
                    lesson_text TEXT NOT NULL
                );
                """
            )
            # Seed baseline institutional lessons if table is empty
            cur = conn.execute("SELECT COUNT(*) AS cnt FROM reflection_memory")
            if int(cur.fetchone()["cnt"]) == 0:
                seed_lessons = [
                    (
                        datetime.now(timezone.utc).isoformat(),
                        "XAUUSD",
                        "LIQUIDITY_SWEEP_FVG",
                        "NY_OVERLAP_PRIME",
                        2.3,
                        "Gold liquidity sweeps below Asian/London lows with M5 FVG reclaim perform best when spread < 22 pts and H1 EMA20 slopes up.",
                    ),
                    (
                        datetime.now(timezone.utc).isoformat(),
                        "XAUUSD",
                        "SESSION_ORB_BREAKOUT",
                        "ASIAN_OFF_PEAK",
                        -1.0,
                        "Avoid chasing raw M5 range breakouts on Gold during low-ADX off-peak hours without a prior liquidity sweep.",
                    ),
                    (
                        datetime.now(timezone.utc).isoformat(),
                        "NAS100",
                        "LIQUIDITY_SWEEP_FVG",
                        "NY_OVERLAP_PRIME",
                        2.1,
                        "NAS100 NY open judas-swing sweep followed by displacement FVG offers clean +2R continuation.",
                    ),
                ]
                conn.executemany(
                    """
                    INSERT INTO reflection_memory
                    (created_at, symbol, setup_type, session_label, outcome_r, lesson_text)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    seed_lessons,
                )

    def record_audit_cycle(
        self,
        setup_dict: dict[str, Any],
        decision: CommitteeDecision,
        risk: RiskCheckResult | None,
        receipt: ExecutionReceipt | None,
    ) -> None:
        payload = {
            "setup": setup_dict,
            "decision": decision.model_dump(mode="json"),
            "risk": risk.model_dump(mode="json") if risk else None,
            "receipt": receipt.model_dump(mode="json") if receipt else None,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decision_audit
                (setup_id, timestamp, symbol, side, setup_type, quant_score, agent_action, agent_grade, risk_approved, ticket, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    setup_dict["setup_id"],
                    datetime.now(timezone.utc).isoformat(),
                    setup_dict["symbol"],
                    setup_dict["side"],
                    setup_dict["setup_type"],
                    float(setup_dict["quant_score"]),
                    decision.action.value,
                    decision.grade.value,
                    1 if (risk and risk.approved) else 0,
                    receipt.ticket if (receipt and receipt.success) else None,
                    json.dumps(payload),
                ),
            )

    def add_settled_lesson(
        self,
        symbol: str,
        setup_type: str,
        session_label: str,
        outcome_r: float,
        lesson_text: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reflection_memory
                (created_at, symbol, setup_type, session_label, outcome_r, lesson_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    symbol,
                    setup_type,
                    session_label,
                    round(outcome_r, 2),
                    lesson_text,
                ),
            )

    def get_lessons_for_symbol(self, symbol: str, limit: int = 4) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT lesson_text, outcome_r, session_label
                FROM reflection_memory
                WHERE symbol = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
        return [f"[{r['session_label']} | {r['outcome_r']:+.1f}R] {r['lesson_text']}" for r in rows]

    def get_recent_audits(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json FROM decision_audit
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]
