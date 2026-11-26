"""Layer 3: Deterministic Pre-Trade Risk Firewall & Intraday Position Manager (Zero-Trust Python)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

from tradingagent.broker import BrokerAdapter
from tradingagent.config import AppSettings, SymbolProfile
from tradingagent.indicators import compute_atr
from tradingagent.models import (
    AccountSnapshot,
    CandidateSetup,
    OpenPosition,
    QuoteTick,
    RiskCheckResult,
    Side,
)


class DeterministicRiskFirewall:
    """
    100% pure tested Python risk engine inspired by jkspeaks/deterministic-trader & FinRobot V2.
    No LLM prompt can bypass or override any check in this module:
      1. Kill-switch status
      2. Daily Drawdown Lock (e.g. 2.50%) & Total Drawdown Lock (e.g. 6.00%)
      3. Max concurrent open positions & Max Gold positions
      4. Per-symbol anti-overtrading cooldown timer
      5. Live spread ceiling & spread-to-SL ratio
      6. Broker stops_level enforcement
      7. Net R:R after deducting live spread >= min_risk_reward_ratio
      8. Net USD Directional Exposure Cap across correlated pairs (XAUUSD, EURUSD, GBPUSD, USDCAD, BTCUSD)
      9. Exact lot sizing from broker tick_value, point, volume_step, and free margin check
    """

    def __init__(self, settings: AppSettings, profiles: dict[str, SymbolProfile]) -> None:
        self.settings = settings
        self.profiles = profiles
        self.kill_switch_active: bool = False
        self._last_trade_at: dict[str, datetime] = {}
        self._last_sl_at: dict[str, datetime] = {}

    def record_trade_timestamp(self, symbol: str, ts: datetime | None = None) -> None:
        self._last_trade_at[symbol] = ts or datetime.now(timezone.utc)

    def record_stop_loss_timestamp(self, symbol: str, ts: datetime | None = None) -> None:
        self._last_sl_at[symbol] = ts or datetime.now(timezone.utc)

    def compute_open_net_usd_risk_pct(
        self,
        open_positions: list[OpenPosition],
        equity: float,
    ) -> float:
        """
        Signed USD directional risk % currently on the book:
          Positive = Net Long USD, Negative = Net Short USD.
          Example: Long XAUUSD (usd_sign=-1, LONG=+1 => -1 * 0.75% = -0.75% Short USD).
                   Long EURUSD (usd_sign=-1, LONG=+1 => -0.75% Short USD).
                   Combined = -1.50% Short USD!
        """
        if equity <= 0:
            return 0.0
        net_usd_pct = 0.0
        for p in open_positions:
            prof = self.profiles.get(p.symbol)
            if not prof or prof.usd_direction_sign == 0:
                continue
            side_sign = 1 if p.side == Side.LONG else -1
            # Effective risk remaining (0 if SL already moved to break-even or better)
            if p.be_moved:
                pos_risk_pct = 0.0
            elif p.initial_risk_usd > 0:
                pos_risk_pct = (p.initial_risk_usd / equity) * 100.0
            else:
                pos_risk_pct = self.settings.risk_per_trade_pct
            net_usd_pct += prof.usd_direction_sign * side_sign * pos_risk_pct
        return round(net_usd_pct, 4)

    def effective_min_lot_risk_cap_pct(self, symbol: str) -> float:
        """
        Tier-1 Gold (XAUUSD) has a larger contract notional (100 oz -> $1/pt on 0.01 lot) than FX micro-lots.
        Allows 0.01 min lot on XAUUSD up to gold_min_lot_max_risk_pct while keeping FX/secondary pairs tight.
        """
        if symbol.upper() == "XAUUSD":
            return max(self.settings.max_min_lot_risk_pct, self.settings.gold_min_lot_max_risk_pct)
        return self.settings.max_min_lot_risk_pct

    def compute_contract_unit_telemetry(self, prof: SymbolProfile, quote: QuoteTick) -> dict[str, float | str]:
        """Returns live broker-authoritative $/pip (or $/pt) at 0.01 min lot and 1.00 full lot."""
        if quote.point <= 0 or quote.tick_value <= 0:
            return {"min_lot": quote.volume_min, "usd_per_pip_min_lot": 0.0, "unit_label": "—"}
        pips_in_points = prof.pip_size / quote.point
        usd_per_pip_1lot = pips_in_points * quote.tick_value
        usd_per_pip_min_lot = round(usd_per_pip_1lot * quote.volume_min, 3)
        unit_name = "pip" if prof.asset_class in ("forex", "metal") else "pt"
        return {
            "min_lot": quote.volume_min,
            "usd_per_pip_min_lot": usd_per_pip_min_lot,
            "usd_per_pip_1lot": round(usd_per_pip_1lot, 2),
            "unit_label": f"${usd_per_pip_min_lot:.2f}/{unit_name} @{quote.volume_min:.2f}",
        }

    def calculate_exact_lots(
        self,
        equity: float,
        risk_pct: float,
        sl_distance_price: float,
        quote: QuoteTick,
    ) -> tuple[float, float]:
        """
        Returns (rounded_lots, exact_risk_usd).
        Uses broker's authoritative point & tick_value (value per 1 point move per 1.0 lot),
        automatically adapting lot sizes to each asset's contract specification (e.g. XAUUSD vs EURUSD vs NAS100).
        """
        if equity <= 0 or risk_pct <= 0 or sl_distance_price <= 0 or quote.point <= 0 or quote.tick_value <= 0:
            return 0.0, 0.0

        target_risk_usd = equity * (risk_pct / 100.0)
        sl_points = sl_distance_price / quote.point
        loss_per_one_lot = sl_points * quote.tick_value
        if loss_per_one_lot <= 0:
            return 0.0, 0.0

        raw_lots = target_risk_usd / loss_per_one_lot
        step = quote.volume_step if quote.volume_step > 0 else 0.01
        # Floor to volume_step so we NEVER exceed target_risk_usd on standard sizing
        floored_lots = math.floor(raw_lots / step) * step
        decimals = max(0, int(round(-math.log10(step)))) if step < 1 else 0
        lots = round(floored_lots, decimals)

        if lots < quote.volume_min:
            min_lot_risk_usd = round(quote.volume_min * loss_per_one_lot, 2)
            min_lot_risk_pct = (min_lot_risk_usd / equity) * 100.0
            cap_pct = self.effective_min_lot_risk_cap_pct(quote.symbol)
            # Allow broker minimum lot (0.01) ONLY if its exact dollar risk <= symbol's min-lot cap
            if min_lot_risk_pct <= cap_pct:
                return round(quote.volume_min, decimals), min_lot_risk_usd
            return 0.0, min_lot_risk_usd

        lots = min(lots, quote.volume_max)
        actual_risk_usd = round(lots * loss_per_one_lot, 2)
        return lots, actual_risk_usd

    def validate_and_size(
        self,
        setup: CandidateSetup,
        account: AccountSnapshot,
        quote: QuoteTick,
        open_positions: list[OpenPosition],
        now_utc: datetime | None = None,
    ) -> RiskCheckResult:
        now = now_utc or datetime.now(timezone.utc)
        prof = self.profiles[setup.symbol]
        rejections: list[str] = []
        passed: list[str] = []

        # 1. Emergency Kill Switch
        if self.kill_switch_active:
            rejections.append("KILL_SWITCH_ACTIVE: Trading halted by operator or max drawdown lock.")
        else:
            passed.append("Kill-switch inactive")

        # 2. Daily & Total Drawdown Firewall
        if account.daily_drawdown_pct >= self.settings.max_daily_loss_pct:
            rejections.append(
                f"DAILY_LOSS_LOCK: Current daily DD {account.daily_drawdown_pct:.2f}% >= {self.settings.max_daily_loss_pct:.2f}%"
            )
        else:
            passed.append(f"Daily DD {account.daily_drawdown_pct:.2f}% within {self.settings.max_daily_loss_pct:.2f}% cap")

        if account.total_drawdown_pct >= self.settings.max_total_drawdown_pct:
            rejections.append(
                f"MAX_TOTAL_DD_LOCK: Total DD {account.total_drawdown_pct:.2f}% >= {self.settings.max_total_drawdown_pct:.2f}%"
            )
        else:
            passed.append(f"Total DD {account.total_drawdown_pct:.2f}% within {self.settings.max_total_drawdown_pct:.2f}% cap")

        # 3. Position Count Caps (Base cap = max_open_positions for non-Gold; +1 VIP reserved 3rd slot for XAUUSD only)
        gold_open = sum(1 for p in open_positions if p.symbol == "XAUUSD")
        effective_max_open = (
            self.settings.max_open_positions + (1 if self.settings.gold_reserved_extra_slot else 0)
            if setup.symbol == "XAUUSD"
            else self.settings.max_open_positions
        )

        if len(open_positions) >= effective_max_open:
            if setup.symbol != "XAUUSD" and self.settings.gold_reserved_extra_slot:
                rejections.append(
                    f"MAX_OPEN_POSITIONS: {len(open_positions)} open >= non-Gold limit {self.settings.max_open_positions} "
                    f"(slot #{self.settings.max_open_positions + 1} is reserved exclusively for XAUUSD)"
                )
            else:
                rejections.append(
                    f"MAX_OPEN_POSITIONS: {len(open_positions)} open >= limit {effective_max_open}"
                )
        else:
            passed.append(f"Open positions {len(open_positions)}/{effective_max_open}")

        if setup.symbol == "XAUUSD" and gold_open >= self.settings.max_gold_positions:
            rejections.append(f"MAX_GOLD_POSITIONS: {gold_open} >= {self.settings.max_gold_positions}")

        # 4. Cooldown Check (standard cooldown + extended post-SL anti-whipsaw cooldown)
        last_sl_ts = self._last_sl_at.get(setup.symbol)
        last_ts = self._last_trade_at.get(setup.symbol)
        if last_sl_ts is not None and (now - last_sl_ts) < timedelta(minutes=self.settings.loss_cooldown_minutes):
            rem = self.settings.loss_cooldown_minutes - int((now - last_sl_ts).total_seconds() // 60)
            rejections.append(f"POST_SL_COOLDOWN: {setup.symbol} anti-whipsaw lock ({rem}m remaining)")
        elif last_ts is not None and (now - last_ts) < timedelta(minutes=self.settings.cooldown_minutes_per_symbol):
            rem = self.settings.cooldown_minutes_per_symbol - int((now - last_ts).total_seconds() // 60)
            rejections.append(f"SYMBOL_COOLDOWN: {setup.symbol} cooling down ({rem}m remaining)")
        else:
            passed.append("Cooldown clear")

        # 5. Live Spread & Broker Stops-Level Check (normalized to canonical profile points)
        fill_price = quote.ask if setup.side == Side.LONG else quote.bid
        live_canonical_spread = prof.canonical_spread_points(quote.spread_price)
        if live_canonical_spread > prof.max_spread_points:
            rejections.append(
                f"SPREAD_SPIKE: Live spread {live_canonical_spread:.1f} pts > max {prof.max_spread_points:.1f} pts"
            )
        else:
            passed.append(f"Live spread {live_canonical_spread:.1f} pts <= {prof.max_spread_points:.1f}")

        sl_dist = abs(fill_price - setup.stop_loss)
        tp_dist = abs(setup.take_profit - fill_price)
        min_stop_dist = max(quote.stops_level_points * quote.point * 1.2, quote.spread_price * 2.5)
        if sl_dist < min_stop_dist:
            rejections.append(
                f"STOP_TOO_TIGHT: SL distance {sl_dist:.5f} < broker/spread minimum {min_stop_dist:.5f}"
            )

        spread_ratio = quote.spread_price / sl_dist if sl_dist > 0 else 1.0
        if spread_ratio > prof.max_spread_to_sl_ratio:
            rejections.append(
                f"SPREAD_TO_SL_HIGH: Spread consumes {spread_ratio * 100:.1f}% of SL (max {prof.max_spread_to_sl_ratio * 100:.1f}%)"
            )

        # 6. Net R:R Check After Live Entry Fill
        net_rr = tp_dist / sl_dist if sl_dist > 0 else 0.0
        if net_rr < self.settings.min_risk_reward_ratio:
            rejections.append(
                f"NET_RR_TOO_LOW: Net R:R {net_rr:.2f} < minimum {self.settings.min_risk_reward_ratio:.2f}"
            )
        else:
            passed.append(f"Net R:R {net_rr:.2f} >= {self.settings.min_risk_reward_ratio:.2f}")

        # 7. Exact Lot Sizing
        lots, actual_risk_usd = self.calculate_exact_lots(
            equity=account.equity,
            risk_pct=self.settings.risk_per_trade_pct,
            sl_distance_price=sl_dist,
            quote=quote,
        )
        sym_min_cap = self.effective_min_lot_risk_cap_pct(setup.symbol)
        if lots <= 0:
            min_pct = (actual_risk_usd / account.equity * 100.0) if account.equity > 0 else 0.0
            rejections.append(
                f"LOT_SIZE_BELOW_MIN: Min lot {quote.volume_min:.2f} risks ${actual_risk_usd:.2f} "
                f"({min_pct:.2f}% > {sym_min_cap:.2f}% cap)."
            )
            actual_risk_usd = 0.0
        else:
            passed.append(f"Calculated {lots:.2f} lots (${actual_risk_usd:.2f} risk)")

        actual_risk_pct = (actual_risk_usd / account.equity * 100.0) if account.equity > 0 else 0.0

        # 8. Net USD Directional Correlation Guard
        # - Non-Gold assets strictly obey max_net_usd_risk_pct (so secondary FX/crypto cannot stack on top of an
        #   un-hedged Gold position until Gold locks Break-Even at +1.0R).
        # - Tier-1 Gold (XAUUSD) is never starved out of its reserved VIP slot by existing secondary FX trades:
        #   at 0.01 min lot it is permitted up to gold_min_lot_max_risk_pct (+ max_net_usd_risk_pct when VIP slot active),
        #   and at standard sizing up to max_net_usd_risk_pct + risk_per_trade_pct when VIP slot is enabled.
        current_net_usd = self.compute_open_net_usd_risk_pct(open_positions, account.equity)
        side_sign = 1 if setup.side == Side.LONG else -1
        trade_usd_contrib = prof.usd_direction_sign * side_sign * actual_risk_pct
        projected_net_usd = round(current_net_usd + trade_usd_contrib, 4)

        is_gold_min_lot = (
            setup.symbol == "XAUUSD"
            and lots > 0
            and abs(lots - quote.volume_min) < 1e-6
        )
        if setup.symbol == "XAUUSD":
            if is_gold_min_lot:
                base_gold_cap = max(self.settings.max_net_usd_risk_pct, self.settings.gold_min_lot_max_risk_pct)
                effective_usd_cap = (
                    base_gold_cap + self.settings.max_net_usd_risk_pct
                    if (self.settings.gold_reserved_extra_slot and abs(current_net_usd) > 1e-4)
                    else base_gold_cap
                )
            elif self.settings.gold_reserved_extra_slot:
                effective_usd_cap = self.settings.max_net_usd_risk_pct + self.settings.risk_per_trade_pct
            else:
                effective_usd_cap = self.settings.max_net_usd_risk_pct
        else:
            effective_usd_cap = self.settings.max_net_usd_risk_pct

        if prof.usd_direction_sign != 0 and abs(projected_net_usd) > effective_usd_cap:
            rejections.append(
                f"USD_CORRELATION_CAP: Projected net USD directional risk {abs(projected_net_usd):.2f}% > {effective_usd_cap:.2f}% cap"
            )
        else:
            passed.append(f"Net USD exposure {projected_net_usd:+.2f}% within ±{effective_usd_cap:.2f}% cap")

        # 9. Free Margin Check (with 150% safety buffer)
        required_margin = (fill_price * prof.contract_size * lots) / max(1, account.leverage)
        if required_margin * 1.5 > account.free_margin:
            rejections.append(
                f"INSUFFICIENT_FREE_MARGIN: Required ${required_margin:.2f} (x1.5 buffer) > free margin ${account.free_margin:.2f}"
            )
        else:
            passed.append(f"Margin check OK (${required_margin:.2f} req vs ${account.free_margin:.2f} free)")

        return RiskCheckResult(
            approved=(len(rejections) == 0),
            symbol=setup.symbol,
            side=setup.side,
            calculated_lots=lots,
            risk_usd=actual_risk_usd,
            risk_pct=round(actual_risk_pct, 3),
            net_rr_after_spread=round(net_rr, 2),
            projected_net_usd_risk_pct=projected_net_usd,
            adjusted_sl=setup.stop_loss,
            adjusted_tp=setup.take_profit,
            rejection_reasons=rejections,
            passed_checks=passed,
        )

    def manage_open_positions_intraday(self, broker: BrokerAdapter) -> list[str]:
        """
        Real-time 3-Stage Server-Side Ratchet & Chandelier ATR Trailing Engine (0 LLM calls):
          - Stage 1 (0.0R -> +be_trigger_r): Structural wick SL gives trade breathing room
          - Stage 2 (At +be_trigger_r, e.g. +1.15R): Profit-Plus BE Shield locks +0.20R above spread
          - Stage 3 (At +chandelier_start_r, e.g. +1.55R+): Dynamic Chandelier Volatility Trail
            ratchets MT5 server-side SL behind `Highest_High_Since_Entry - 1.65 * ATR(14)` (strictly one-way!)
        """
        actions: list[str] = []
        for pos in broker.get_open_positions():
            is_long = pos.side == Side.LONG
            try:
                q = broker.get_quote(pos.symbol)
            except Exception:
                continue

            prof = self.profiles.get(pos.symbol)
            min_pt = prof.point if prof else q.point

            # Live ATR & post-entry price extreme from M5 bars (survives app restarts seamlessly!)
            live_atr = abs(pos.tp - pos.entry_price) / 4.5 if pos.tp > 0 else q.spread_price * 8.0
            extreme_bar_price = q.bid if is_long else q.ask
            try:
                m5 = broker.get_bars(pos.symbol, "M5", 48)
                if len(m5) >= 16:
                    live_atr = max(float(compute_atr(m5, 14).iloc[-1]), min_pt * 20.0)
                    since_mask = m5["timestamp"] >= (pos.opened_at - timedelta(minutes=5))
                    recent_slice = m5.loc[since_mask] if since_mask.any() else m5.tail(3)
                    extreme_bar_price = (
                        float(recent_slice["high"].max()) if is_long else float(recent_slice["low"].min())
                    )
            except Exception:
                pass

            sl_in_profit = (pos.sl > pos.entry_price) if (is_long and pos.sl > 0) else (0 < pos.sl < pos.entry_price)
            already_be = bool(pos.be_moved or sl_in_profit)

            # Recover true initial 1R distance even after SL has already moved into profit
            tp_dist = abs(pos.tp - pos.entry_price) if pos.tp > 0 else 0.0
            if not already_be and pos.sl > 0:
                initial_r_dist = abs(pos.entry_price - pos.sl)
            elif tp_dist > 0:
                initial_r_dist = tp_dist / 3.20
            else:
                initial_r_dist = max(1.5 * live_atr, min_pt * 25.0)

            if initial_r_dist <= 0:
                continue

            mark = q.bid if is_long else q.ask
            if is_long:
                peak_price = max(pos.entry_price, mark, pos.peak_favorable_price, extreme_bar_price)
                favorable_now = mark - pos.entry_price
                favorable_peak = peak_price - pos.entry_price
            else:
                prev_trough = pos.peak_favorable_price if pos.peak_favorable_price > 0 else mark
                peak_price = min(pos.entry_price, mark, prev_trough, extreme_bar_price)
                favorable_now = pos.entry_price - mark
                favorable_peak = pos.entry_price - peak_price

            pos.peak_favorable_price = peak_price
            live_r = favorable_now / initial_r_dist
            peak_r = favorable_peak / initial_r_dist
            broker_min_gap = max(q.stops_level_points * q.point * 1.25, q.spread_price * 1.6)
            min_ratchet_step = max(0.14 * live_atr, q.spread_price * 1.2)

            # Stage 3: Chandelier ATR Volatility Ratchet Trail (+1.55R+)
            if max(live_r, peak_r) >= self.settings.chandelier_start_r:
                if is_long:
                    raw_trail = max(
                        pos.entry_price + 0.55 * initial_r_dist,
                        peak_price - self.settings.chandelier_atr_mult * live_atr,
                    )
                    safe_trail = round(min(raw_trail, q.bid - broker_min_gap), 5)
                    improves = safe_trail > (pos.sl + min_ratchet_step)
                else:
                    raw_trail = min(
                        pos.entry_price - 0.55 * initial_r_dist,
                        peak_price + self.settings.chandelier_atr_mult * live_atr,
                    )
                    safe_trail = round(max(raw_trail, q.ask + broker_min_gap), 5)
                    improves = (pos.sl <= 0) or (safe_trail < (pos.sl - min_ratchet_step))

                if improves and broker.modify_position_sl_tp(pos.ticket, new_sl=safe_trail, new_tp=pos.tp):
                    pos.sl = safe_trail
                    pos.be_moved = True
                    pos.trail_stage = 2
                    locked_r = abs(safe_trail - pos.entry_price) / initial_r_dist
                    actions.append(
                        f"Ticket #{pos.ticket} ({pos.symbol}): Chandelier ATR Trail ratcheted SL -> {safe_trail} (+{locked_r:.2f}R locked)"
                    )
                continue

            # Stage 2: Profit-Plus Break-Even Shield (+1.15R -> locks +0.20R above spread)
            if live_r >= self.settings.be_trigger_r and not already_be:
                lock_offset = max(self.settings.be_lock_r * initial_r_dist, q.spread_price * 1.5)
                if is_long:
                    new_sl = round(min(pos.entry_price + lock_offset, q.bid - broker_min_gap), 5)
                    valid_be = new_sl > pos.entry_price
                else:
                    new_sl = round(max(pos.entry_price - lock_offset, q.ask + broker_min_gap), 5)
                    valid_be = new_sl < pos.entry_price

                if valid_be and broker.modify_position_sl_tp(pos.ticket, new_sl=new_sl, new_tp=pos.tp):
                    pos.sl = new_sl
                    pos.be_moved = True
                    pos.trail_stage = 1
                    locked_r = abs(new_sl - pos.entry_price) / initial_r_dist
                    actions.append(
                        f"Ticket #{pos.ticket} ({pos.symbol}): Profit-Plus BE Shield locked SL -> {new_sl} (+{locked_r:.2f}R banked)"
                    )

        return actions
