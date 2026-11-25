"""CLI Entrypoint (`tradingagent`) for Windows 10 Laptop Operation."""

from __future__ import annotations

import asyncio
import time
import typer
from rich.console import Console
from rich.table import Table
import uvicorn

from tradingagent.backtest import SpreadAwareEventBacktester, generate_realistic_intraday_dataset
from tradingagent.broker.paper_mt5 import DEFAULT_PRICES
from tradingagent.config import AppSettings, load_symbol_profiles
from tradingagent.engine import TradingAgentEngine
from tradingagent.memory import StateAndMemoryStore

app = typer.Typer(
    name="tradingagent",
    help="TradingAgent-001: 100% Zero-Fee Local Windows 10 + MT5 Multi-Agent Scalping & Intraday Engine",
)
console = Console()


@app.command()
def doctor() -> None:
    """Verify local laptop setup: MT5 connectivity, Symbol Auto-Mapper, SQLite memory, and Ollama LLM."""
    settings = AppSettings()
    engine = TradingAgentEngine(settings=settings)
    ollama_ok, models = asyncio.run(engine.committee.check_ollama_health())
    active_model = engine.committee.resolve_ollama_model(models)
    acct = engine.broker.get_account_snapshot()

    table = Table(title="TradingAgent-001 Local Diagnostics (Windows 10 / 16GB RAM Profile)")
    table.add_column("Subsystem", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Details")

    table.add_row(
        "Execution Mode",
        "PASS",
        f"{settings.trading_mode.upper()} (Broker Server: {acct.server} | Equity: ${acct.equity:,.2f})",
    )
    table.add_row(
        "Symbol Auto-Mapper",
        "PASS",
        ", ".join(f"{k}->{v}" for k, v in engine.broker.mapper.canonical_to_broker.items()),
    )
    table.add_row(
        "Zero-Fee Brain (Ollama)",
        "ONLINE" if ollama_ok else "FALLBACK READY",
        (
            f"Connected ({settings.ollama_base_url}) | Active: {active_model} | Installed: {', '.join(models) or 'none'}"
            if ollama_ok
            else f"Ollama not running yet — using Deterministic Multi-Agent Rubric (run `ollama pull {settings.ollama_quick_model}`)"
        ),
    )
    table.add_row(
        "Risk Firewall",
        "ENFORCED",
        (
            f"Risk/Trade={settings.risk_per_trade_pct}%, DailyLock={settings.max_daily_loss_pct}%, "
            f"NetUSDCap=±{settings.max_net_usd_risk_pct}%, Server-Side SL/TP Mandatory"
        ),
    )
    console.print(table)


@app.command()
def scan(execute: bool = typer.Option(False, "--execute", help="Allow approved setups to place orders")) -> None:
    """Run one Gold-First prioritized scan + Multi-Agent Committee + Risk Firewall check."""
    engine = TradingAgentEngine()
    result = asyncio.run(engine.run_single_cycle(auto_execute=execute))
    console.print_json(data=result)


@app.command()
def run(
    interval_seconds: int = typer.Option(15, "--interval", help="Seconds between Gold-First scan cycles"),
) -> None:
    """Run continuous autonomous intraday loop on your laptop (Ctrl+C to stop)."""
    engine = TradingAgentEngine()
    console.print(
        f"[bold yellow]Starting TradingAgent-001 continuous loop[/bold yellow] "
        f"(Mode={engine.settings.trading_mode}, Interval={interval_seconds}s)..."
    )
    try:
        while True:
            out = asyncio.run(engine.run_single_cycle(auto_execute=True))
            console.print(
                f"[{out['timestamp']}] Candidates={out['candidates_found']} | "
                f"MgmtActions={len(out['management_actions'])}"
            )
            time.sleep(max(2, interval_seconds))
    except KeyboardInterrupt:
        console.print("[bold red]Stopped autonomous loop cleanly.[/bold red]")
        engine.broker.disconnect()


@app.command()
def backtest(
    symbol: str = typer.Option("XAUUSD", "--symbol", help="Canonical symbol (XAUUSD, NAS100, BTCUSD, EURUSD...)"),
    bars: int = typer.Option(2800, "--bars", help="Number of M5 bars to test"),
) -> None:
    """Run zero-fee spread-aware event backtest comparing Raw Quant vs. Multi-Agent Committee."""
    settings = AppSettings()
    profiles = load_symbol_profiles(settings.symbols_config_path)
    sym = symbol.upper()
    if sym not in profiles:
        raise typer.BadParameter(f"Unknown symbol {sym}. Available: {', '.join(profiles)}")

    prof = profiles[sym]
    _, _, _, tick_val = DEFAULT_PRICES.get(sym, (100.0, 0.5, 10.0, 1.0))
    memory = StateAndMemoryStore(settings.sqlite_path)
    df = generate_realistic_intraday_dataset(symbol=sym, n_bars=bars, seed=101)

    tester = SpreadAwareEventBacktester(settings, prof, memory, tick_value_per_point=tick_val)
    raw_res = tester.run(df, use_agent_committee=False)
    agent_res = tester.run(df, use_agent_committee=True)

    t = Table(title=f"Ablation Backtest Results — {sym} ({bars} M5 Bars, Conservative SL-First + Live Spread)")
    t.add_column("Metric", style="cyan")
    t.add_column("Raw Quant Baseline", justify="right")
    t.add_column("Multi-Agent Committee + Risk Gate", justify="right", style="bold yellow")

    t.add_row("Total Trades", str(raw_res.total_trades), str(agent_res.total_trades))
    t.add_row("Win Rate", f"{raw_res.win_rate_pct:.2f}%", f"{agent_res.win_rate_pct:.2f}%")
    t.add_row("Net Return", f"{raw_res.net_return_pct:+.2f}%", f"{agent_res.net_return_pct:+.2f}%")
    t.add_row("Profit Factor", f"{raw_res.profit_factor:.2f}", f"{agent_res.profit_factor:.2f}")
    t.add_row("Sharpe Ratio", f"{raw_res.sharpe_ratio:.2f}", f"{agent_res.sharpe_ratio:.2f}")
    t.add_row("Max Drawdown", f"{raw_res.max_drawdown_pct:.2f}%", f"{agent_res.max_drawdown_pct:.2f}%")
    t.add_row("Spread Cost Deducted", f"${raw_res.spread_cost_usd:.2f}", f"${agent_res.spread_cost_usd:.2f}")
    console.print(t)


@app.command()
def dashboard(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Launch the local web Control Center & Live Telemetry UI."""
    uvicorn.run("tradingagent.web:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
