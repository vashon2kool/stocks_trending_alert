"""
Nasdaq 100 Dip Scanner
======================
Scans all Nasdaq 100 stocks for "buy the dip" opportunities using:
  - RSI (Relative Strength Index) < 40  → oversold territory
  - Price vs 50-day SMA                 → % below moving average
  - Price vs 52-week high               → % off peak
  - Volume spike                        → unusual selling pressure (capitulation)
  - MACD crossover                      → early momentum reversal signal

Requirements:
    pip install yfinance pandas ta rich
"""

import yfinance as yf
import pandas as pd
import ta
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import box
from datetime import datetime

console = Console()

# ── Nasdaq 100 tickers (as of mid-2025) ──────────────────────────────────────
NASDAQ_100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO",
    "COST", "NFLX", "ASML", "AZN", "TMUS", "AMD", "PEP", "QCOM", "LIN",
    "ADBE", "TXN", "ISRG", "AMAT", "BKNG", "AMGN", "INTU", "MU", "HON",
    "VRTX", "LRCX", "REGN", "PANW", "GILD", "ADI", "KLAC", "SBUX", "MELI",
    "INTC", "CDNS", "CRWD", "SNPS", "MDLZ", "CTAS", "ORLY", "PYPL", "WDAY",
    "MAR", "ABNB", "CEG", "NXPI", "ADP", "CHTR", "MRVL", "PCAR", "FTNT",
    "MNST", "PAYX", "KDP", "DXCM", "ROST", "CPRT", "ODFL", "IDXX", "FAST",
    "BIIB", "VRSK", "DLTR", "EA", "XEL", "CTSH", "CSGP", "ZS", "TEAM",
    "FANG", "GEHC", "ON", "EXC", "TSLA", "TTWO", "DDOG", "BKR", "ILMN",
    "WBD", "SIRI", "GFS", "ENPH", "ZM", "LCID", "RIVN", "SMCI", "TTD",
    "OKTA", "ALGN", "MRNA", "CCEP", "CDW", "CINF", "FITB", "PDD", "EBAY",
]

# ── Dip thresholds (tune these to your risk appetite) ────────────────────────
RSI_OVERSOLD        = 40     # RSI below this = oversold
SMA50_DIP_PCT       = 5.0    # % below 50-day SMA
HIGH52W_DIP_PCT     = 15.0   # % below 52-week high
VOLUME_SPIKE_MULT   = 1.5    # today's volume > X × 20-day avg volume


def score_dip(rsi, pct_below_sma50, pct_off_high, vol_spike, macd_cross):
    """Return a 0-5 composite dip score (higher = stronger buy signal)."""
    score = 0
    if rsi is not None and rsi < RSI_OVERSOLD:
        score += 1
        if rsi < 30:
            score += 1                     # extra point for deeply oversold
    if pct_below_sma50 is not None and pct_below_sma50 >= SMA50_DIP_PCT:
        score += 1
    if pct_off_high is not None and pct_off_high >= HIGH52W_DIP_PCT:
        score += 1
    if vol_spike:
        score += 1
    if macd_cross:
        score += 1
    return score


def analyze_ticker(ticker: str) -> dict | None:
    """Download data and compute dip indicators for one ticker."""
    try:
        df = yf.download(ticker, period="1y", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 60:
            return None

        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        # ── RSI ──────────────────────────────────────────────────────────────
        rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi()
        rsi = round(float(rsi_series.iloc[-1]), 1)

        # ── 50-day SMA ───────────────────────────────────────────────────────
        sma50 = close.rolling(50).mean().iloc[-1]
        price = float(close.iloc[-1])
        pct_below_sma50 = round((sma50 - price) / sma50 * 100, 2)

        # ── 52-week high ─────────────────────────────────────────────────────
        high_52w = float(close.tail(252).max())
        pct_off_high = round((high_52w - price) / high_52w * 100, 2)

        # ── Volume spike ─────────────────────────────────────────────────────
        avg_vol = float(volume.tail(20).mean())
        today_vol = float(volume.iloc[-1])
        vol_spike = today_vol > VOLUME_SPIKE_MULT * avg_vol

        # ── MACD bullish crossover (signal crossed above MACD line) ──────────
        macd_obj  = ta.trend.MACD(close)
        macd_line = macd_obj.macd()
        signal    = macd_obj.macd_signal()
        # Crossover: yesterday signal > macd, today signal < macd
        macd_cross = (
            float(signal.iloc[-2]) > float(macd_line.iloc[-2]) and
            float(signal.iloc[-1]) < float(macd_line.iloc[-1])
        )

        # ── Composite score ──────────────────────────────────────────────────
        dip_score = score_dip(rsi, pct_below_sma50, pct_off_high,
                              vol_spike, macd_cross)

        return {
            "ticker":          ticker,
            "price":           round(price, 2),
            "rsi":             rsi,
            "pct_below_sma50": pct_below_sma50,
            "pct_off_high":    pct_off_high,
            "vol_spike":       "🔥 Yes" if vol_spike else "No",
            "macd_cross":      "✅ Yes" if macd_cross else "No",
            "dip_score":       dip_score,
        }

    except Exception as e:
        console.print(f"[dim]  ⚠ {ticker}: {e}[/dim]")
        return None


def render_table(results: list[dict]):
    """Print a Rich table sorted by dip score."""
    results.sort(key=lambda r: r["dip_score"], reverse=True)
    strong = [r for r in results if r["dip_score"] >= 3]

    table = Table(
        title=f"📉  Nasdaq 100 Dip Scanner  —  {datetime.today().strftime('%Y-%m-%d')}",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )

    table.add_column("Ticker",          style="bold white",  justify="center")
    table.add_column("Price $",         style="yellow",      justify="right")
    table.add_column("RSI",             justify="right")
    table.add_column("% below SMA50",  justify="right")
    table.add_column("% off 52w High", justify="right")
    table.add_column("Vol Spike",       justify="center")
    table.add_column("MACD Cross",      justify="center")
    table.add_column("Dip Score ★",    justify="center")

    for r in strong:
        score   = r["dip_score"]
        rsi_str = f"[red]{r['rsi']}[/red]" if r["rsi"] < 30 else f"[yellow]{r['rsi']}[/yellow]"
        s_style = "green" if score >= 4 else "yellow"
        stars   = "★" * score + "☆" * (5 - score)

        table.add_row(
            r["ticker"],
            str(r["price"]),
            rsi_str,
            f"{r['pct_below_sma50']:.1f}%",
            f"{r['pct_off_high']:.1f}%",
            r["vol_spike"],
            r["macd_cross"],
            f"[{s_style}]{stars} ({score}/5)[/{s_style}]",
        )

    console.print(table)
    console.print(
        f"\n[bold green]✅ {len(strong)} stocks[/bold green] met the dip criteria "
        f"(score ≥ 3/5) out of {len(results)} scanned.\n"
    )

    # ── Export to CSV ─────────────────────────────────────────────────────────
    df_out = pd.DataFrame(results)
    df_out.to_csv("nasdaq100_dip_results.csv", index=False)
    console.print("[dim]Full results saved → nasdaq100_dip_results.csv[/dim]\n")


def main():
    console.rule("[bold cyan]Nasdaq 100 Dip Scanner[/bold cyan]")
    console.print(
        "\n[bold]Criteria:[/bold]\n"
        f"  • RSI < [yellow]{RSI_OVERSOLD}[/yellow] (oversold)\n"
        f"  • Price ≥ [yellow]{SMA50_DIP_PCT}%[/yellow] below 50-day SMA\n"
        f"  • Price ≥ [yellow]{HIGH52W_DIP_PCT}%[/yellow] below 52-week high\n"
        f"  • Volume spike > [yellow]{VOLUME_SPIKE_MULT}×[/yellow] 20-day average\n"
        f"  • MACD bullish crossover\n"
        "\n[dim]Scoring: 1 point per condition met; stocks with ≥ 3 points shown.[/dim]\n"
    )

    results = []
    for ticker in track(NASDAQ_100, description="Scanning…"):
        result = analyze_ticker(ticker)
        if result:
            results.append(result)

    if not results:
        console.print("[red]No data retrieved. Check your internet connection.[/red]")
        return

    render_table(results)


if __name__ == "__main__":
    main()