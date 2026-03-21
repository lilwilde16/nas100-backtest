from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

@dataclass
class Summary:
    total_signals: int
    buys: int
    sells: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def prepare_data(
    symbol: str = "NQ=F",
    period: str = "8d",
    interval: str = "1m",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("No data returned from yfinance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)

    m1 = df.copy()

    # Derive M5 from the same M1 source so both timeframes stay consistent
    m5 = (
        m1.resample("5min")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )

    return m1, m5


def add_indicators(m1: pd.DataFrame, m5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m1 = m1.copy()
    m5 = m5.copy()

    # M1 Bollinger Bands: 15,1
    m1["BB_MID"] = m1["Close"].rolling(15).mean()
    bb_std = m1["Close"].rolling(15).std(ddof=0)
    m1["BB_UPPER"] = m1["BB_MID"] + (bb_std * 1.0)
    m1["BB_LOWER"] = m1["BB_MID"] - (bb_std * 1.0)

    # M5 trend filter
    m5["EMA20"] = ema(m5["Close"], 20)
    m5["EMA50"] = ema(m5["Close"], 50)

    return m1.dropna().copy(), m5.dropna().copy()


def in_session(ts: pd.Timestamp) -> bool:
    hour = ts.hour
    in_london = 7 <= hour < 11
    in_newyork = 13 <= hour < 17
    return in_london or in_newyork


def body_ok(row: pd.Series, min_ratio: float = 0.40) -> bool:
    full_range = row["High"] - row["Low"]
    if full_range <= 0:
        return False
    body = abs(row["Close"] - row["Open"])
    return (body / full_range) >= min_ratio


def get_m5_context(m5_hist: pd.DataFrame) -> str:
    """
    Returns:
    - BULL
    - BEAR
    - WAIT
    """
    if len(m5_hist) < 55:
        return "WAIT"

    last = m5_hist.iloc[-1]
    three_back = m5_hist.iloc[-4]

    bullish = (
        last["EMA20"] > last["EMA50"]
        and last["EMA20"] > three_back["EMA20"]
        and last["Close"] > last["EMA20"]
    )

    bearish = (
        last["EMA20"] < last["EMA50"]
        and last["EMA20"] < three_back["EMA20"]
        and last["Close"] < last["EMA20"]
    )

    if bullish:
        return "BULL"
    if bearish:
        return "BEAR"
    return "WAIT"


def get_m1_action(m1: pd.DataFrame, idx: int, m5_context: str) -> str:
    """
    Returns:
    - BUY
    - SELL
    - WATCH
    - WAIT
    """
    if idx < 5:
        return "WAIT"

    curr = m1.iloc[idx]
    recent = m1.iloc[idx - 4 : idx + 1]  # current bar + previous 4 bars

    if m5_context == "BULL":
        touched_lower = (recent["Low"] <= recent["BB_LOWER"]).any()
        closed_back_inside = curr["Close"] > curr["BB_LOWER"]
        bullish_close = curr["Close"] > curr["Open"]
        confirmed = touched_lower and closed_back_inside and bullish_close and body_ok(curr)

        if confirmed:
            return "BUY"
        if touched_lower:
            return "WATCH"
        return "WAIT"

    if m5_context == "BEAR":
        touched_upper = (recent["High"] >= recent["BB_UPPER"]).any()
        closed_back_inside = curr["Close"] < curr["BB_UPPER"]
        bearish_close = curr["Close"] < curr["Open"]
        confirmed = touched_upper and closed_back_inside and bearish_close and body_ok(curr)

        if confirmed:
            return "SELL"
        if touched_upper:
            return "WATCH"
        return "WAIT"

    return "WAIT"


def evaluate_trade(
    m1: pd.DataFrame,
    signal_idx: int,
    side: str,
    rr_target: float = 1.5,
    max_hold_bars: int = 20,
) -> tuple[str, int]:
    """
    Returns:
    - outcome: WIN / LOSS / EXPIRED
    - bars held
    """
    signal = m1.iloc[signal_idx]
    entry_bar = m1.iloc[signal_idx + 1]
    entry = float(entry_bar["Open"])

    if side == "BUY":
        stop = float(signal["Low"])
        risk = entry - stop
        if risk <= 0:
            return "EXPIRED", 0
        target = entry + (risk * rr_target)

    else:  # SELL
        stop = float(signal["High"])
        risk = stop - entry
        if risk <= 0:
            return "EXPIRED", 0
        target = entry - (risk * rr_target)

    future = m1.iloc[signal_idx + 1 : signal_idx + 1 + max_hold_bars]

    bars_held = 0
    for _, row in future.iterrows():
        bars_held += 1

        if side == "BUY":
            hit_tp = float(row["High"]) >= target
            hit_sl = float(row["Low"]) <= stop
            if hit_tp and not hit_sl:
                return "WIN", bars_held
            if hit_sl:
                return "LOSS", bars_held

        else:  # SELL
            hit_tp = float(row["Low"]) <= target
            hit_sl = float(row["High"]) >= stop
            if hit_tp and not hit_sl:
                return "WIN", bars_held
            if hit_sl:
                return "LOSS", bars_held

    return "EXPIRED", bars_held


def run_backtest(m1: pd.DataFrame, m5: pd.DataFrame) -> Summary:
    buys = sells = watch_count = wait_count = 0
    wins = losses = expired = 0
    rr_target = 1.5
    max_hold_bars = 20
    m5_index = m5.index

    # One trade at a time
    i = 20
    last_index = len(m1) - max_hold_bars - 2

    while i < last_index:
        ts = m1.index[i]

        if not in_session(ts):
            wait_count += 1
            i += 1
            continue

        m5_hist = m5.loc[m5_index <= ts]
        m5_context = get_m5_context(m5_hist)

        if m5_context == "WAIT":
            wait_count += 1
            i += 1
            continue

        action = get_m1_action(m1, i, m5_context)

        if action == "WAIT":
            wait_count += 1
            i += 1
            continue

        if action == "WATCH":
            watch_count += 1
            i += 1
            continue

        if action == "BUY":
            buys += 1
            outcome, bars_held = evaluate_trade(m1, i, "BUY", rr_target, max_hold_bars)
        else:
            sells += 1
            outcome, bars_held = evaluate_trade(m1, i, "SELL", rr_target, max_hold_bars)

        if outcome == "WIN":
            wins += 1
        elif outcome == "LOSS":
            losses += 1
        else:
            expired += 1

        i += max(bars_held, 1)

    total = buys + sells
    gross_profit = wins * rr_target
    gross_loss = losses * 1.0
    profit_factor = gross_profit / max(gross_loss, 1e-9)
    win_rate = (wins / total) if total else 0.0

    return Summary(
        total_signals=total,
        buys=buys,
        sells=sells,
        watch_count=watch_count,
        wait_count=wait_count,
        wins=wins,
        losses=losses,
        expired=expired,
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
    )


def main() -> None:
    m1, m5 = prepare_data()
    m1, m5 = add_indicators(m1, m5)
    summary = run_backtest(m1, m5)

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
