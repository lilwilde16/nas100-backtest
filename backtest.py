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


def prepare_data(symbol: str = "NQ=F", period: str = "8d", interval: str = "1m") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("No data returned from yfinance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)

    m1 = df.copy()

    m5 = (
        m1.resample("5min")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        .dropna()
    )

    return m1, m5


def add_indicators(m1: pd.DataFrame, m5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m1 = m1.copy()
    m5 = m5.copy()

    m1["EMA9"] = ema(m1["Close"], 9)
    m1["EMA20"] = ema(m1["Close"], 20)

    m5["EMA20"] = ema(m5["Close"], 20)
    m5["EMA50"] = ema(m5["Close"], 50)

    return m1, m5


def get_m5_context(m5_hist: pd.DataFrame) -> str:
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


def body_ok(row: pd.Series, min_ratio: float = 0.40) -> bool:
    full_range = row["High"] - row["Low"]
    if full_range <= 0:
        return False
    body = abs(row["Close"] - row["Open"])
    return (body / full_range) >= min_ratio


def run_backtest(m1: pd.DataFrame, m5: pd.DataFrame) -> Summary:
    buys = sells = wins = losses = 0
    rr_target = 1.5

    m5_index = m5.index

    for i in range(25, len(m1) - 2):
        ts = m1.index[i]

        # simple session filter: 07:00-11:00 UTC and 13:00-17:00 UTC
        hour = ts.hour
        in_london = 7 <= hour < 11
        in_newyork = 13 <= hour < 17
        if not (in_london or in_newyork):
            continue

        m5_hist = m5.loc[m5_index <= ts]
        ctx = get_m5_context(m5_hist)
        if ctx == "WAIT":
            continue

        recent = m1.iloc[i - 5:i]
        curr = m1.iloc[i]
        prev = m1.iloc[i - 1]
        next_open = m1.iloc[i + 1]["Open"]

        if ctx == "BULL":
            touched_pullback = (recent["Low"] <= recent["EMA20"]).any()
            trigger = (
                touched_pullback
                and curr["Close"] > curr["EMA9"]
                and curr["High"] > prev["High"]
                and body_ok(curr)
            )
            if trigger:
                buys += 1
                stop = curr["Low"]
                risk = next_open - stop
                if risk > 0:
                    target = next_open + (risk * rr_target)
                    future = m1.iloc[i + 1:i + 21]
                    hit_tp = (future["High"] >= target).any()
                    hit_sl = (future["Low"] <= stop).any()
                    if hit_tp and not hit_sl:
                        wins += 1
                    else:
                        losses += 1

        elif ctx == "BEAR":
            touched_pullback = (recent["High"] >= recent["EMA20"]).any()
            trigger = (
                touched_pullback
                and curr["Close"] < curr["EMA9"]
                and curr["Low"] < prev["Low"]
                and body_ok(curr)
            )
            if trigger:
                sells += 1
                stop = curr["High"]
                risk = stop - next_open
                if risk > 0:
                    target = next_open - (risk * rr_target)
                    future = m1.iloc[i + 1:i + 21]
                    hit_tp = (future["Low"] <= target).any()
                    hit_sl = (future["High"] >= stop).any()
                    if hit_tp and not hit_sl:
                        wins += 1
                    else:
                        losses += 1

    total = buys + sells
    losses_nonzero = max(losses, 1)
    gross_profit = wins * 1.5
    gross_loss = losses * 1.0
    profit_factor = gross_profit / max(gross_loss, 1e-9)
    win_rate = (wins / total) if total else 0.0

    return Summary(
        total_signals=total,
        buys=buys,
        sells=sells,
        wins=wins,
        losses=losses,
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
