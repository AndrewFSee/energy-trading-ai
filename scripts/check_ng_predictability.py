#!/usr/bin/env python3
"""Quick analysis: is natural gas more predictable than crude oil?"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

def analyse(name, path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    close = df["Close"]
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Range: {df.index.min().date()} → {df.index.max().date()}  ({len(df)} rows)")
    print(f"  Latest close: ${close.iloc[-1]:.2f}")

    ret_1d = np.log(close / close.shift(1)).dropna()
    ret_5d = np.log(close / close.shift(5)).dropna()
    ret_20d = np.log(close / close.shift(20)).dropna()

    print(f"\n  Annualized vol: {ret_1d.std() * np.sqrt(252):.1%}")
    print(f"  Return autocorrelation:")
    print(f"    1d  lag-1: {ret_1d.autocorr(1):+.4f}   lag-5: {ret_1d.autocorr(5):+.4f}")
    print(f"    5d  lag-1: {ret_5d.autocorr(1):+.4f}")
    print(f"    20d lag-1: {ret_20d.autocorr(1):+.4f}")

    # Seasonality: month-of-year effect
    df["month"] = df.index.month
    df["ret_20d"] = ret_20d
    monthly = df.groupby("month")["ret_20d"].agg(["mean", "std", "count"])
    print(f"\n  Monthly 20d return patterns:")
    for m, row in monthly.iterrows():
        bar = "+" * int(abs(row["mean"]) * 200) if row["mean"] > 0 else "-" * int(abs(row["mean"]) * 200)
        print(f"    Month {m:2d}: mean={row['mean']:+.4f} std={row['std']:.4f} n={int(row['count']):>4}  {bar}")

    # Non-overlapping 20d directional consistency
    nonoverlap = df.iloc[::20].copy()
    nonoverlap["target"] = np.log(close.shift(-20) / close).reindex(nonoverlap.index)
    nonoverlap = nonoverlap.dropna(subset=["target"])
    up_pct = (nonoverlap["target"] > 0).mean()
    print(f"\n  Non-overlapping 20d samples: {len(nonoverlap)}")
    print(f"  Up %: {up_pct:.1%} (base rate for classification)")

    return df, ret_1d


print("NATURAL GAS vs CRUDE OIL: Predictability Comparison")
print("=" * 60)

ng_df, ng_ret = analyse("Natural Gas (Henry Hub)", "data/raw/prices_natural_gas.csv")
wti_df, wti_ret = analyse("WTI Crude Oil", "data/raw/prices_wti.csv")
analyse("Heating Oil", "data/raw/prices_heating_oil.csv")

# Also check NG storage data
print(f"\n{'='*60}")
print("  EIA Natural Gas Storage")
ngs = pd.read_csv("data/raw/eia_natgas_storage.csv", index_col=0, parse_dates=True)
print(f"  Range: {ngs.index.min().date()} → {ngs.index.max().date()}")
print(f"  Rows: {len(ngs)}")
print(f"  Columns: {list(ngs.columns)}")
print(f"  Sample:\n{ngs.tail(5)}")

# Weather sensitivity test: NG price vs seasonal pattern
print(f"\n{'='*60}")
print("  Natural Gas: Seasonal Price Level")
ng_monthly_price = ng_df.groupby("month")["Close"].mean()
for m, p in ng_monthly_price.items():
    bar = "#" * int(p / ng_monthly_price.max() * 40)
    print(f"    Month {m:2d}: ${p:6.2f}  {bar}")

print(f"\n{'='*60}")
print("  KEY TAKEAWAY")
print(f"  NG annualized vol: {ng_ret.std() * np.sqrt(252):.1%}")
print(f"  NG 1d autocorr:    {ng_ret.autocorr(1):+.4f}")
print(f"  WTI annualized vol: {wti_ret.std() * np.sqrt(252):.1%}")
print(f"  WTI 1d autocorr:   {wti_ret.autocorr(1):+.4f}")

ng_20d = np.log(ng_df["Close"] / ng_df["Close"].shift(20)).dropna()
wti_20d = np.log(wti_df["Close"] / wti_df["Close"].shift(20)).dropna()
print(f"  NG 20d autocorr:   {ng_20d.autocorr(1):+.4f}")
print(f"  WTI 20d autocorr:  {wti_20d.autocorr(1):+.4f}")
print(f"\n  Higher autocorrelation = more momentum = potentially more predictable")
