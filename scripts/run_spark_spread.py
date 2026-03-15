"""Run Spark-Spread & Heat-Rate Analysis.

Analyses the profitability of gas-fired power generation across 7 US ISOs:
1. Estimate implied heat rates from generation dispatch data
2. Compute spark spreads (raw and clean)
3. Estimate merit-order supply stacks
4. Build gas dispatch prediction model
5. Seasonal and cross-regional comparison
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from src.strategy.spark_spread import SparkSpreadModel, HEAT_RATES, VOM_COSTS


REGION_MAP = {
    "PJM":   "pjm",
    "MISO":  "miso",
    "NYISO": "nyis",
    "ISONE": "isne",
    "CAISO": "ciso",
    "ERCOT": "erco",
    "SPP":   "swpp",
}


def main():
    print("=" * 72)
    print("  SPARK SPREAD & HEAT RATE ANALYSIS")
    print("  Gas-Fired Generation Profitability by Region")
    print("=" * 72)

    # --- Load data ---
    gen_path = "data/raw/generation_daily.csv"
    ng_path = "data/raw/prices_natural_gas.csv"

    gen = pd.read_csv(gen_path, parse_dates=["datetime"]).set_index("datetime")
    gen.index.name = "date"
    ng = pd.read_csv(ng_path, parse_dates=["date"]).set_index("date")
    ng = ng[["Close"]].rename(columns={"Close": "ng_price"})
    ng["ng_price"] = pd.to_numeric(ng["ng_price"], errors="coerce")

    # Rename columns to match model expectations: {REGION}_gas_mwh, {REGION}_total_mwh
    rename_map = {}
    for label, prefix in REGION_MAP.items():
        ng_col = f"{prefix}_ng_total_mwh"
        wnd_col = f"{prefix}_wnd_total_mwh"
        sun_col = f"{prefix}_sun_total_mwh"
        if ng_col in gen.columns:
            rename_map[ng_col] = f"{label}_gas_mwh"
        if wnd_col in gen.columns:
            rename_map[wnd_col] = f"{label}_wind_mwh"
        if sun_col in gen.columns:
            rename_map[sun_col] = f"{label}_solar_mwh"

    gen = gen.rename(columns=rename_map)

    # Compute total generation per region (gas + wind + solar as proxy)
    for label in REGION_MAP:
        gas_c = f"{label}_gas_mwh"
        wind_c = f"{label}_wind_mwh"
        solar_c = f"{label}_solar_mwh"
        cols = [c for c in [gas_c, wind_c, solar_c] if c in gen.columns]
        if cols:
            gen[f"{label}_total_mwh"] = gen[cols].sum(axis=1)

    merged = gen.join(ng, how="inner").dropna(subset=["ng_price"])

    print(f"\nData: {len(merged)} days ({merged.index[0].date()} → {merged.index[-1].date()})")
    print(f"  Generation columns: {len(gen.columns)}")
    print(f"  NG price range: ${merged['ng_price'].min():.2f} - ${merged['ng_price'].max():.2f}/MMBtu")

    regions = [r for r in REGION_MAP if f"{r}_gas_mwh" in merged.columns]
    print(f"  Regions with gas data: {', '.join(regions)}")

    # --- Reference heat rates ---
    print(f"\n{'─' * 72}")
    print(f"  REFERENCE HEAT RATES (MMBtu/MWh)")
    print(f"{'─' * 72}")
    for plant, hr in HEAT_RATES.items():
        print(f"    {plant:25s}: {hr:5.1f}  VOM=${VOM_COSTS.get(plant, 0):.2f}/MWh")

    # --- Spark spread model ---
    model = SparkSpreadModel(carbon_price=0, assumed_power_price_premium=5.0)

    # --- Multi-region analysis ---
    print(f"\n{'─' * 72}")
    print(f"  REGIONAL SPARK SPREAD ANALYSIS")
    print(f"{'─' * 72}")

    all_spreads = {}
    all_hrs = {}
    for region in regions:
        print(f"\n  ▸ {region}")

        # Implied heat rate
        hr = model.estimate_implied_heat_rate(merged, region)
        all_hrs[region] = hr
        print(f"    Implied fleet heat rate: {hr:.1f} MMBtu/MWh")

        # Spark spreads
        spreads = model.compute_spark_spreads(merged, region, heat_rate=hr)
        all_spreads[region] = spreads

        # Statistics
        print(f"    Avg spark spread:       ${spreads['spark_spread'].mean():>6.1f}/MWh")
        print(f"    Avg clean spark spread: ${spreads['clean_spark_spread'].mean():>6.1f}/MWh")
        profitable = (spreads['spark_spread'] > 0).mean() * 100
        print(f"    Profitable periods:     {profitable:.0f}%")

        # By plant type
        print(f"    By plant type:")
        for ptype in HEAT_RATES:
            col = f"spark_{ptype}"
            if col in spreads.columns:
                avg = spreads[col].mean()
                pos = (spreads[col] > 0).mean() * 100
                print(f"      {ptype:25s}: avg=${avg:>6.1f}/MWh  profitable={pos:.0f}%")

        # Gas share stats
        if "gas_share" in spreads.columns:
            gs = spreads["gas_share"].dropna()
            if len(gs) > 0:
                print(f"    Gas generation share: mean={gs.mean():.1%}, "
                      f"median={gs.median():.1%}, max={gs.max():.1%}")

        # Seasonal pattern
        if hasattr(spreads.index, 'month'):
            monthly = spreads.groupby(spreads.index.month)["spark_spread"].mean()
            months_str = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            print(f"    Seasonal spark spread:")
            for m in range(1, 13):
                if m in monthly.index:
                    val = monthly[m]
                    if val > 0:
                        bar = "█" * min(40, int(val * 1.5))
                    else:
                        bar = "░" * min(40, int(-val * 1.5))
                    print(f"      {months_str[m-1]}: ${val:>6.1f}  {bar}")

    # --- Merit Order ---
    print(f"\n{'─' * 72}")
    print(f"  MERIT ORDER (SUPPLY STACK) ESTIMATION")
    print(f"{'─' * 72}")

    for region in regions:
        result = model.estimate_merit_order(merged, region)
        if result.marginal_fuel:
            print(f"\n  {region}:")
            print(f"    Marginal fuel:         {result.marginal_fuel}")
            print(f"    Est marginal cost:     ${result.marginal_cost_estimate:.1f}/MWh")
            print(f"    Baseload capacity:     {result.base_load_capacity_mw:,.0f} MWh/day")
            print(f"    Gas share (overall):   {result.gas_share_at_peak:.1%}")

            if len(result.stack) > 0:
                print(f"    Supply stack by load level:")
                pivoted = result.stack.pivot_table(
                    index="load_quantile", columns="fuel",
                    values="share_pct", aggfunc="mean"
                )
                if len(pivoted) > 0:
                    print(f"    {'Quantile':>10}", end="")
                    for fuel in sorted(pivoted.columns):
                        print(f"  {fuel:>8}", end="")
                    print()
                    for q, row in pivoted.iterrows():
                        print(f"    {q:>10.0%}", end="")
                        for fuel in sorted(pivoted.columns):
                            val = row.get(fuel, 0)
                            print(f"  {val:>7.1f}%", end="")
                        print()

    # --- Gas Dispatch Model ---
    print(f"\n{'─' * 72}")
    print(f"  GAS DISPATCH PREDICTION MODEL")
    print(f"{'─' * 72}")

    for region in regions[:2]:  # top 2 for speed
        print(f"\n  {region}:")
        dispatch = model.dispatch_model(merged, region)
        if len(dispatch) > 0:
            r2 = dispatch.attrs.get("r2", 0)
            mape = dispatch.attrs.get("mape", 0)
            print(f"    Expanding-window GBM dispatch model:")
            print(f"    R²:   {r2:.3f}")
            print(f"    MAPE: {mape:.1f}%")

            resid = dispatch["residual"].dropna()
            print(f"    Residual stats: mean={resid.mean():>8,.0f}, "
                  f"std={resid.std():>8,.0f}, "
                  f"skew={resid.skew():.2f}")

    # --- Cross-region comparison ---
    if len(all_spreads) >= 2:
        print(f"\n{'─' * 72}")
        print(f"  CROSS-REGION COMPARISON")
        print(f"{'─' * 72}")

        print(f"\n  {'Region':>8} {'Avg Spread':>11} {'Clean':>8} {'Profitable':>11} {'Heat Rate':>10}")
        for region in all_spreads:
            spreads = all_spreads[region]
            hr = all_hrs.get(region, 7.5)
            avg_ss = spreads['spark_spread'].mean()
            avg_cs = spreads['clean_spark_spread'].mean()
            prof = (spreads['spark_spread'] > 0).mean() * 100
            print(f"  {region:>8} ${avg_ss:>9.1f} ${avg_cs:>6.1f} {prof:>10.0f}% {hr:>9.1f}")

        # Correlation of spark spreads across regions
        ss_df = pd.DataFrame({r: s["spark_spread"] for r, s in all_spreads.items()}).dropna()
        if len(ss_df) > 30 and len(ss_df.columns) >= 2:
            print(f"\n  Spark spread correlation matrix:")
            corr = ss_df.corr()
            print(f"  {'':>8}", end="")
            for r in corr.columns:
                print(f"  {r:>8}", end="")
            print()
            for r in corr.index:
                print(f"  {r:>8}", end="")
                for c in corr.columns:
                    print(f"  {corr.loc[r,c]:>8.2f}", end="")
                print()

    # --- Summary ---
    print(f"\n{'=' * 72}")
    print("  INTERPRETATION")
    print("=" * 72)
    print("""
  • Spark spread = Power Price - Heat Rate × Gas Price
    Positive → gas plants are profitable to run
    Negative → gas plants should be turned off

  • Clean spark spread nets out variable O&M and carbon costs
    This is the actual profit margin for generation

  • Implied heat rate > nameplate efficiency means the fleet is
    dispatching less efficient peaking units at the margin

  • Seasonal pattern: spark spreads typically widen in summer
    (cooling demand → high power prices) and winter (heating demand)

  • Gas dispatch model: predicts gas generation from price + load
    This is used by gas traders to forecast "gas burn" — the key
    demand driver for pipeline flows and basis differentials

  • A real desk would use: hourly LMP data, unit-level heat rates,
    pipeline constraint models, and real-time generation stack
""")


if __name__ == "__main__":
    main()
