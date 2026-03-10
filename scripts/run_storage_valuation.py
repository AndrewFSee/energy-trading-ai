"""Run NG Storage Valuation — Intrinsic, Rolling Intrinsic, LSMC Extrinsic.

Demonstrates professional gas storage valuation methodology:
1. Calibrate OU process from Henry Hub history
2. Build seasonal forward curve
3. Compute intrinsic value via LP optimisation
4. Compute extrinsic value via Longstaff-Schwartz Monte Carlo
5. Compute storage Greeks (delta, gamma, theta, vega)
6. Show rolling intrinsic P&L path
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from src.strategy.storage_valuation import (
    StorageValuationEngine, StorageAsset, ForwardCurve,
    PriceSimulator, OUParams,
)


def main():
    print("=" * 72)
    print("  NATURAL GAS STORAGE VALUATION")
    print("  Intrinsic (LP) + Extrinsic (LSMC Longstaff-Schwartz)")
    print("=" * 72)

    # --- Load NG price history ---
    price_path = "data/raw/prices_natural_gas.csv"
    df = pd.read_csv(price_path, parse_dates=["date"]).set_index("date")
    prices = pd.to_numeric(df["Close"], errors="coerce").dropna()
    prices = prices.loc[~prices.index.duplicated(keep="first")].sort_index()
    print(f"\nLoaded {len(prices)} days of Henry Hub prices "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")
    print(f"  Current price: ${prices.iloc[-1]:.2f}/MMBtu")
    print(f"  5-year mean:   ${prices.tail(252*5).mean():.2f}/MMBtu")

    # --- Create two facility types ---
    salt = StorageAsset.salt_cavern()
    reservoir = StorageAsset.depleted_reservoir()

    for asset in [salt, reservoir]:
        print(f"\n{'─' * 72}")
        print(f"  FACILITY: {asset.name}")
        print(f"  Capacity: {asset.capacity_bcf:.0f} Bcf | "
              f"Inj: {asset.max_injection_bcf_day*1000:.0f} MMcf/d | "
              f"Wdl: {asset.max_withdrawal_bcf_day*1000:.0f} MMcf/d")
        print(f"  Costs: Inj ${asset.injection_cost}/MMBtu, "
              f"Wdl ${asset.withdrawal_cost}/MMBtu, "
              f"Fuel loss {asset.fuel_loss_pct*100:.0f}%")
        print(f"{'─' * 72}")

        # --- Calibrate ---
        engine = StorageValuationEngine(asset=asset)
        params = engine.calibrate(prices)

        print(f"\n  OU Process Calibration:")
        print(f"    κ (mean-reversion): {params['kappa']:.3f}")
        print(f"    θ (long-run mean):  ${params['theta']:.2f}/MMBtu")
        print(f"    σ (volatility):     {params['sigma']:.3f}")
        print(f"    Half-life:          {params['half_life_days']:.0f} trading days "
              f"({params['half_life_days']/21:.1f} months)")
        print(f"    Winter premium:     +{params['winter_premium_pct']:.1f}%")
        print(f"    Summer discount:    -{params['summer_discount_pct']:.1f}%")

        # --- Forward Curve ---
        fc = engine.forward_curve
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        print(f"\n  Seasonal Forward Curve (from current ${prices.iloc[-1]:.2f}):")
        fwd = fc.curve(1, 12)
        for i, (m, p) in enumerate(zip(months, fwd)):
            bar = "█" * int(p * 3)
            print(f"    {m}: ${p:5.2f}  {bar}")

        # --- Intrinsic Value (April start = typical injection season) ---
        for start_month, label in [(4, "Apr→Mar (full year)"), (10, "Oct→Mar (winter)")]:
            n_months = 12 if start_month == 4 else 6
            print(f"\n  Intrinsic Valuation ({label}):")
            # Values from LP are in $/MMBtu × Bcf; 1 Bcf = 10^6 MMBtu
            MM = 1_000_000  # conversion factor
            result = engine.intrinsic(
                start_month=start_month,
                n_months=n_months,
                initial_inventory=asset.min_inventory_bcf,
            )
            print(f"    Value: ${result.total_value * MM / 1e6:.2f}M")
            print(f"    Spread captured: ${result.spread_captured:,.2f}/MMBtu")
            print(f"    Cycles: {result.cycles:.2f}")
            if len(result.schedule) > 0:
                print(f"\n    Optimal Schedule:")
                print(f"    {'Month':>5} {'Fwd Price':>9} {'Inject':>8} {'Withdraw':>8} {'Inventory':>10} {'Fill%':>6}")
                for _, row in result.schedule.iterrows():
                    action = ""
                    if row['injection_bcf'] > 0.01:
                        action = f"{row['injection_bcf']:7.2f}"
                    else:
                        action = f"{'':>7}"
                    wdl = ""
                    if row['withdrawal_bcf'] > 0.01:
                        wdl = f"{row['withdrawal_bcf']:7.2f}"
                    else:
                        wdl = f"{'':>7}"
                    print(f"    {int(row['month']):>5} ${row['forward_price']:>7.2f} "
                          f"{action} {wdl} {row['inventory_bcf']:>9.2f} {row['fill_pct']:>5.1f}%")

        # --- LSMC Extrinsic Only for salt cavern (faster) ---
        if asset.name == "Salt_Cavern":
            print(f"\n  LSMC Extrinsic Valuation (5,000 paths):")
            t0 = time.time()
            ext = engine.extrinsic(
                start_month=4, n_months=12,
                initial_inventory=asset.min_inventory_bcf,
                n_paths=5000, seed=42,
            )
            elapsed = time.time() - t0
            MM = 1_000_000  # Bcf → MMBtu conversion
            print(f"    Total option value:  ${ext.total_option_value * MM / 1e6:>10.2f}M")
            print(f"    Intrinsic component: ${ext.intrinsic_value * MM / 1e6:>10.2f}M")
            print(f"    Extrinsic premium:   ${ext.extrinsic_value * MM / 1e6:>10.2f}M ({ext.extrinsic_pct:.1f}%)")
            print(f"    Std error:           ${ext.std_error * MM / 1e6:>10.2f}M")
            print(f"    Computation time:    {elapsed:.1f}s")

            # --- Greeks ---
            print(f"\n  Storage Greeks (finite difference, 3,000 paths):")
            t0 = time.time()
            greeks = engine.greeks(
                start_month=4, n_months=12,
                initial_inventory=asset.min_inventory_bcf,
                n_paths=3000, seed=42,
            )
            elapsed = time.time() - t0
            MM = 1_000_000
            print(f"    Delta (dV/dS):   ${greeks.delta * MM / 1e6:>10.2f}M  per $1/MMBtu spot move")
            print(f"    Gamma (d²V/dS²): ${greeks.gamma * MM / 1e6:>10.2f}M")
            print(f"    Theta (monthly): ${greeks.theta * MM / 1e6:>10.2f}M  per month time decay")
            print(f"    Vega (dV/dσ):    ${greeks.vega * MM / 1e6:>10.2f}M  per 1% vol")
            print(f"    Computation:     {elapsed:.1f}s")

        # --- Rolling Intrinsic ---
        print(f"\n  Rolling Intrinsic (last 12 months, weekly rebalance):")
        rolling = engine.rolling_intrinsic(
            prices, start_month=4, n_months=12, rebalance_freq=5,
        )
        if len(rolling) > 0:
            print(f"    Observations: {len(rolling)}")
            print(f"    {'Date':>12} {'Spot':>7} {'Value($M)':>10} {'Inv':>6} {'Action':>10}")
            # Show first 5 and last 5
            show = pd.concat([rolling.head(5), rolling.tail(5)]) if len(rolling) > 10 else rolling
            for _, row in show.iterrows():
                dt = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                val_m = row['intrinsic_value'] * 1_000_000 / 1e6  # Bcf*$/MMBtu → $M
                print(f"    {dt:>12} ${row['spot_price']:>5.2f} ${val_m:>8.2f}M "
                      f"{row['inventory_bcf']:>5.1f} {row['optimal_action']:>10}")
            if len(rolling) > 10:
                print(f"    ... ({len(rolling)-10} rows omitted) ...")

    # --- Summary ---
    print(f"\n{'=' * 72}")
    print("  INTERPRETATION")
    print("=" * 72)
    print("""
  • Intrinsic value captures deterministic calendar-spread arbitrage:
    buy in summer (low season), sell in winter (high season).

  • Extrinsic value represents the real-option premium from volatility:
    the ability to re-optimise injection/withdrawal as prices move.
    Typically 30-60% of total value for salt caverns (fast cycling).

  • Storage Greeks:
    - Delta > 0 means value increases with gas price (net long bias)
    - High Gamma means non-linear exposure → option-like payoff
    - Theta < 0 means time decay as remaining optionality shrinks
    - Vega > 0 because higher vol increases option value

  • A real desk would calibrate from broker forward curves (ICE)
    and use trinomial trees or full LSMC with multiple state variables
    (temperature, storage levels, basis differentials).
""")


if __name__ == "__main__":
    main()
