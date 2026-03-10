"""Run Portfolio VaR & Stress-Testing Analysis.

Demonstrates professional risk management for an energy trading book:
1. Compute VaR by 6 different methodologies (99% 1-day and 10-day)
2. Component VaR decomposition
3. Energy-specific stress scenarios (polar vortex, hurricane, etc.)
4. VaR backtesting with Kupiec coverage test
5. Reverse stress testing
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from src.strategy.var_model import (
    VaREngine, StressTestEngine, VaRBacktester,
    build_risk_dashboard, ENERGY_STRESS_SCENARIOS,
)


def main():
    print("=" * 72)
    print("  PORTFOLIO VALUE-AT-RISK & STRESS TESTING")
    print("  Multi-Method Risk Analysis for Energy Trading Book")
    print("=" * 72)

    # --- Load NG price history ---
    ng = pd.read_csv("data/raw/prices_natural_gas.csv", parse_dates=["date"]).set_index("date")
    ng["Close"] = pd.to_numeric(ng["Close"], errors="coerce")
    ng = ng.dropna(subset=["Close"]).sort_index()
    ng = ng.loc[~ng.index.duplicated(keep="first")]

    # Daily returns
    ng["return"] = ng["Close"].pct_change()
    returns = ng["return"].dropna()

    # Also try to load crude oil for multi-asset portfolio
    try:
        import yfinance as yf
        crude = yf.download("CL=F", start=returns.index[0].strftime('%Y-%m-%d'),
                           end=returns.index[-1].strftime('%Y-%m-%d'), progress=False)
        close_col = crude["Close"]
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        crude_ret = close_col.pct_change().dropna()
        crude_ret = pd.Series(crude_ret.values, index=crude_ret.index, name="crude_oil")
        has_crude = len(crude_ret) > 100
    except Exception:
        has_crude = False

    # Try to load power ETF as proxy
    try:
        import yfinance as yf
        ung = yf.download("UNG", start="2015-01-01",
                         end=returns.index[-1].strftime('%Y-%m-%d'), progress=False)
        close_col = ung["Close"]
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        ung_ret = close_col.pct_change().dropna()
        ung_ret = pd.Series(ung_ret.values, index=ung_ret.index, name="ng_etf")
        has_ung = len(ung_ret) > 100
    except Exception:
        has_ung = False

    print(f"\nData: {len(returns)} days ({returns.index[0].date()} → {returns.index[-1].date()})")
    print(f"  NG price range: ${ng['Close'].min():.2f} - ${ng['Close'].max():.2f}/MMBtu")
    if has_crude:
        print(f"  Crude data:     {len(crude_ret)} days")
    if has_ung:
        print(f"  UNG ETF data:   {len(ung_ret)} days")

    # --- Tail Statistics ---
    engine = VaREngine(confidence=0.99, horizon_days=1)
    tail = engine.tail_statistics(returns)

    print(f"\n{'─' * 72}")
    print(f"  RETURN DISTRIBUTION STATISTICS (Henry Hub NG)")
    print(f"{'─' * 72}")
    print(f"    Mean daily return:    {tail['mean_daily']*100:>8.3f}%")
    print(f"    Daily volatility:     {tail['std_daily']*100:>8.3f}%")
    print(f"    Annualised volatility:{tail['annualized_vol']*100:>8.1f}%")
    print(f"    Skewness:             {tail['skewness']:>8.2f}  "
          f"({'left-skewed' if tail['skewness'] < 0 else 'right-skewed'})")
    print(f"    Excess kurtosis:      {tail['excess_kurtosis']:>8.2f}  "
          f"({'fat tails' if tail['excess_kurtosis'] > 0 else 'thin tails'})")
    print(f"    Worst daily return:   {tail['min_daily_return']*100:>8.1f}%")
    print(f"    Best daily return:    {tail['max_daily_return']*100:>8.1f}%")
    print(f"    Worst 5-day avg:      {tail['worst_5_avg']*100:>8.1f}%")
    print(f"    Max drawdown:         {tail['max_drawdown']*100:>8.1f}%")
    print(f"    Jarque-Bera stat:     {tail['jarque_bera_stat']:>8.0f}  "
          f"(p={tail['jarque_bera_pval']:.4f} → {'REJECT normality' if tail['jarque_bera_pval'] < 0.05 else 'normal'})")
    print(f"    Negative days:        {tail['pct_negative_days']:>8.1f}%")

    # --- VaR Methods (1-day, 99%) ---
    PORTFOLIO_VALUE = 10_000_000  # $10M notional

    print(f"\n{'─' * 72}")
    print(f"  VALUE-AT-RISK: 99% 1-DAY (Portfolio = ${PORTFOLIO_VALUE/1e6:.0f}M)")
    print(f"{'─' * 72}")

    results_1d = engine.full_analysis(returns, PORTFOLIO_VALUE)
    print(f"\n  {'Method':<35} {'VaR%':>7} {'VaR$':>12} {'CVaR%':>7} {'CVaR$':>12}")
    print(f"  {'─'*35} {'─'*7} {'─'*12} {'─'*7} {'─'*12}")
    for r in results_1d:
        print(f"  {r.method:<35} {r.var*100:>6.2f}% ${r.var_dollar:>10,.0f} "
              f"{r.cvar*100:>6.2f}% ${r.cvar_dollar:>10,.0f}")

    # --- VaR Methods (10-day, 99%) ---
    engine_10d = VaREngine(confidence=0.99, horizon_days=10)
    results_10d = engine_10d.full_analysis(returns, PORTFOLIO_VALUE)

    print(f"\n{'─' * 72}")
    print(f"  VALUE-AT-RISK: 99% 10-DAY (Portfolio = ${PORTFOLIO_VALUE/1e6:.0f}M)")
    print(f"{'─' * 72}")
    print(f"\n  {'Method':<35} {'VaR%':>7} {'VaR$':>12} {'CVaR%':>7} {'CVaR$':>12}")
    print(f"  {'─'*35} {'─'*7} {'─'*12} {'─'*7} {'─'*12}")
    for r in results_10d:
        print(f"  {r.method:<35} {r.var*100:>6.2f}% ${r.var_dollar:>10,.0f} "
              f"{r.cvar*100:>6.2f}% ${r.cvar_dollar:>10,.0f}")

    # --- VaR at multiple confidence levels ---
    print(f"\n{'─' * 72}")
    print(f"  VaR BY CONFIDENCE LEVEL (Historical, 1-day)")
    print(f"{'─' * 72}")
    for conf in [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]:
        eng = VaREngine(confidence=conf, horizon_days=1)
        r = eng.historical_var(returns, PORTFOLIO_VALUE)
        print(f"    {conf*100:>5.1f}%: VaR={r.var*100:>5.2f}% (${r.var_dollar:>10,.0f})  "
              f"CVaR={r.cvar*100:>5.2f}% (${r.cvar_dollar:>10,.0f})")

    # --- Component VaR (multi-asset portfolio) ---
    if has_crude or has_ung:
        print(f"\n{'─' * 72}")
        print(f"  COMPONENT VaR — MULTI-ASSET PORTFOLIO")
        print(f"{'─' * 72}")

        # Build multi-asset return DataFrame
        multi_rets = pd.DataFrame({"natural_gas": returns})
        positions = {"natural_gas": 5_000_000}  # $5M
        asset_exp = {"natural_gas": "natural_gas"}

        if has_crude:
            multi_rets = multi_rets.join(pd.DataFrame({"crude_oil": crude_ret}), how="inner")
            positions["crude_oil"] = 3_000_000
            asset_exp["crude_oil"] = "crude_oil"

        if has_ung:
            multi_rets = multi_rets.join(pd.DataFrame({"ng_etf": ung_ret}), how="inner")
            positions["ng_etf"] = 2_000_000
            asset_exp["ng_etf"] = "natural_gas"

        multi_rets = multi_rets.dropna()
        total_pos = sum(positions.values())

        print(f"\n  Portfolio composition:")
        for asset, pos in positions.items():
            print(f"    {asset:20s}: ${pos/1e6:.1f}M ({pos/total_pos*100:.0f}%)")
        print(f"    {'TOTAL':20s}: ${total_pos/1e6:.1f}M")

        comp = engine.component_var(multi_rets, 
                                     {k: v/total_pos for k, v in positions.items()},
                                     total_pos)
        if comp:
            print(f"\n  Component VaR (99%, 1-day):")
            total_cvar = sum(comp.values())
            for asset, cv in sorted(comp.items(), key=lambda x: -abs(x[1])):
                pct = cv / total_cvar * 100 if total_cvar != 0 else 0
                print(f"    {asset:20s}: ${cv:>10,.0f}  ({pct:.0f}% of total)")
            print(f"    {'TOTAL':20s}: ${total_cvar:>10,.0f}")

        # Correlation matrix
        print(f"\n  Return correlation matrix ({len(multi_rets)} observations):")
        corr = multi_rets.corr()
        print(f"  {'':>20}", end="")
        for c in corr.columns:
            print(f"  {c:>14}", end="")
        print()
        for r in corr.index:
            print(f"  {r:>20}", end="")
            for c in corr.columns:
                print(f"  {corr.loc[r,c]:>14.3f}", end="")
            print()

    else:
        positions = {"natural_gas": PORTFOLIO_VALUE}
        asset_exp = {"natural_gas": "natural_gas"}

    # --- Stress Testing ---
    print(f"\n{'─' * 72}")
    print(f"  STRESS TESTING — ENERGY-SPECIFIC SCENARIOS")
    print(f"{'─' * 72}")

    stress_engine = StressTestEngine()
    stress_results = stress_engine.run_all(positions, asset_exp, PORTFOLIO_VALUE)

    print(f"\n  {'Scenario':<25} {'Port P&L':>12} {'P&L%':>7} {'Worst Asset':>15} {'Loss%':>7}")
    print(f"  {'─'*25} {'─'*12} {'─'*7} {'─'*15} {'─'*7}")
    for sr in stress_results:
        print(f"  {sr.scenario:<25} ${sr.portfolio_pnl:>10,.0f} {sr.portfolio_pnl_pct:>6.1f}% "
              f"{sr.worst_asset:>15} {sr.worst_loss_pct:>6.1f}%")

    # Detail on worst scenario
    worst = stress_results[0]  # already sorted by P&L
    print(f"\n  Worst scenario: {worst.scenario}")
    print(f"  {worst.description}")
    print(f"  Asset-level P&L:")
    for asset, pnl in sorted(worst.asset_impacts.items(), key=lambda x: x[1]):
        print(f"    {asset:20s}: ${pnl:>10,.0f}")

    # --- Reverse Stress Test ---
    print(f"\n{'─' * 72}")
    print(f"  REVERSE STRESS TEST")
    print(f"{'─' * 72}")

    for threshold in [0.05, 0.10, 0.20]:
        print(f"\n  Shocks needed for {threshold*100:.0f}% portfolio loss:")
        reverse = stress_engine.reverse_stress_test(positions, asset_exp, PORTFOLIO_VALUE, threshold)
        for factor, shock in sorted(reverse.items(), key=lambda x: abs(x[1])):
            direction = "decline" if shock < 0 else "rally"
            print(f"    {factor:20s}: {abs(shock)*100:.1f}% {direction}")

    # --- VaR Backtesting ---
    print(f"\n{'─' * 72}")
    print(f"  VaR MODEL BACKTESTING (Kupiec Coverage Test)")
    print(f"{'─' * 72}")

    for conf in [0.95, 0.99]:
        rolling_var = VaRBacktester.rolling_var(returns, window=252, confidence=conf)
        bt = VaRBacktester.backtest(returns, rolling_var, confidence=conf)

        if bt:
            print(f"\n  {conf*100:.0f}% VaR Backtest (252-day rolling window):")
            print(f"    Observations:     {bt['n_observations']}")
            print(f"    Exceedances:      {bt['n_exceedances']} "
                  f"(expected ≈ {bt['expected_rate'] * bt['n_observations']:.0f})")
            print(f"    Exceedance rate:  {bt['exceedance_rate']*100:.2f}% "
                  f"(expected: {bt['expected_rate']*100:.1f}%)")
            print(f"    Kupiec LR stat:   {bt['kupiec_lr_stat']:.2f}")
            print(f"    Kupiec p-value:   {bt['kupiec_p_value']:.4f}")
            status = "✓ ACCEPTED" if bt['model_accepted'] else "✗ REJECTED"
            print(f"    Model status:     {status} (at 5% significance)")

    # --- Summary ---
    print(f"\n{'=' * 72}")
    print("  INTERPRETATION")
    print("=" * 72)
    print(f"""
  • NG returns show excess kurtosis of {tail['excess_kurtosis']:.1f} — heavy tails mean
    Gaussian VaR UNDERSTATES risk (as shown by the method comparison above).

  • Student-t and Historical VaR better capture tail risk.
    EWMA VaR is most responsive to recent volatility regime changes.

  • Cornish-Fisher adjusts Gaussian VaR for skewness & kurtosis
    without full distributional assumptions — a good middle ground.

  • Stress test results show the portfolio is most vulnerable to
    {worst.scenario.replace('_', ' ')} ({worst.portfolio_pnl_pct:+.1f}% P&L).

  • Kupiec backtest validates whether the VaR model has the correct
    exceedance rate. Rejection means the model needs recalibration.

  • A real desk would additionally compute:
    - Intraday VaR (position changes throughout the day)
    - Greeks-based VaR (for options books)
    - Liquidity-adjusted VaR (for illiquid basis positions)
    - Incremental VaR for new trade approval
""")


if __name__ == "__main__":
    main()
