#!/usr/bin/env python
"""Test whether NG production forecasts help predict price movements.

Three analyses:
  1. Signal analysis:  When model predicts production UP vs DOWN,
     what happens to NG prices at 5/10/20/60/90 day horizons?
  2. Ablation:  XGBoost price direction classifier with and without
     production forecast features (walk-forward CV).
  3. Granger causality: Does production forecast probability
     Granger-cause price returns?

Usage::
    python scripts/test_production_vs_price.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Merge production predictions with NG price data."""
    # Production model predictions (out-of-sample from walk-forward CV)
    pred = pd.read_csv(
        PROJECT_ROOT / "data/processed/ng_production_predictions.csv",
        parse_dates=["date"],
    )
    pred = pred.set_index("date").sort_index()
    pred = pred.rename(columns={
        "actual": "prod_actual",
        "predicted": "prod_predicted",
        "probability": "prod_prob",
    })

    # NG prices
    prices = pd.read_csv(
        PROJECT_ROOT / "data/raw/prices_natural_gas.csv",
        index_col=0, parse_dates=True,
    )
    prices.index.name = "date"
    prices = prices[["Close"]].rename(columns={"Close": "ng_price"})
    prices = prices.sort_index()

    # Merge on date
    df = pred.join(prices, how="inner")
    df = df.dropna(subset=["ng_price"])
    logger.info("Merged data: %d rows (%s to %s)",
                len(df), df.index.min().date(), df.index.max().date())
    return df


def signal_analysis(df: pd.DataFrame) -> None:
    """Analyse price returns conditional on production forecast signal."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: Production Signal vs Future NG Price Returns")
    print("=" * 70)

    # Forward returns at various horizons
    horizons = [5, 10, 20, 60, 90]
    for h in horizons:
        df[f"fwd_ret_{h}d"] = df["ng_price"].pct_change(h).shift(-h)

    # Split into production UP vs DOWN signals
    up_mask = df["prod_predicted"] == 1
    down_mask = df["prod_predicted"] == 0

    print(f"\n  Signal counts: UP={up_mask.sum()}, DOWN={down_mask.sum()}")
    print(f"  Signal ratio:  {up_mask.mean():.1%} UP / {down_mask.mean():.1%} DOWN\n")

    header = f"  {'Horizon':>8s}  {'UP signal':>12s}  {'DOWN signal':>12s}  {'Spread':>10s}  {'t-stat':>8s}  {'p-val':>8s}"
    print(header)
    print("  " + "-" * 68)

    for h in horizons:
        col = f"fwd_ret_{h}d"
        up_rets = df.loc[up_mask, col].dropna()
        dn_rets = df.loc[down_mask, col].dropna()

        up_mean = up_rets.mean()
        dn_mean = dn_rets.mean()
        spread = dn_mean - up_mean  # DOWN signal should be bullish (less supply)

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(dn_rets, up_rets, equal_var=False)

        star = " **" if p_val < 0.01 else " *" if p_val < 0.05 else ""
        print(f"  {h:>5d}d    {up_mean:>+10.2%}    {dn_mean:>+10.2%}    {spread:>+8.2%}    {t_stat:>7.2f}    {p_val:>6.3f}{star}")

    # Quintile analysis by probability
    print("\n  Quintile Analysis: Mean 20d Return by Production Probability Quintile")
    print("  " + "-" * 55)
    df["prob_quintile"] = pd.qcut(df["prod_prob"], 5, labels=False, duplicates="drop")
    for q in sorted(df["prob_quintile"].dropna().unique()):
        mask = df["prob_quintile"] == q
        sub = df.loc[mask]
        prob_range = f"[{sub['prod_prob'].min():.2f}-{sub['prod_prob'].max():.2f}]"
        ret_20 = sub["fwd_ret_20d"].dropna().mean()
        ret_90 = sub["fwd_ret_90d"].dropna().mean()
        n = len(sub)
        print(f"  Q{int(q)}: prob {prob_range:>16s}  ret_20d={ret_20:>+7.2%}  ret_90d={ret_90:>+7.2%}  n={n}")

    # Annual performance of a simple strategy: short when UP, long when DOWN
    print("\n  Hypothetical: Long when prod DOWN signal, Short when prod UP")
    print("  " + "-" * 55)
    df["signal_return_20d"] = np.where(
        df["prod_predicted"] == 0,
        df["fwd_ret_20d"],
        -df["fwd_ret_20d"],
    )
    yearly = df.groupby(df.index.year)["signal_return_20d"].mean()
    for year, ret in yearly.items():
        print(f"  {year}: {ret:>+7.2%} avg 20d return")
    print(f"  Overall: {df['signal_return_20d'].dropna().mean():>+7.2%} avg")


def ablation_test(df: pd.DataFrame) -> None:
    """Walk-forward ablation: price direction with/without production features."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: Ablation - Price Direction Model +/- Production Signal")
    print("=" * 70)

    # Build features
    price = df["ng_price"].copy()

    features = pd.DataFrame(index=df.index)

    # Base price features (technical)
    for lag in [1, 2, 3, 5, 10, 20]:
        features[f"ret_{lag}d"] = price.pct_change(lag)
    for win in [5, 10, 20, 60]:
        features[f"ma_ratio_{win}d"] = price / price.rolling(win).mean()
        features[f"vol_{win}d"] = price.pct_change().rolling(win).std()
    features["rsi_14"] = _rsi(price, 14)
    features["month"] = df.index.month
    features["day_of_year"] = df.index.dayofyear
    doy_frac = features["day_of_year"] / 365.25
    features["doy_sin"] = np.sin(2 * np.pi * doy_frac)
    features["doy_cos"] = np.cos(2 * np.pi * doy_frac)

    # Production forecast features (the signal we're testing)
    prod_features = pd.DataFrame(index=df.index)
    prod_features["prod_prob"] = df["prod_prob"]
    prod_features["prod_signal"] = df["prod_predicted"]
    prod_features["prod_prob_5d"] = df["prod_prob"].rolling(5).mean()
    prod_features["prod_prob_20d"] = df["prod_prob"].rolling(20).mean()
    prod_features["prod_prob_change"] = df["prod_prob"].diff(5)
    # Supply pressure: high prob of increase = bearish
    prod_features["supply_pressure"] = -(df["prod_prob"] - 0.5) * 2

    # Target: 20-day forward price direction
    horizon = 20
    fwd_ret = price.pct_change(horizon).shift(-horizon)
    target = (fwd_ret > 0).astype(int)

    # Base features only
    base_X = features.copy()
    # Enhanced = base + production
    enh_X = pd.concat([features, prod_features], axis=1)

    # Drop rows with NaN target
    valid = target.notna() & base_X.notna().all(axis=1) & enh_X.notna().all(axis=1)
    base_X = base_X[valid]
    enh_X = enh_X[valid]
    y = target[valid]

    logger.info("Ablation data: %d rows, base=%d features, enhanced=%d features",
                len(y), base_X.shape[1], enh_X.shape[1])
    print(f"\n  Data: {len(y)} rows | Base: {base_X.shape[1]} features | Enhanced: {enh_X.shape[1]} features")
    print(f"  Target: 20d price direction | Base rate: {y.mean():.1%}\n")

    # Walk-forward CV
    min_train = 500
    step = 250
    n = len(y)

    base_results = []
    enh_results = []

    for split_idx in range(min_train, n - step, step):
        train_end = split_idx
        test_end = min(split_idx + step, n)

        test_start_date = y.index[train_end].strftime("%Y-%m-%d")
        test_end_date = y.index[test_end - 1].strftime("%Y-%m-%d")

        for name, X_all, results_list in [
            ("Base", base_X, base_results),
            ("Enhanced", enh_X, enh_results),
        ]:
            X_tr, y_tr = X_all.iloc[:train_end], y.iloc[:train_end]
            X_te, y_te = X_all.iloc[train_end:test_end], y.iloc[train_end:test_end]

            if y_tr.nunique() < 2:
                continue

            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=10,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                verbosity=0,
                eval_metric="logloss",
            )
            model.fit(X_tr, y_tr, verbose=False)
            preds = model.predict(X_te)
            probs = model.predict_proba(X_te)[:, 1]

            acc = accuracy_score(y_te, preds)
            f1 = f1_score(y_te, preds, zero_division=0)
            base_rate = y_te.mean()
            try:
                auc = roc_auc_score(y_te, probs)
            except ValueError:
                auc = np.nan

            results_list.append({
                "accuracy": acc,
                "f1": f1,
                "auc": auc,
                "base_rate": base_rate,
                "edge": acc - base_rate,
                "test_start": test_start_date,
                "test_end": test_end_date,
            })

    # Print results
    print(f"  {'Fold':>4s}  {'Period':>25s}  {'Base Acc':>10s}  {'+ Prod Acc':>10s}  {'Delta':>8s}  {'Base Rate':>10s}")
    print("  " + "-" * 75)
    for i, (b, e) in enumerate(zip(base_results, enh_results)):
        delta = e["accuracy"] - b["accuracy"]
        marker = " +" if delta > 0.01 else " -" if delta < -0.01 else "  "
        print(f"  {i+1:>4d}  {b['test_start']} - {b['test_end']}  {b['accuracy']:>8.1%}  {e['accuracy']:>8.1%}  {delta:>+6.1%}{marker}  {b['base_rate']:>8.1%}")

    # Summary
    base_acc = np.mean([r["accuracy"] for r in base_results])
    enh_acc = np.mean([r["accuracy"] for r in enh_results])
    base_f1 = np.mean([r["f1"] for r in base_results])
    enh_f1 = np.mean([r["f1"] for r in enh_results])
    base_auc_vals = [r["auc"] for r in base_results if not np.isnan(r["auc"])]
    enh_auc_vals = [r["auc"] for r in enh_results if not np.isnan(r["auc"])]
    base_auc = np.mean(base_auc_vals) if base_auc_vals else np.nan
    enh_auc = np.mean(enh_auc_vals) if enh_auc_vals else np.nan
    avg_base_rate = np.mean([r["base_rate"] for r in base_results])

    delta_acc = enh_acc - base_acc
    delta_f1 = enh_f1 - base_f1
    delta_auc = enh_auc - base_auc

    print("\n  Summary:")
    print(f"  {'':>20s}  {'Baseline':>10s}  {'+ Production':>12s}  {'Delta':>10s}")
    print("  " + "-" * 60)
    print(f"  {'Accuracy':>20s}  {base_acc:>8.1%}    {enh_acc:>8.1%}      {delta_acc:>+6.1%}")
    print(f"  {'F1 Score':>20s}  {base_f1:>8.3f}    {enh_f1:>8.3f}      {delta_f1:>+6.3f}")
    print(f"  {'AUC':>20s}  {base_auc:>8.3f}    {enh_auc:>8.3f}      {delta_auc:>+6.3f}")
    print(f"  {'Base Rate':>20s}  {avg_base_rate:>8.1%}")

    # Verdict
    print()
    if abs(delta_acc) < 0.005:
        print("  VERDICT: NEGLIGIBLE -- Production signal does not meaningfully change price prediction")
    elif delta_acc > 0.005:
        print(f"  VERDICT: IMPROVES price prediction by {delta_acc:+.1%} accuracy")
    else:
        print(f"  VERDICT: HURTS price prediction by {delta_acc:+.1%} accuracy")

    # Feature importance from last enhanced model
    if enh_results:
        # Retrain on full data to get importances
        valid_enh = enh_X.dropna()
        valid_y = y.loc[valid_enh.index]
        model_full = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
            random_state=42, verbosity=0, eval_metric="logloss",
        )
        model_full.fit(valid_enh, valid_y, verbose=False)
        imp = pd.Series(model_full.feature_importances_, index=valid_enh.columns).sort_values(ascending=False)

        print("\n  Top 15 features (enhanced model):")
        for feat, val in imp.head(15).items():
            tag = " <-- PRODUCTION" if feat.startswith("prod_") or feat == "supply_pressure" else ""
            print(f"    {feat:30s}  {val:.4f}{tag}")

        prod_total = imp[[c for c in imp.index if c.startswith("prod_") or c == "supply_pressure"]].sum()
        print(f"\n  Production feature total importance: {prod_total:.4f} ({prod_total/imp.sum():.1%} of total)")


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.clip(lower=1e-10)
    return 100 - (100 / (1 + rs))


def correlation_analysis(df: pd.DataFrame) -> None:
    """Test correlation between production probability and future returns."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: Correlation & Information Content")
    print("=" * 70)

    horizons = [5, 10, 20, 60, 90]
    print(f"\n  Spearman rank correlation: prod_prob vs forward NG return")
    print("  " + "-" * 50)

    for h in horizons:
        fwd = df["ng_price"].pct_change(h).shift(-h)
        valid = fwd.notna() & df["prod_prob"].notna()
        corr, pval = stats.spearmanr(df.loc[valid, "prod_prob"], fwd[valid])
        star = " **" if pval < 0.01 else " *" if pval < 0.05 else ""
        # Negative correlation expected (higher prob of production increase = bearish)
        print(f"  {h:>5d}d: rho={corr:>+.4f}  p={pval:.4f}{star}")

    # Information content: entropy reduction
    print(f"\n  Conditional entropy: Does knowing prod signal reduce price direction uncertainty?")
    print("  " + "-" * 55)

    for h in [20, 60, 90]:
        fwd_ret = df["ng_price"].pct_change(h).shift(-h)
        direction = (fwd_ret > 0).astype(int)
        valid = direction.notna() & df["prod_predicted"].notna()
        d = direction[valid]
        s = df.loc[valid, "prod_predicted"]

        # Unconditional entropy
        p_up = d.mean()
        h_uncond = -p_up * np.log2(max(p_up, 1e-10)) - (1 - p_up) * np.log2(max(1 - p_up, 1e-10))

        # Conditional entropy H(D|S)
        h_cond = 0
        for sig in [0, 1]:
            mask = s == sig
            if mask.sum() == 0:
                continue
            p_sig = mask.mean()
            p_up_given_sig = d[mask].mean()
            if 0 < p_up_given_sig < 1:
                h_given = -p_up_given_sig * np.log2(p_up_given_sig) - (1 - p_up_given_sig) * np.log2(1 - p_up_given_sig)
            else:
                h_given = 0
            h_cond += p_sig * h_given

        mi = h_uncond - h_cond
        print(f"  {h:>3d}d:  H(dir)={h_uncond:.4f}  H(dir|signal)={h_cond:.4f}  MI={mi:.4f} bits  ({mi/max(h_uncond,1e-10):.1%} reduction)")


def main() -> None:
    df = load_data()

    signal_analysis(df)
    correlation_analysis(df)
    ablation_test(df)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
