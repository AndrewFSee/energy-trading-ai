#!/usr/bin/env python3
"""Deep-dive into WHEN and WHY the 20-day directional edge works.

Breaks down model accuracy by:
1. Walk-forward fold (time period covered)
2. Year-by-year accuracy
3. Market regime (trend, volatility, drawdown)
4. Per-instrument contribution
5. Feature importance from XGBoost
6. Correct vs incorrect predictions: what do they look like?

Uses the same purged walk-forward with embargo=5 as validate_edge.py so
the numbers are directly comparable.
"""

from __future__ import annotations

import copy, sys, logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(level=logging.WARNING)

HORIZON = 20
FOLDS = 5
EMBARGO = 5
VAL_RATIO = 0.15
FEATURES_DIR = Path("data/processed")
INSTRUMENTS = ["wti", "brent", "heating_oil", "rbob_gasoline"]


# ── data loading (same as validate_edge.py) ──────────────────────────
def load_instrument(instrument: str):
    path = FEATURES_DIR / f"features_{instrument}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    feature_cols = [
        c for c in df.columns
        if c not in {"target", "Open", "High", "Low", "Close", "Volume"}
    ]
    close = df["Close"]
    df["target_20d"] = np.log(close.shift(-HORIZON) / close)
    df = df.dropna(subset=["target_20d"])
    df = df.iloc[::HORIZON]  # non-overlapping
    return df, feature_cols


# ── collect per-sample predictions from purged walk-forward ──────────
def collect_predictions(model_cls, df, feature_cols, is_regression=True):
    """Run purged walk-forward and return a DataFrame with predictions
    aligned to the original dates."""
    X = df[feature_cols].fillna(0).values
    y_reg = df["target_20d"].values
    y_cls = (y_reg > 0).astype(int)
    y = y_reg if is_regression else y_cls
    dates = df.index
    close = df["Close"].values

    n = len(X)
    fold_size = n // (FOLDS + 1)

    results = []
    feature_imp_accum = np.zeros(len(feature_cols))
    n_models = 0

    for fold in range(FOLDS):
        train_end = fold_size * (fold + 1)
        test_start = min(train_end + EMBARGO, n)
        test_end = min(test_start + fold_size, n)
        if test_start >= test_end:
            continue

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        y_reg_test = y_reg[test_start:test_end]
        dates_test = dates[test_start:test_end]
        close_test = close[test_start:test_end]

        val_size = max(1, int(len(X_train) * VAL_RATIO))
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

        m = copy.deepcopy(model_cls)
        m.train(X_tr, y_tr, X_val=X_val, y_val=y_val)
        preds = m.predict(X_test)

        # Feature importance
        fi = getattr(m, "feature_importances", None)
        if fi is not None:
            feature_imp_accum += fi
            n_models += 1

        if is_regression:
            pred_dir = np.sign(preds)
            actual_dir = np.sign(y_reg_test)
        else:
            pred_dir = preds
            actual_dir = y_cls[test_start:test_end]

        for i in range(len(y_test)):
            results.append({
                "date": dates_test[i],
                "fold": fold + 1,
                "close": close_test[i],
                "actual_return": y_reg_test[i],
                "actual_dir": int(y_reg_test[i] > 0),
                "pred_raw": float(preds[i]),
                "pred_dir": int(pred_dir[i] > 0) if is_regression else int(pred_dir[i]),
                "correct": int(pred_dir[i] > 0) == int(y_reg_test[i] > 0) if is_regression
                           else int(preds[i]) == int((y_reg_test[i] > 0)),
            })

    res_df = pd.DataFrame(results)
    if len(res_df):
        res_df["date"] = pd.to_datetime(res_df["date"])
        res_df = res_df.set_index("date").sort_index()

    avg_fi = feature_imp_accum / max(n_models, 1)
    return res_df, avg_fi


def main():
    from rich.console import Console
    from rich.table import Table

    console = Console(width=130)
    console.print("[bold green]Deep Dive: What Drives the 20-Day Directional Edge?[/bold green]\n")

    from src.models.xgboost_model import XGBoostModel
    from src.models.linear_model import LogisticModel

    # ── Load all instruments ─────────────────────────────────────────
    instrument_dfs = {}
    for inst in INSTRUMENTS:
        result = load_instrument(inst)
        if result is not None:
            instrument_dfs[inst] = result
            console.print(f"  {inst}: {len(result[0])} samples "
                          f"({result[0].index[0].strftime('%Y-%m')} → "
                          f"{result[0].index[-1].strftime('%Y-%m')})")

    # ── Pool data (preserving dates) ─────────────────────────────────
    all_dfs = []
    for inst, (df, fcols) in instrument_dfs.items():
        tmp = df.copy()
        tmp["instrument"] = inst
        all_dfs.append(tmp)
    pooled_df = pd.concat(all_dfs).sort_index()
    # Use feature columns from WTI (all the same)
    feature_cols = instrument_dfs["wti"][1]
    console.print(f"  [bold]Pooled: {len(pooled_df)} samples[/bold]\n")

    # =================================================================
    # PART A: XGBoost regression — per-fold & per-year on POOLED data
    # =================================================================
    console.print("[bold cyan]═══ A. XGBoost Regression (pooled, purged WF) ═══[/bold cyan]\n")
    xgb_preds, xgb_fi = collect_predictions(XGBoostModel(), pooled_df, feature_cols, is_regression=True)

    # ── A1. Per-fold breakdown ───────────────────────────────────────
    console.print("[bold]A1. Accuracy by walk-forward fold[/bold]")
    fold_tbl = Table()
    fold_tbl.add_column("Fold", style="cyan")
    fold_tbl.add_column("Period", style="dim")
    fold_tbl.add_column("N samples")
    fold_tbl.add_column("Dir. Acc", style="green")
    fold_tbl.add_column("Avg |return|", style="dim")
    fold_tbl.add_column("Up %", style="dim")

    for fold in sorted(xgb_preds["fold"].unique()):
        sub = xgb_preds[xgb_preds["fold"] == fold]
        period = f"{sub.index[0].strftime('%Y-%m')} → {sub.index[-1].strftime('%Y-%m')}"
        acc = sub["correct"].mean()
        avg_ret = sub["actual_return"].abs().mean()
        up_pct = sub["actual_dir"].mean() * 100
        fold_tbl.add_row(
            str(fold), period, str(len(sub)),
            f"{acc:.3f}", f"{avg_ret:.4f}", f"{up_pct:.1f}%"
        )
    console.print(fold_tbl)
    console.print()

    # ── A2. Year-by-year accuracy ────────────────────────────────────
    console.print("[bold]A2. Accuracy by year[/bold]")
    xgb_preds["year"] = xgb_preds.index.year
    year_tbl = Table()
    year_tbl.add_column("Year", style="cyan")
    year_tbl.add_column("N", style="dim")
    year_tbl.add_column("Dir. Acc", style="green")
    year_tbl.add_column("Up%", style="dim")
    year_tbl.add_column("Avg Ret", style="dim")
    year_tbl.add_column("Regime", style="yellow")

    for year in sorted(xgb_preds["year"].unique()):
        sub = xgb_preds[xgb_preds["year"] == year]
        acc = sub["correct"].mean()
        up = sub["actual_dir"].mean() * 100
        avg_ret = sub["actual_return"].mean()
        # Simple regime tag
        if avg_ret > 0.03:
            regime = "strong bull"
        elif avg_ret > 0.005:
            regime = "bull"
        elif avg_ret > -0.005:
            regime = "sideways"
        elif avg_ret > -0.03:
            regime = "bear"
        else:
            regime = "strong bear"
        year_tbl.add_row(
            str(year), str(len(sub)), f"{acc:.3f}",
            f"{up:.0f}%", f"{avg_ret:+.4f}", regime
        )
    console.print(year_tbl)
    console.print()

    # ── A3. Regime-based accuracy ────────────────────────────────────
    console.print("[bold]A3. Accuracy by market regime[/bold]")
    # Tag each sample with volatility and trend regimes
    abs_rets = xgb_preds["actual_return"].abs()
    vol_median = abs_rets.median()
    xgb_preds["high_vol"] = abs_rets > vol_median
    xgb_preds["big_move"] = abs_rets > abs_rets.quantile(0.75)

    regime_tbl = Table()
    regime_tbl.add_column("Regime Filter", style="cyan")
    regime_tbl.add_column("N", style="dim")
    regime_tbl.add_column("Dir. Acc", style="green")
    regime_tbl.add_column("vs Overall", style="red")

    overall = xgb_preds["correct"].mean()
    slices = {
        "All samples": xgb_preds,
        "High volatility (|ret| > median)": xgb_preds[xgb_preds["high_vol"]],
        "Low volatility (|ret| ≤ median)": xgb_preds[~xgb_preds["high_vol"]],
        "Big moves (|ret| > P75)": xgb_preds[xgb_preds["big_move"]],
        "Small moves (|ret| ≤ P25)": xgb_preds[abs_rets <= abs_rets.quantile(0.25)],
        "Up periods (actual up)": xgb_preds[xgb_preds["actual_dir"] == 1],
        "Down periods (actual down)": xgb_preds[xgb_preds["actual_dir"] == 0],
        "Extreme up (ret > P90)": xgb_preds[xgb_preds["actual_return"] > xgb_preds["actual_return"].quantile(0.9)],
        "Extreme down (ret < P10)": xgb_preds[xgb_preds["actual_return"] < xgb_preds["actual_return"].quantile(0.1)],
    }
    for label, sub in slices.items():
        if len(sub) == 0:
            continue
        acc = sub["correct"].mean()
        regime_tbl.add_row(label, str(len(sub)), f"{acc:.3f}", f"{acc - overall:+.3f}")
    console.print(regime_tbl)
    console.print()

    # ── A4. Per-instrument contribution ──────────────────────────────
    console.print("[bold]A4. Accuracy by instrument (within pooled model)[/bold]")
    inst_tbl = Table()
    inst_tbl.add_column("Instrument", style="cyan")
    inst_tbl.add_column("N (in test)", style="dim")
    inst_tbl.add_column("Dir. Acc", style="green")
    inst_tbl.add_column("vs Overall", style="red")

    # Map samples back to instruments — the pooled_df was sorted by date
    # so we need to track instrument labels through the walk-forward
    xgb_preds_inst, _ = collect_predictions_with_instrument(
        XGBoostModel(), pooled_df, feature_cols, is_regression=True
    )
    if xgb_preds_inst is not None and "instrument" in xgb_preds_inst.columns:
        for inst in INSTRUMENTS:
            sub = xgb_preds_inst[xgb_preds_inst["instrument"] == inst]
            if len(sub) == 0:
                continue
            acc = sub["correct"].mean()
            inst_tbl.add_row(inst, str(len(sub)), f"{acc:.3f}", f"{acc - overall:+.3f}")
        console.print(inst_tbl)
    console.print()

    # ── A5. Top XGBoost feature importances ──────────────────────────
    console.print("[bold]A5. Top 20 XGBoost feature importances (avg across folds)[/bold]")
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": xgb_fi})
    fi_df = fi_df.sort_values("importance", ascending=False).head(20)
    fi_tbl = Table()
    fi_tbl.add_column("#", style="dim")
    fi_tbl.add_column("Feature", style="cyan")
    fi_tbl.add_column("Importance", style="green")
    fi_tbl.add_column("Bar", style="yellow")
    max_imp = fi_df["importance"].max()
    for rank, (_, row) in enumerate(fi_df.iterrows(), 1):
        bar_len = int(30 * row["importance"] / max_imp) if max_imp > 0 else 0
        fi_tbl.add_row(str(rank), row["feature"], f"{row['importance']:.4f}", "█" * bar_len)
    console.print(fi_tbl)
    console.print()

    # ── A6. Prediction confidence vs correctness ─────────────────────
    console.print("[bold]A6. XGBoost prediction magnitude vs accuracy[/bold]")
    abs_pred = xgb_preds["pred_raw"].abs()
    quantiles = [0, 0.25, 0.50, 0.75, 0.90, 1.0]
    cuts = abs_pred.quantile(quantiles).values
    conf_tbl = Table()
    conf_tbl.add_column("|pred| range", style="cyan")
    conf_tbl.add_column("N", style="dim")
    conf_tbl.add_column("Dir. Acc", style="green")
    conf_tbl.add_column("Avg actual |ret|", style="dim")
    for i in range(len(cuts) - 1):
        mask = (abs_pred >= cuts[i]) & (abs_pred < cuts[i + 1]) if i < len(cuts) - 2 \
            else (abs_pred >= cuts[i])
        sub = xgb_preds[mask]
        if len(sub) == 0:
            continue
        conf_tbl.add_row(
            f"{cuts[i]:.4f} – {cuts[i+1]:.4f}",
            str(len(sub)),
            f"{sub['correct'].mean():.3f}",
            f"{sub['actual_return'].abs().mean():.4f}",
        )
    console.print(conf_tbl)
    console.print()

    # ── A7. Worst misses — largest wrong predictions ─────────────────
    console.print("[bold]A7. Biggest wrong calls (XGBoost)[/bold]")
    wrong = xgb_preds[~xgb_preds["correct"]].copy()
    wrong["miss_size"] = wrong["actual_return"].abs()
    worst = wrong.nlargest(15, "miss_size")
    miss_tbl = Table()
    miss_tbl.add_column("Date", style="cyan")
    miss_tbl.add_column("Actual Ret", style="red")
    miss_tbl.add_column("Pred Raw", style="dim")
    miss_tbl.add_column("Close", style="dim")
    miss_tbl.add_column("Fold", style="dim")
    for _, row in worst.iterrows():
        miss_tbl.add_row(
            row.name.strftime("%Y-%m-%d"),
            f"{row['actual_return']:+.4f}",
            f"{row['pred_raw']:+.4f}",
            f"{row['close']:.2f}",
            str(int(row["fold"])),
        )
    console.print(miss_tbl)
    console.print()

    # =================================================================
    # PART B: Logistic Classification — same breakdowns (abbreviated)
    # =================================================================
    console.print("[bold cyan]═══ B. Logistic Classification (pooled, purged WF) ═══[/bold cyan]\n")
    log_preds, _ = collect_predictions(LogisticModel(), pooled_df, feature_cols, is_regression=False)

    console.print("[bold]B1. Accuracy by fold[/bold]")
    log_fold_tbl = Table()
    log_fold_tbl.add_column("Fold", style="cyan")
    log_fold_tbl.add_column("Period", style="dim")
    log_fold_tbl.add_column("N")
    log_fold_tbl.add_column("Accuracy", style="green")
    for fold in sorted(log_preds["fold"].unique()):
        sub = log_preds[log_preds["fold"] == fold]
        period = f"{sub.index[0].strftime('%Y-%m')} → {sub.index[-1].strftime('%Y-%m')}"
        log_fold_tbl.add_row(str(fold), period, str(len(sub)), f"{sub['correct'].mean():.3f}")
    console.print(log_fold_tbl)
    console.print()

    console.print("[bold]B2. Year-by-year accuracy[/bold]")
    log_preds["year"] = log_preds.index.year
    log_yr_tbl = Table()
    log_yr_tbl.add_column("Year", style="cyan")
    log_yr_tbl.add_column("N", style="dim")
    log_yr_tbl.add_column("Accuracy", style="green")
    for year in sorted(log_preds["year"].unique()):
        sub = log_preds[log_preds["year"] == year]
        log_yr_tbl.add_row(str(year), str(len(sub)), f"{sub['correct'].mean():.3f}")
    console.print(log_yr_tbl)
    console.print()

    # ── C. Compare XGB vs Logistic agreement ─────────────────────────
    console.print("[bold cyan]═══ C. Model Agreement Analysis ═══[/bold cyan]\n")
    # Align on dates present in both
    both = xgb_preds[["pred_dir", "correct"]].rename(
        columns={"pred_dir": "xgb_dir", "correct": "xgb_correct"}
    ).join(
        log_preds[["pred_dir", "correct"]].rename(
            columns={"pred_dir": "log_dir", "correct": "log_correct"}
        ),
        how="inner",
    )
    agree = both["xgb_dir"] == both["log_dir"]
    both_right = both["xgb_correct"] & both["log_correct"]
    both_wrong = ~both["xgb_correct"] & ~both["log_correct"]

    agree_tbl = Table(title="Model Agreement")
    agree_tbl.add_column("Scenario", style="cyan")
    agree_tbl.add_column("N", style="dim")
    agree_tbl.add_column("% of total", style="green")
    agree_tbl.add_column("Combined accuracy (where applicable)", style="yellow")

    n_total = len(both)
    n_agree = agree.sum()
    agree_sub = both[agree]
    disagree_sub = both[~agree]

    agree_tbl.add_row("Both agree", str(n_agree), f"{n_agree/n_total:.1%}",
                       f"{both_right[agree].mean():.3f} (when agree)")
    agree_tbl.add_row("They disagree", str(len(disagree_sub)), f"{len(disagree_sub)/n_total:.1%}", "—")
    agree_tbl.add_row("Both correct", str(int(both_right.sum())), f"{both_right.mean():.1%}", "—")
    agree_tbl.add_row("Both wrong", str(int(both_wrong.sum())), f"{both_wrong.mean():.1%}", "—")

    # When both agree, what's accuracy?
    if len(agree_sub) > 0:
        agree_acc = agree_sub["xgb_correct"].mean()  # same as log_correct when they agree
        agree_tbl.add_row("Accuracy when agree", "—", "—", f"{agree_acc:.3f}")
    console.print(agree_tbl)
    console.print()

    # ── D. Rolling accuracy (trailing 3-year window) ─────────────────
    console.print("[bold cyan]═══ D. Rolling accuracy (trailing ~60 samples) ═══[/bold cyan]\n")
    window = 60  # ~3 years of 20-day samples
    if len(xgb_preds) >= window:
        xgb_preds["rolling_acc"] = xgb_preds["correct"].rolling(window, min_periods=window).mean()
        roll_df = xgb_preds.dropna(subset=["rolling_acc"])
        if len(roll_df) > 0:
            roll_tbl = Table(title=f"Rolling {window}-sample XGB dir. accuracy")
            roll_tbl.add_column("Date", style="cyan")
            roll_tbl.add_column("Rolling Acc", style="green")
            # Show every ~40 rows for a readable summary
            step = max(1, len(roll_df) // 20)
            for i in range(0, len(roll_df), step):
                row = roll_df.iloc[i]
                roll_tbl.add_row(row.name.strftime("%Y-%m-%d"), f"{row['rolling_acc']:.3f}")
            # Always show last
            last = roll_df.iloc[-1]
            roll_tbl.add_row(last.name.strftime("%Y-%m-%d") + " (latest)", f"{last['rolling_acc']:.3f}")
            console.print(roll_tbl)
    console.print()

    console.print("[bold green]Analysis complete![/bold green]")


# ── Helper: predictions with instrument label ────────────────────────
def collect_predictions_with_instrument(model_cls, df, feature_cols, is_regression=True):
    """Same as collect_predictions but also tracks the instrument column."""
    X = df[feature_cols].fillna(0).values
    y_reg = df["target_20d"].values
    y_cls = (y_reg > 0).astype(int)
    y = y_reg if is_regression else y_cls
    dates = df.index
    instruments = df["instrument"].values if "instrument" in df.columns else None

    n = len(X)
    fold_size = n // (FOLDS + 1)

    results = []
    for fold in range(FOLDS):
        train_end = fold_size * (fold + 1)
        test_start = min(train_end + EMBARGO, n)
        test_end = min(test_start + fold_size, n)
        if test_start >= test_end:
            continue

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        y_reg_test = y_reg[test_start:test_end]
        dates_test = dates[test_start:test_end]
        inst_test = instruments[test_start:test_end] if instruments is not None else [None] * (test_end - test_start)

        val_size = max(1, int(len(X_train) * VAL_RATIO))
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

        m = copy.deepcopy(model_cls)
        m.train(X_tr, y_tr, X_val=X_val, y_val=y_val)
        preds = m.predict(X_test)

        if is_regression:
            pred_dir = np.sign(preds)
        else:
            pred_dir = preds

        for i in range(len(y_test)):
            correct = (int(pred_dir[i] > 0) == int(y_reg_test[i] > 0)) if is_regression \
                else (int(preds[i]) == int(y_reg_test[i] > 0))
            results.append({
                "date": dates_test[i],
                "instrument": inst_test[i],
                "fold": fold + 1,
                "actual_return": y_reg_test[i],
                "pred_raw": float(preds[i]),
                "pred_dir": int(pred_dir[i] > 0) if is_regression else int(pred_dir[i]),
                "correct": correct,
            })

    res_df = pd.DataFrame(results)
    if len(res_df):
        res_df["date"] = pd.to_datetime(res_df["date"])
        res_df = res_df.set_index("date").sort_index()
    return res_df, None


if __name__ == "__main__":
    main()
