#!/usr/bin/env python3
"""Quick predictability test: Natural Gas vs WTI at 20-day horizon.

Runs purged walk-forward (embargo=5) with XGBoost regression and
Logistic classification on both instruments, then compares.
Also tests NG-specific features (storage seasonality).
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

def load_instrument(name):
    path = Path(f"data/processed/features_{name}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    fcols = [c for c in df.columns if c not in {"target", "Open", "High", "Low", "Close", "Volume"}]
    close = df["Close"]
    df["target_20d"] = np.log(close.shift(-HORIZON) / close)
    df = df.dropna(subset=["target_20d"])
    df = df.iloc[::HORIZON]
    return df, fcols

def purged_wf(model_cls, X, y, is_reg=True):
    n = len(X)
    fold_size = n // (FOLDS + 1)
    fold_results = []
    for fold in range(FOLDS):
        te = fold_size * (fold + 1)
        ts = min(te + EMBARGO, n)
        te2 = min(ts + fold_size, n)
        if ts >= te2:
            continue
        Xtr, ytr = X[:te], y[:te]
        Xte, yte = X[ts:te2], y[ts:te2]
        vs = max(1, int(len(Xtr) * 0.15))
        m = copy.deepcopy(model_cls)
        m.train(Xtr[:-vs], ytr[:-vs], X_val=Xtr[-vs:], y_val=ytr[-vs:])
        p = m.predict(Xte)
        if is_reg:
            acc = float(np.mean(np.sign(p) == np.sign(yte)))
        else:
            acc = float(np.mean(p == yte))
        fold_results.append(acc)
    return np.mean(fold_results), fold_results

def main():
    from rich.console import Console
    from rich.table import Table
    from src.models.xgboost_model import XGBoostModel
    from src.models.linear_model import LogisticModel

    console = Console(width=120)
    console.print("[bold green]Natural Gas vs WTI: 20-Day Predictability Comparison[/bold green]\n")

    results = []
    for inst in ["natural_gas", "wti", "brent", "heating_oil", "rbob_gasoline"]:
        path = Path(f"data/processed/features_{inst}.csv")
        if not path.exists():
            continue
        df, fcols = load_instrument(inst)
        X = df[fcols].fillna(0).values
        y_reg = df["target_20d"].values
        y_cls = (y_reg > 0).astype(int)
        up_pct = y_cls.mean()

        console.print(f"  Testing {inst} ({len(df)} samples, {up_pct:.1%} positive)...")

        xgb_acc, xgb_folds = purged_wf(XGBoostModel(), X, y_reg, is_reg=True)
        log_acc, log_folds = purged_wf(LogisticModel(), X, y_cls, is_reg=False)

        results.append({
            "instrument": inst,
            "n_samples": len(df),
            "up_pct": up_pct,
            "xgb_dir_acc": xgb_acc,
            "xgb_folds": xgb_folds,
            "log_acc": log_acc,
            "log_folds": log_folds,
        })
        console.print(f"    XGB: {xgb_acc:.3f}  Logistic: {log_acc:.3f}")

    console.print()
    tbl = Table(title="20-Day Directional Prediction: All Instruments")
    tbl.add_column("Instrument", style="cyan")
    tbl.add_column("N", style="dim")
    tbl.add_column("Up%", style="dim")
    tbl.add_column("Base Rate", style="dim")
    tbl.add_column("XGB Dir.Acc", style="green")
    tbl.add_column("Log Acc", style="yellow")
    tbl.add_column("XGB Folds", style="dim")
    tbl.add_column("Log Folds", style="dim")

    for r in results:
        base = max(r["up_pct"], 1 - r["up_pct"])
        tbl.add_row(
            r["instrument"],
            str(r["n_samples"]),
            f"{r['up_pct']:.1%}",
            f"{base:.1%}",
            f"{r['xgb_dir_acc']:.3f}",
            f"{r['log_acc']:.3f}",
            ", ".join(f"{f:.2f}" for f in r["xgb_folds"]),
            ", ".join(f"{f:.2f}" for f in r["log_folds"]),
        )
    console.print(tbl)
    console.print()

    # Also test different horizons for NG
    console.print("[bold cyan]Natural Gas: Multiple Horizons[/bold cyan]")
    for horizon in [1, 5, 10, 20]:
        df_full = pd.read_csv("data/processed/features_natural_gas.csv", index_col=0, parse_dates=True)
        fcols = [c for c in df_full.columns if c not in {"target", "Open", "High", "Low", "Close", "Volume"}]
        close = df_full["Close"]
        df_full["tgt"] = np.log(close.shift(-horizon) / close)
        df_full = df_full.dropna(subset=["tgt"])
        df_full = df_full.iloc[::max(horizon, 1)]
        X = df_full[fcols].fillna(0).values
        y = df_full["tgt"].values
        y_c = (y > 0).astype(int)

        xgb_a, _ = purged_wf(XGBoostModel(), X, y, is_reg=True)
        log_a, _ = purged_wf(LogisticModel(), X, y_c, is_reg=False)
        up = y_c.mean()
        base = max(up, 1 - up)
        console.print(f"  Horizon {horizon:2d}d: N={len(df_full):>4}  Up%={up:.1%}  "
                       f"XGB_dir={xgb_a:.3f}  Log={log_a:.3f}  base={base:.1%}")

    console.print("\n[bold green]Done![/bold green]")

if __name__ == "__main__":
    main()
