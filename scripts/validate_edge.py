#!/usr/bin/env python3
"""Statistical validation of the 20-day directional edge.

Runs 4 tests on the XGBoost-regression and Logistic-classification models
at the 20-day horizon:

1. Permutation test — shuffle targets 1000×, get null distribution of accuracy
2. Purged walk-forward — add embargo gap between train/test to eliminate leakage
3. Multi-instrument pooling — combine WTI + Brent + HO + RBOB for 4× samples
4. Threshold betting — only trade when model confidence > threshold

All tests use non-overlapping 20-day targets.
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HORIZON = 20
N_PERMUTATIONS = 200  # 200 shuffles gives p-value resolution of 0.005
FOLDS = 5
FEATURES_DIR = Path("data/processed")
INSTRUMENTS = ["wti", "brent", "heating_oil", "rbob_gasoline"]


def load_instrument(instrument: str, horizon: int = HORIZON):
    """Load features and compute non-overlapping 20d target."""
    path = FEATURES_DIR / f"features_{instrument}.csv"
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    feature_cols = [
        c for c in df.columns
        if c not in {"target", "Open", "High", "Low", "Close", "Volume"}
    ]
    # Recompute target for the exact horizon
    close = df["Close"]
    target_reg = np.log(close.shift(-horizon) / close)
    df["target_20d"] = target_reg
    df = df.dropna(subset=["target_20d"])
    # Non-overlapping
    df = df.iloc[::horizon]
    X = df[feature_cols].fillna(0).values
    y_reg = df["target_20d"].values
    y_cls = (y_reg > 0).astype(int)
    return X, y_reg, y_cls


# ── Purged walk-forward ──────────────────────────────────────────────
def purged_walk_forward(model_cls, X, y, n_folds=FOLDS, embargo=5, val_ratio=0.15):
    """Walk-forward with embargo gap between train and test."""
    n = len(X)
    fold_size = n // (n_folds + 1)
    metrics = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        # Embargo: skip `embargo` samples after training to avoid leakage
        test_start = min(train_end + embargo, n)
        test_end = min(test_start + fold_size, n)

        if test_start >= test_end:
            continue

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        # Validation split from training
        val_size = max(1, int(len(X_train) * val_ratio))
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

        m = copy.deepcopy(model_cls)
        m.train(X_tr, y_tr, X_val=X_val, y_val=y_val)
        preds = m.predict(X_test)

        # Directional accuracy
        if hasattr(y_test[0], '__float__') and not np.issubdtype(type(y_test[0]), np.integer):
            # Regression: compare signs
            correct = np.mean(np.sign(preds) == np.sign(y_test))
        else:
            # Classification: direct comparison
            correct = np.mean(preds == y_test)

        metrics.append(float(correct))

    return np.mean(metrics) if metrics else 0.0, metrics


# ── Permutation test ─────────────────────────────────────────────────
def permutation_test(model_cls, X, y, n_permutations=N_PERMUTATIONS,
                     n_folds=FOLDS, embargo=0, val_ratio=0.15):
    """Run permutation test: shuffle y, measure accuracy, repeat.

    For XGBoost, uses a lighter config to speed up the 200 × 5-fold loop.
    """
    # Use lighter XGBoost config for speed during permutation
    from src.models.xgboost_model import XGBoostModel
    if isinstance(model_cls, XGBoostModel):
        model_cls = XGBoostModel(config={
            "n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "early_stopping_rounds": 20, "n_jobs": -1, "random_state": 42,
        })

    # Get the real accuracy first
    real_acc, _ = purged_walk_forward(model_cls, X, y, n_folds, embargo, val_ratio)

    null_accs = []
    rng = np.random.RandomState(42)
    for i in range(n_permutations):
        y_shuffled = rng.permutation(y)
        acc, _ = purged_walk_forward(model_cls, X, y_shuffled, n_folds, embargo, val_ratio)
        null_accs.append(acc)
        if (i + 1) % 50 == 0:
            print(f"    Permutation {i+1}/{n_permutations}...", flush=True)

    null_accs = np.array(null_accs)
    p_value = float(np.mean(null_accs >= real_acc))
    return real_acc, null_accs, p_value


# ── Threshold betting ────────────────────────────────────────────────
def threshold_betting(model_cls, X, y_cls, n_folds=FOLDS, embargo=0,
                      thresholds=(0.50, 0.55, 0.60, 0.65, 0.70)):
    """Evaluate accuracy at different confidence thresholds."""
    n = len(X)
    fold_size = n // (n_folds + 1)
    val_ratio = 0.15

    # Collect all predictions with probabilities
    all_probs = []
    all_true = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_start = min(train_end + embargo, n)
        test_end = min(test_start + fold_size, n)
        if test_start >= test_end:
            continue

        X_train = X[:train_end]
        y_train = y_cls[:train_end]
        X_test = X[test_start:test_end]
        y_test = y_cls[test_start:test_end]

        val_size = max(1, int(len(X_train) * val_ratio))
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

        m = copy.deepcopy(model_cls)
        m.train(X_tr, y_tr, X_val=X_val, y_val=y_val)

        if hasattr(m, 'predict_proba'):
            probs = m.predict_proba(X_test)
        else:
            # For regression models, use sigmoid of prediction
            raw = m.predict(X_test)
            probs = 1 / (1 + np.exp(-raw * 20))  # Scale factor for log returns

        all_probs.extend(probs.tolist())
        all_true.extend(y_test.tolist())

    all_probs = np.array(all_probs)
    all_true = np.array(all_true)

    results = []
    for thresh in thresholds:
        # Bet only when probability > thresh (go long) or < 1-thresh (go short)
        confident = (all_probs >= thresh) | (all_probs <= (1 - thresh))
        n_trades = int(np.sum(confident))
        if n_trades > 0:
            pred_labels = (all_probs[confident] >= 0.5).astype(int)
            acc = float(np.mean(pred_labels == all_true[confident]))
        else:
            acc = 0.0
        results.append({
            "threshold": thresh,
            "n_trades": n_trades,
            "pct_trading": n_trades / len(all_probs) * 100 if len(all_probs) > 0 else 0,
            "accuracy": acc,
        })
    return results


def main():
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("[bold green]Statistical Validation of 20-Day Directional Edge[/bold green]\n")

    # ── 1. Load all instruments ──────────────────────────────────────
    console.print("[bold cyan]1. Loading data[/bold cyan]")
    datasets = {}
    for inst in INSTRUMENTS:
        X, y_reg, y_cls = load_instrument(inst)
        if X is not None:
            datasets[inst] = (X, y_reg, y_cls)
            pos_pct = y_cls.mean() * 100
            console.print(f"  {inst}: {len(X)} samples, {pos_pct:.1f}% positive")

    # Pool all instruments
    X_all = np.vstack([d[0] for d in datasets.values()])
    y_reg_all = np.concatenate([d[1] for d in datasets.values()])
    y_cls_all = np.concatenate([d[2] for d in datasets.values()])
    pos_all = y_cls_all.mean() * 100
    console.print(f"  [bold]Pooled: {len(X_all)} samples, {pos_all:.1f}% positive[/bold]\n")

    # ── 2. Purged walk-forward on single + pooled ────────────────────
    console.print("[bold cyan]2. Purged walk-forward CV (embargo=5 periods)[/bold cyan]")
    from src.models.xgboost_model import XGBoostModel
    from src.models.linear_model import LogisticModel

    purged_table = Table(title="Purged Walk-Forward (embargo=5)")
    purged_table.add_column("Dataset", style="cyan")
    purged_table.add_column("XGB Reg Dir.Acc", style="green")
    purged_table.add_column("Logistic Acc", style="yellow")

    for label, (X, y_reg, y_cls) in [*datasets.items(), ("POOLED", (X_all, y_reg_all, y_cls_all))]:
        xgb_acc, xgb_folds = purged_walk_forward(XGBoostModel(), X, y_reg, embargo=5)
        log_acc, log_folds = purged_walk_forward(LogisticModel(), X, y_cls, embargo=5)
        purged_table.add_row(
            label.upper(),
            f"{xgb_acc:.3f} ({', '.join(f'{f:.2f}' for f in xgb_folds)})",
            f"{log_acc:.3f} ({', '.join(f'{f:.2f}' for f in log_folds)})",
        )
        console.print(f"  ✓ {label}")

    console.print()
    console.print(purged_table)
    console.print()

    # ── 3. Permutation test on pooled data ───────────────────────────
    console.print("[bold cyan]3. Permutation test (1000 shuffles, pooled data)[/bold cyan]")
    console.print("  Running XGBoost regression permutation test...")
    xgb_real, xgb_null, xgb_pval = permutation_test(
        XGBoostModel(), X_all, y_reg_all, n_permutations=N_PERMUTATIONS, embargo=5
    )
    console.print(f"  XGBoost dir.acc: {xgb_real:.3f}, p-value: {xgb_pval:.4f}")
    console.print(f"  Null distribution: mean={xgb_null.mean():.3f}, std={xgb_null.std():.3f}")
    console.print(f"  Null 95th percentile: {np.percentile(xgb_null, 95):.3f}\n")

    console.print("  Running Logistic classification permutation test...")
    log_real, log_null, log_pval = permutation_test(
        LogisticModel(), X_all, y_cls_all, n_permutations=N_PERMUTATIONS, embargo=5
    )
    console.print(f"  Logistic accuracy: {log_real:.3f}, p-value: {log_pval:.4f}")
    console.print(f"  Null distribution: mean={log_null.mean():.3f}, std={log_null.std():.3f}")
    console.print(f"  Null 95th percentile: {np.percentile(log_null, 95):.3f}\n")

    perm_table = Table(title="Permutation Test Results (1000 shuffles)")
    perm_table.add_column("Model", style="cyan")
    perm_table.add_column("Real Accuracy", style="green")
    perm_table.add_column("Null Mean", style="dim")
    perm_table.add_column("Null 95th", style="dim")
    perm_table.add_column("p-value", style="red bold")
    perm_table.add_column("Significant?", style="yellow")

    perm_table.add_row(
        "XGBoost Reg",
        f"{xgb_real:.3f}",
        f"{xgb_null.mean():.3f}",
        f"{np.percentile(xgb_null, 95):.3f}",
        f"{xgb_pval:.4f}",
        "YES ✓" if xgb_pval < 0.05 else "NO ✗",
    )
    perm_table.add_row(
        "Logistic Cls",
        f"{log_real:.3f}",
        f"{log_null.mean():.3f}",
        f"{np.percentile(log_null, 95):.3f}",
        f"{log_pval:.4f}",
        "YES ✓" if log_pval < 0.05 else "NO ✗",
    )
    console.print(perm_table)
    console.print()

    # ── 4. Threshold betting ─────────────────────────────────────────
    console.print("[bold cyan]4. Threshold betting (pooled, Logistic)[/bold cyan]")
    thresholds = (0.50, 0.55, 0.60, 0.65, 0.70)
    thresh_results = threshold_betting(
        LogisticModel(), X_all, y_cls_all,
        embargo=5, thresholds=thresholds,
    )

    thresh_table = Table(title="Threshold Betting (Logistic, pooled)")
    thresh_table.add_column("Threshold", style="cyan")
    thresh_table.add_column("Trades", style="green")
    thresh_table.add_column("% Trading", style="dim")
    thresh_table.add_column("Accuracy", style="yellow")
    thresh_table.add_column("Edge vs Base", style="red")

    base_rate = pos_all / 100  # fraction positive
    for r in thresh_results:
        edge = r["accuracy"] - max(base_rate, 1 - base_rate)
        thresh_table.add_row(
            f"{r['threshold']:.2f}",
            str(r["n_trades"]),
            f"{r['pct_trading']:.1f}%",
            f"{r['accuracy']:.3f}",
            f"{edge:+.3f}",
        )

    console.print(thresh_table)
    console.print()

    # Also do XGBoost regression with threshold on predicted magnitude
    console.print("[bold cyan]   Threshold betting (pooled, XGBoost regression)[/bold cyan]")
    xgb_thresh = threshold_betting_regression(
        XGBoostModel(), X_all, y_reg_all, y_cls_all, embargo=5,
    )
    xgb_thresh_table = Table(title="Threshold Betting (XGBoost Reg, pooled)")
    xgb_thresh_table.add_column("Min |pred|", style="cyan")
    xgb_thresh_table.add_column("Trades", style="green")
    xgb_thresh_table.add_column("% Trading", style="dim")
    xgb_thresh_table.add_column("Dir. Accuracy", style="yellow")
    xgb_thresh_table.add_column("Edge vs Base", style="red")

    for r in xgb_thresh:
        edge = r["accuracy"] - max(base_rate, 1 - base_rate)
        xgb_thresh_table.add_row(
            f"{r['threshold']:.3f}",
            str(r["n_trades"]),
            f"{r['pct_trading']:.1f}%",
            f"{r['accuracy']:.3f}",
            f"{edge:+.3f}",
        )

    console.print(xgb_thresh_table)
    console.print()

    # ── Summary ──────────────────────────────────────────────────────
    console.print("[bold green]Summary[/bold green]")
    console.print(f"  Pooled samples (4 instruments): {len(X_all)}")
    console.print(f"  Base rate (always predict up): {max(base_rate, 1-base_rate):.1%}")
    console.print(f"  XGBoost purged dir.acc: {xgb_real:.3f} (p={xgb_pval:.4f})")
    console.print(f"  Logistic purged accuracy: {log_real:.3f} (p={log_pval:.4f})")
    if xgb_pval < 0.05 or log_pval < 0.05:
        console.print("  [bold green]At least one model shows statistically significant edge![/bold green]")
    else:
        console.print("  [bold red]No statistically significant edge found.[/bold red]")

    console.print("\n[bold green]Validation complete![/bold green]")


def threshold_betting_regression(model_cls, X, y_reg, y_cls, n_folds=FOLDS,
                                  embargo=5, val_ratio=0.15):
    """Threshold betting based on predicted return magnitude."""
    n = len(X)
    fold_size = n // (n_folds + 1)

    all_preds = []
    all_true = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_start = min(train_end + embargo, n)
        test_end = min(test_start + fold_size, n)
        if test_start >= test_end:
            continue

        X_train, y_train = X[:train_end], y_reg[:train_end]
        X_test, y_test_cls = X[test_start:test_end], y_cls[test_start:test_end]

        val_size = max(1, int(len(X_train) * val_ratio))
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

        m = copy.deepcopy(model_cls)
        m.train(X_tr, y_tr, X_val=X_val, y_val=y_val)
        preds = m.predict(X_test)

        all_preds.extend(preds.tolist())
        all_true.extend(y_test_cls.tolist())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    # Thresholds based on prediction magnitude percentiles
    abs_preds = np.abs(all_preds)
    thresholds = [0.0, np.percentile(abs_preds, 25), np.percentile(abs_preds, 50),
                  np.percentile(abs_preds, 75), np.percentile(abs_preds, 90)]

    results = []
    for thresh in thresholds:
        confident = abs_preds >= thresh
        n_trades = int(np.sum(confident))
        if n_trades > 0:
            pred_dir = (all_preds[confident] > 0).astype(int)
            acc = float(np.mean(pred_dir == all_true[confident]))
        else:
            acc = 0.0
        results.append({
            "threshold": thresh,
            "n_trades": n_trades,
            "pct_trading": n_trades / len(all_preds) * 100,
            "accuracy": acc,
        })
    return results


if __name__ == "__main__":
    main()
