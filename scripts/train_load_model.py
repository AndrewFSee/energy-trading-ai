#!/usr/bin/env python3
"""Train and evaluate electricity demand forecasting models.

Trains XGBoost and LSTM models on the load feature dataset, evaluates
with walk-forward cross-validation, and saves trained models to disk.

Metrics reported: MAE, RMSE, MAPE (%), R², Peak Error (%).
Expected performance: R² > 0.85, MAPE < 5% (load forecasting is
physically driven and highly predictable from weather + calendar).

Usage:
    python scripts/train_load_model.py [OPTIONS]

Examples:
    # Train both models
    python scripts/train_load_model.py

    # Train only XGBoost with 4 folds
    python scripts/train_load_model.py --model xgboost --folds 4

    # Show feature importance
    python scripts/train_load_model.py --feature-importance
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def walk_forward_load(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    val_ratio: float = 0.15,
) -> list[dict[str, float]]:
    """Walk-forward cross-validation for load models.

    Uses expanding training window with load-specific metrics.

    Args:
        model: Model instance (must have train/predict/evaluate).
        X: Feature matrix.
        y: Target array.
        n_folds: Number of walk-forward splits.
        val_ratio: Fraction of train window for validation.

    Returns:
        List of metric dicts per fold.
    """
    from src.models.load_forecaster import load_forecast_metrics

    n = len(X)
    fold_size = n // (n_folds + 1)
    fold_results = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_end = min(train_end + fold_size, n)

        X_tr_full = X[:train_end]
        y_tr_full = y[:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        # Validation split from end of training window
        val_size = max(1, int(len(X_tr_full) * val_ratio))
        X_val = X_tr_full[-val_size:]
        y_val = y_tr_full[-val_size:]
        X_tr = X_tr_full[:-val_size]
        y_tr = y_tr_full[:-val_size]

        logger.info(
            "Fold %d/%d: train=%d, val=%d, test=%d",
            fold + 1, n_folds, len(X_tr), len(X_val), len(X_test),
        )

        fold_model = copy.deepcopy(model)
        fold_model.train(X_tr, y_tr, X_val=X_val, y_val=y_val)
        preds = fold_model.predict(X_test)

        # Align lengths for LSTM (creates sequences internally)
        y_eval = y_test
        if len(preds) < len(y_test):
            y_eval = y_test[len(y_test) - len(preds):]

        metrics = load_forecast_metrics(y_eval, preds)
        fold_results.append(metrics)
        logger.info("  Fold %d — MAPE: %.2f%%, R²: %.4f", fold + 1, metrics["mape"], metrics["r2"])

    return fold_results


@click.command()
@click.option("--features-file", default="data/processed/load_features.csv",
              help="Path to load feature CSV")
@click.option("--models-dir", default="models", help="Directory to save trained models")
@click.option("--model", default="all", type=click.Choice(["all", "xgboost", "lstm"]))
@click.option("--folds", default=5, help="Walk-forward CV folds")
@click.option("--feature-importance", is_flag=True, help="Show top feature importances")
def main(
    features_file: str,
    models_dir: str,
    model: str,
    folds: int,
    feature_importance: bool,
) -> None:
    """Train and evaluate load forecasting models."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    features_path = Path(features_file)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        console.print(f"[red]Feature file not found: {features_path}[/red]")
        console.print("Run: python scripts/ingest_demand_data.py first")
        return

    console.print("[bold green]Load Forecasting — Model Training[/bold green]\n")

    # Load data
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    target_col = "target"
    if target_col not in df.columns:
        console.print(f"[red]Target column not found in {features_path}[/red]")
        return

    feature_cols = sorted([c for c in df.columns if c != target_col])
    df = df.dropna(subset=[target_col])

    # Drop rows with too many NaN features (early rows with no lags)
    n_feature_nans = df[feature_cols].isna().sum(axis=1)
    df = df[n_feature_nans < len(feature_cols) * 0.3]

    X = df[feature_cols].fillna(0).values
    y = df[target_col].values

    console.print(f"  Dataset: {len(X)} days, {len(feature_cols)} features")
    console.print(f"  Walk-forward folds: {folds}\n")

    # Models to train
    models_to_train = []

    if model in ("all", "xgboost"):
        from src.models.load_forecaster import XGBLoadForecaster
        models_to_train.append(XGBLoadForecaster())

    if model in ("all", "lstm"):
        from src.models.load_forecaster import LSTMLoadForecaster
        models_to_train.append(LSTMLoadForecaster())

    # Results table
    results = Table(title="Walk-Forward Load Forecast Results")
    results.add_column("Model", style="cyan")
    results.add_column("MAE (MWh)", style="green")
    results.add_column("RMSE (MWh)", style="green")
    results.add_column("MAPE (%)", style="yellow", justify="right")
    results.add_column("R²", style="magenta", justify="right")
    results.add_column("Peak Err (%)", style="red", justify="right")

    for m in models_to_train:
        console.print(f"  Training [cyan]{m.name}[/cyan]...")

        fold_metrics = walk_forward_load(m, X, y, n_folds=folds)

        # Aggregate metrics
        metric_keys = fold_metrics[0].keys()
        mean_m = {k: np.mean([f[k] for f in fold_metrics]) for k in metric_keys}
        std_m = {k: np.std([f[k] for f in fold_metrics]) for k in metric_keys}

        results.add_row(
            m.name,
            f"{mean_m['mae']:,.0f} ± {std_m['mae']:,.0f}",
            f"{mean_m['rmse']:,.0f} ± {std_m['rmse']:,.0f}",
            f"{mean_m['mape']:.2f} ± {std_m['mape']:.2f}",
            f"{mean_m['r2']:.4f} ± {std_m['r2']:.4f}",
            f"{mean_m['peak_error_pct']:.1f} ± {std_m['peak_error_pct']:.1f}",
        )

        # Final train on all data and save
        console.print(f"    Final training on all {len(X)} samples...")
        val_size = max(1, int(len(X) * 0.1))
        m.train(X[:-val_size], y[:-val_size], X_val=X[-val_size:], y_val=y[-val_size:])

        ext = ".pt" if "LSTM" in m.name else ".pkl"
        save_path = models_path / f"{m.name.lower().replace('-', '_')}{ext}"
        m.save(save_path)
        console.print(f"    ✓ Saved to {save_path}")

        # Feature importance for XGBoost
        if feature_importance and hasattr(m, "feature_importances") and m.feature_importances is not None:
            console.print(f"\n  [bold]Top 20 Feature Importances ({m.name}):[/bold]")
            imp = Table()
            imp.add_column("Rank", style="dim")
            imp.add_column("Feature", style="cyan")
            imp.add_column("Importance", style="green")
            indices = np.argsort(m.feature_importances)[::-1][:20]
            for rank, idx in enumerate(indices, 1):
                imp.add_row(
                    str(rank),
                    feature_cols[idx],
                    f"{m.feature_importances[idx]:.4f}",
                )
            console.print(imp)

    console.print()
    console.print(results)
    console.print("\n[bold green]Load forecasting training complete![/bold green]")


if __name__ == "__main__":
    main()
