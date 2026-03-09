#!/usr/bin/env python3
"""Two-stage price prediction: demand forecast -> NG price direction.

This script implements the full two-stage pipeline:
  1. Load the trained XGB demand model and generate historical demand
     "backcasts" (what the model WOULD have predicted on each past day).
  2. Build price features WITH and WITHOUT demand forecasts.
  3. Train NG price direction models on both feature sets.
  4. Compare performance via walk-forward cross-validation.

The ablation study directly measures whether demand forecasts add
predictive information for natural gas prices beyond what is available
from technical/calendar/fundamental features alone.

Usage:
    python scripts/train_price_model.py [OPTIONS]

Examples:
    # Full pipeline with ablation
    python scripts/train_price_model.py

    # Quick test with fewer folds
    python scripts/train_price_model.py --folds 3
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
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ====================================================================== #
#  Demand Backcast Generation
# ====================================================================== #
def generate_demand_backcasts(
    load_model_path: str,
    features_path: str,
) -> pd.Series:
    """Generate historical demand forecasts using the trained XGB model.

    Runs the load model on the feature matrix in an expanding-window
    fashion to simulate what the model would have predicted at each
    point in time (out-of-sample backcasts).

    We use an expanding window starting at 365 days to avoid using
    the model's own training data for the first year, then predict
    one day at a time going forward.

    Args:
        load_model_path: Path to saved XGB load model.
        features_path: Path to load_features.csv.

    Returns:
        Series indexed by date with backcast demand predictions.
    """
    from src.models.load_forecaster import XGBLoadForecaster

    logger.info("Generating demand backcasts...")

    # Load features
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    feature_cols = sorted([c for c in df.columns if c != "target"])
    df = df.dropna(subset=["target"])

    X = df[feature_cols].fillna(0).values
    y = df["target"].values
    dates = df.index

    # Load the trained model for feature structure reference
    model_template = XGBLoadForecaster()

    # Expanding-window backcasts (simulate real-time prediction)
    min_train = 365  # Need at least 1 year of training data
    backcasts = {}

    # For efficiency, retrain every 30 days rather than every day
    retrain_interval = 30
    current_model = None

    for i in range(min_train, len(X)):
        # Retrain periodically
        if current_model is None or (i - min_train) % retrain_interval == 0:
            train_X = X[:i]
            train_y = y[:i]
            val_size = max(1, int(len(train_X) * 0.1))

            current_model = copy.deepcopy(model_template)
            current_model.train(
                train_X[:-val_size], train_y[:-val_size],
                X_val=train_X[-val_size:], y_val=train_y[-val_size:],
            )

        # Predict for day i (out-of-sample)
        pred = current_model.predict(X[i:i+1])
        backcasts[dates[i]] = float(pred[0])

    backcast_series = pd.Series(backcasts, name="demand_forecast")
    backcast_series.index.name = "date"

    logger.info(
        "Generated %d demand backcasts (%.1f%% of dataset)",
        len(backcast_series), 100 * len(backcast_series) / len(df),
    )

    return backcast_series


# ====================================================================== #
#  Walk-Forward Price CV
# ====================================================================== #
def walk_forward_price(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    task: str = "classification",
) -> list[dict[str, float]]:
    """Walk-forward cross-validation for price models.

    Args:
        X: Feature matrix.
        y: Target (direction for classification, returns for regression).
        n_folds: Number of walk-forward folds.
        task: 'classification' or 'regression'.

    Returns:
        List of metric dicts per fold.
    """
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.metrics import accuracy_score, log_loss, f1_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    n = len(X)
    fold_size = n // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_end = min(train_end + fold_size, n)

        X_tr = np.nan_to_num(X[:train_end], nan=0.0, posinf=0.0, neginf=0.0)
        y_tr = y[:train_end]
        X_te = np.nan_to_num(X[train_end:test_end], nan=0.0, posinf=0.0, neginf=0.0)
        y_te = y[train_end:test_end]

        if task == "classification":
            model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric="logloss",
                early_stopping_rounds=30,
                n_jobs=-1,
                random_state=42,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                verbose=False,
            )
            preds = model.predict(X_te)
            proba = model.predict_proba(X_te)[:, 1]

            acc = accuracy_score(y_te, preds)
            base_rate = max(y_te.mean(), 1 - y_te.mean())
            f1 = f1_score(y_te, preds, zero_division=0)
            # Win rate on predicted "up" days
            up_mask = preds == 1
            if up_mask.sum() > 0:
                up_precision = accuracy_score(y_te[up_mask], preds[up_mask])
            else:
                up_precision = 0.0

            # Long bias check: fraction of predictions that are 1
            long_pct = float(preds.mean())

            fold_metrics = {
                "accuracy": acc,
                "base_rate": base_rate,
                "edge": acc - base_rate,
                "f1": f1,
                "up_precision": up_precision,
                "long_pct": long_pct,
                "n_test": len(y_te),
            }
            try:
                fold_metrics["log_loss"] = log_loss(y_te, proba)
            except ValueError:
                fold_metrics["log_loss"] = np.nan

        else:  # regression
            model = XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=5,
                gamma=0.1,
                n_jobs=-1,
                random_state=42,
                early_stopping_rounds=30,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                verbose=False,
            )
            preds = model.predict(X_te)

            mae = mean_absolute_error(y_te, preds)
            rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
            r2 = r2_score(y_te, preds)

            # Directional accuracy
            actual_dir = (y_te > 0).astype(int)
            pred_dir = (preds > 0).astype(int)
            dir_acc = accuracy_score(actual_dir, pred_dir)

            fold_metrics = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "directional_accuracy": dir_acc,
                "n_test": len(y_te),
            }

        results.append(fold_metrics)
        logger.info("  Fold %d/%d: %s", fold + 1, n_folds, fold_metrics)

    return results


# ====================================================================== #
#  Feature Importance
# ====================================================================== #
def get_feature_importance(X, y, feature_names, task="classification"):
    """Train a final model and return feature importances."""
    from xgboost import XGBClassifier, XGBRegressor

    if task == "classification":
        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, n_jobs=-1, random_state=42,
            use_label_encoder=False, eval_metric="logloss",
        )
    else:
        model = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, n_jobs=-1, random_state=42,
        )

    val_size = max(1, int(len(X) * 0.1))
    model.fit(
        X[:-val_size], y[:-val_size],
        eval_set=[(X[-val_size:], y[-val_size:])],
        verbose=False,
    )

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:25]
    return [(feature_names[i], importances[i]) for i in top_idx]


# ====================================================================== #
#  CLI
# ====================================================================== #
@click.command()
@click.option("--folds", default=5, help="Walk-forward CV folds")
@click.option("--horizon", default=1, help="Forecast horizon in days")
@click.option("--task", default="classification",
              type=click.Choice(["classification", "regression"]))
@click.option("--model-path", default="models/xgb_load.pkl",
              help="Path to trained demand model")
@click.option("--skip-backcast", is_flag=True,
              help="Skip backcast generation (use cached)")
@click.option("--save-features", is_flag=True,
              help="Save feature DataFrames to CSV")
def main(
    folds: int,
    horizon: int,
    task: str,
    model_path: str,
    skip_backcast: bool,
    save_features: bool,
) -> None:
    """Two-stage price prediction with demand-forecast ablation."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("[bold cyan]Two-Stage Price Prediction: Demand -> NG Price[/bold cyan]\n")

    # ── Load Data ─────────────────────────────────────────────────────
    console.print("[bold]Stage 0: Loading data...[/bold]")

    # NG prices
    ng_path = Path("data/raw/prices_natural_gas.csv")
    if not ng_path.exists():
        console.print("[dim]  Fetching NG prices from yfinance...[/dim]")
        from src.data.price_fetcher import PriceFetcher
        pf = PriceFetcher()
        data = pf.fetch(symbols=["natural_gas"], start="2019-01-01")
        ng_prices = data["natural_gas"]
        ng_prices.to_csv(ng_path)
    else:
        ng_prices = pd.read_csv(ng_path, index_col=0, parse_dates=True)
    console.print(f"  NG prices: {len(ng_prices)} days")

    # Weather
    weather_path = Path("data/raw/weather_daily.csv")
    weather_df = None
    if weather_path.exists():
        weather_df = pd.read_csv(weather_path, index_col=0, parse_dates=True)
        console.print(f"  Weather: {len(weather_df)} days")

    # Demand (actual)
    demand_path = Path("data/raw/demand_daily.csv")
    demand_df = None
    if demand_path.exists():
        demand_df = pd.read_csv(demand_path, index_col=0, parse_dates=True)
        console.print(f"  Demand: {len(demand_df)} days")

    # NG storage
    storage_path = Path("data/raw/eia_natgas_storage.csv")
    ng_storage = None
    if storage_path.exists():
        ng_storage = pd.read_csv(storage_path, index_col=0, parse_dates=True)
        console.print(f"  NG storage: {len(ng_storage)} rows")

    # ── Stage 1: Demand Backcasts ─────────────────────────────────────
    console.print("\n[bold]Stage 1: Generating demand backcasts...[/bold]")

    backcast_path = Path("data/processed/demand_backcasts.csv")
    features_path = Path("data/processed/load_features.csv")

    if skip_backcast and backcast_path.exists():
        backcasts = pd.read_csv(backcast_path, index_col=0, parse_dates=True).squeeze()
        console.print(f"  Loaded cached backcasts: {len(backcasts)} days")
    elif Path(model_path).exists() and features_path.exists():
        backcasts = generate_demand_backcasts(model_path, str(features_path))
        backcasts.to_csv(backcast_path)
        console.print(f"  Generated {len(backcasts)} demand backcasts")
    else:
        console.print("  [yellow]No demand model or features -- running WITHOUT demand[/yellow]")
        backcasts = None

    # ── Stage 2: Build Features ───────────────────────────────────────
    console.print("\n[bold]Stage 2: Building price features...[/bold]")

    from src.features.price_features import PriceFeatureBuilder

    builder = PriceFeatureBuilder(horizons=[horizon])
    feature_sets = builder.build(
        ng_prices=ng_prices,
        weather_df=weather_df,
        demand_df=demand_df,
        demand_forecast=backcasts,
        ng_storage=ng_storage,
    )

    baseline_df = feature_sets["baseline"]
    enhanced_df = feature_sets["enhanced"]

    # Set target column
    if task == "classification":
        target = f"target_dir_{horizon}d"
    else:
        target = f"target_ret_{horizon}d"

    # Feature columns (exclude all targets)
    target_cols = [c for c in baseline_df.columns if c.startswith("target_")]

    base_fcols = sorted([c for c in baseline_df.columns if c not in target_cols])
    enh_fcols = sorted([c for c in enhanced_df.columns if c not in target_cols])

    # Identify demand-specific columns
    demand_feature_names = sorted(set(enh_fcols) - set(base_fcols))

    console.print(f"  Baseline features: {len(base_fcols)}")
    console.print(f"  Enhanced features: {len(enh_fcols)} (+{len(demand_feature_names)} demand)")
    if demand_feature_names:
        console.print(f"  Demand features: {', '.join(demand_feature_names[:10])}")

    if save_features:
        baseline_df.to_csv("data/processed/price_features_baseline.csv")
        enhanced_df.to_csv("data/processed/price_features_enhanced.csv")
        console.print("  Saved feature CSVs")

    # ── Prepare arrays ────────────────────────────────────────────────
    # Drop rows with NaN features (early rows with insufficient lookback)
    for df_name, df_feat, fcols in [("baseline", baseline_df, base_fcols),
                                     ("enhanced", enhanced_df, enh_fcols)]:
        n_nans = df_feat[fcols].isna().sum(axis=1)
        drop_mask = n_nans > len(fcols) * 0.3
        if drop_mask.sum() > 0:
            logger.info("Dropping %d rows with >30%% NaN features from %s", drop_mask.sum(), df_name)

    # Replace inf with NaN, then drop rows missing the target
    baseline_df = baseline_df.replace([np.inf, -np.inf], np.nan)
    enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)

    baseline_clean = baseline_df.dropna(subset=[target])
    enhanced_clean = enhanced_df.dropna(subset=[target])

    # For enhanced: also drop rows that have no demand data at all
    # (demand_forecast is the primary demand feature)
    if "demand_forecast" in enh_fcols:
        enhanced_clean = enhanced_clean.dropna(subset=["demand_forecast"])

    # Align dates so comparison is fair
    common_dates = baseline_clean.index.intersection(enhanced_clean.index)
    baseline_clean = baseline_clean.loc[common_dates]
    enhanced_clean = enhanced_clean.loc[common_dates]

    X_base = baseline_clean[base_fcols].fillna(0).values
    X_enh = enhanced_clean[enh_fcols].fillna(0).values
    y_base = baseline_clean[target].values
    y_enh = enhanced_clean[target].values

    console.print(f"\n  Aligned dataset: {len(common_dates)} trading days")
    if task == "classification":
        console.print(f"  Up-day rate: {y_base.mean():.1%}")

    # ── Stage 3: Walk-Forward Evaluation ──────────────────────────────
    console.print(f"\n[bold]Stage 3: Walk-forward {task} ({folds} folds, {horizon}d horizon)...[/bold]\n")

    console.print("  [cyan]Training BASELINE (no demand features)...[/cyan]")
    base_results = walk_forward_price(X_base, y_base, n_folds=folds, task=task)

    console.print("\n  [cyan]Training ENHANCED (with demand features)...[/cyan]")
    enh_results = walk_forward_price(X_enh, y_enh, n_folds=folds, task=task)

    # ── Results Comparison ────────────────────────────────────────────
    console.print("\n" + "=" * 70)
    console.print("[bold green]ABLATION RESULTS: Demand Forecast Impact on NG Price Prediction[/bold green]")
    console.print("=" * 70 + "\n")

    if task == "classification":
        result_table = Table(title=f"NG {horizon}d Direction Prediction - Walk-Forward CV")
        result_table.add_column("Metric", style="cyan")
        result_table.add_column("Baseline", style="yellow", justify="right")
        result_table.add_column("+ Demand", style="green", justify="right")
        result_table.add_column("Delta", style="magenta", justify="right")

        metrics_to_show = ["accuracy", "base_rate", "edge", "f1", "log_loss", "long_pct"]
        labels = {
            "accuracy": "Accuracy",
            "base_rate": "Base Rate",
            "edge": "Edge (Acc - Base)",
            "f1": "F1 Score",
            "log_loss": "Log Loss",
            "long_pct": "Long Bias %",
        }

        for metric in metrics_to_show:
            base_vals = [f[metric] for f in base_results if not np.isnan(f.get(metric, np.nan))]
            enh_vals = [f[metric] for f in enh_results if not np.isnan(f.get(metric, np.nan))]
            if not base_vals or not enh_vals:
                continue

            base_mean = np.mean(base_vals)
            enh_mean = np.mean(enh_vals)
            base_std = np.std(base_vals)
            enh_std = np.std(enh_vals)
            delta = enh_mean - base_mean

            fmt = ".4f" if metric == "log_loss" else ".1%" if metric in ("long_pct",) else ".2%"
            if metric == "log_loss":
                result_table.add_row(
                    labels[metric],
                    f"{base_mean:.4f} +/- {base_std:.4f}",
                    f"{enh_mean:.4f} +/- {enh_std:.4f}",
                    f"{delta:+.4f}",
                )
            elif metric == "long_pct":
                result_table.add_row(
                    labels[metric],
                    f"{base_mean:.1%} +/- {base_std:.1%}",
                    f"{enh_mean:.1%} +/- {enh_std:.1%}",
                    f"{delta:+.1%}",
                )
            else:
                result_table.add_row(
                    labels[metric],
                    f"{base_mean:.2%} +/- {base_std:.2%}",
                    f"{enh_mean:.2%} +/- {enh_std:.2%}",
                    f"{delta:+.2%}",
                )

        console.print(result_table)

        # Per-fold detail
        fold_table = Table(title="Per-Fold Accuracy Comparison")
        fold_table.add_column("Fold", style="dim")
        fold_table.add_column("Baseline Acc", style="yellow", justify="right")
        fold_table.add_column("Enhanced Acc", style="green", justify="right")
        fold_table.add_column("Enhanced Edge", style="magenta", justify="right")
        fold_table.add_column("Base Rate", style="dim", justify="right")

        for i, (b, e) in enumerate(zip(base_results, enh_results)):
            fold_table.add_row(
                str(i + 1),
                f"{b['accuracy']:.2%}",
                f"{e['accuracy']:.2%}",
                f"{e['edge']:+.2%}",
                f"{e['base_rate']:.2%}",
            )

        console.print(fold_table)

    else:  # regression
        result_table = Table(title=f"NG {horizon}d Return Prediction - Walk-Forward CV")
        result_table.add_column("Metric", style="cyan")
        result_table.add_column("Baseline", style="yellow", justify="right")
        result_table.add_column("+ Demand", style="green", justify="right")
        result_table.add_column("Delta", style="magenta", justify="right")

        for metric in ["mae", "rmse", "r2", "directional_accuracy"]:
            base_mean = np.mean([f[metric] for f in base_results])
            enh_mean = np.mean([f[metric] for f in enh_results])
            base_std = np.std([f[metric] for f in base_results])
            enh_std = np.std([f[metric] for f in enh_results])
            delta = enh_mean - base_mean

            result_table.add_row(
                metric,
                f"{base_mean:.4f} +/- {base_std:.4f}",
                f"{enh_mean:.4f} +/- {enh_std:.4f}",
                f"{delta:+.4f}",
            )

        console.print(result_table)

    # ── Feature Importance (Enhanced model) ───────────────────────────
    console.print("\n[bold]Feature Importance (Enhanced model, top 25):[/bold]")

    fi = get_feature_importance(X_enh, y_enh, enh_fcols, task=task)
    fi_table = Table()
    fi_table.add_column("Rank", style="dim")
    fi_table.add_column("Feature", style="cyan")
    fi_table.add_column("Importance", style="green")
    fi_table.add_column("Type", style="yellow")

    for rank, (fname, imp) in enumerate(fi, 1):
        ftype = "DEMAND" if fname in demand_feature_names else "base"
        fi_table.add_row(str(rank), fname, f"{imp:.4f}", ftype)

    console.print(fi_table)

    # Count demand features in top-10
    top10 = fi[:10]
    n_demand_in_top10 = sum(1 for fname, _ in top10 if fname in demand_feature_names)
    console.print(f"\n  Demand features in top-10: {n_demand_in_top10}/10")

    # ── Summary ───────────────────────────────────────────────────────
    if task == "classification":
        base_acc = np.mean([f["accuracy"] for f in base_results])
        enh_acc = np.mean([f["accuracy"] for f in enh_results])
        base_edge = np.mean([f["edge"] for f in base_results])
        enh_edge = np.mean([f["edge"] for f in enh_results])

        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Baseline accuracy:  {base_acc:.2%} (edge: {base_edge:+.2%})")
        console.print(f"  Enhanced accuracy:  {enh_acc:.2%} (edge: {enh_edge:+.2%})")

        improvement = enh_acc - base_acc
        if improvement > 0.005:
            console.print(f"\n  [bold green]Demand forecasts IMPROVE NG direction by {improvement:+.2%}[/bold green]")
        elif improvement < -0.005:
            console.print(f"\n  [bold red]Demand forecasts HURT NG direction by {improvement:+.2%}[/bold red]")
        else:
            console.print(f"\n  [yellow]Demand forecasts have NEGLIGIBLE impact ({improvement:+.2%})[/yellow]")
    else:
        base_r2 = np.mean([f["r2"] for f in base_results])
        enh_r2 = np.mean([f["r2"] for f in enh_results])
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Baseline R2: {base_r2:.4f}")
        console.print(f"  Enhanced R2: {enh_r2:.4f}")
        console.print(f"  Delta:       {enh_r2 - base_r2:+.4f}")

    console.print("\n[bold green]Two-stage price prediction complete![/bold green]")


if __name__ == "__main__":
    main()
