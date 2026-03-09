#!/usr/bin/env python
"""Train wind generation forecasting model.

Uses XGBoost with expanding-window cross-validation to forecast
day-ahead total Eastern US wind generation (MWh).

Evaluation metrics:
  - R² (coefficient of determination)
  - MAPE (mean absolute percentage error)
  - MAE (mean absolute error)
  - Feature importance analysis

Usage::

    python scripts/train_wind_model.py
    python scripts/train_wind_model.py --features data/processed/wind_gen_features.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_features(path: str | None = None) -> pd.DataFrame:
    """Load wind generation feature matrix."""
    if path:
        fp = Path(path)
    else:
        fp = PROCESSED_DIR / "wind_gen_features.csv"

    if not fp.exists():
        raise FileNotFoundError(
            f"Feature file not found: {fp}\n"
            "Run scripts/ingest_generation_data.py first."
        )

    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    logger.info("Loaded features: %d rows × %d cols from %s", len(df), len(df.columns), fp)
    return df


def expanding_window_cv(
    df: pd.DataFrame,
    min_train_days: int = 365,
    step_days: int = 30,
) -> dict:
    """Expanding-window walk-forward validation.

    Args:
        df: Feature matrix with ``target`` column.
        min_train_days: Minimum training set size (days).
        step_days: How many days to advance per fold.

    Returns:
        Dictionary with predictions, actuals, metrics per fold, and overall metrics.
    """
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]

    all_preds = []
    all_actuals = []
    all_dates = []
    fold_metrics = []

    n = len(df)
    fold = 0

    for split_idx in range(min_train_days, n - step_days, step_days):
        fold += 1
        train_end = split_idx
        test_end = min(split_idx + step_days, n)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

        if len(X_test) == 0:
            break

        # Handle non-numeric columns gracefully
        X_train = X_train.select_dtypes(include=[np.number])
        X_test = X_test[X_train.columns]

        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds = model.predict(X_test)
        actuals = y_test.values

        r2 = r2_score(actuals, preds)
        mape = mean_absolute_percentage_error(actuals, preds) * 100
        mae = mean_absolute_error(actuals, preds)

        fold_metrics.append({
            "fold": fold,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "r2": r2,
            "mape": mape,
            "mae": mae,
            "test_start": df.index[train_end].strftime("%Y-%m-%d"),
            "test_end": df.index[test_end - 1].strftime("%Y-%m-%d"),
        })

        all_preds.extend(preds)
        all_actuals.extend(actuals)
        all_dates.extend(df.index[train_end:test_end])

        if fold % 5 == 0:
            logger.info(
                "Fold %d: R²=%.3f  MAPE=%.2f%%  MAE=%.0f  train=%d  test=%d",
                fold, r2, mape, mae, len(X_train), len(X_test),
            )

    # Overall metrics
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    overall = {
        "r2": r2_score(all_actuals, all_preds),
        "mape": mean_absolute_percentage_error(all_actuals, all_preds) * 100,
        "mae": mean_absolute_error(all_actuals, all_preds),
        "n_folds": len(fold_metrics),
        "total_test_days": len(all_preds),
    }

    return {
        "predictions": all_preds,
        "actuals": all_actuals,
        "dates": all_dates,
        "fold_metrics": fold_metrics,
        "overall": overall,
    }


def train_final_model(df: pd.DataFrame) -> tuple:
    """Train a final model on all available data.

    Returns the model and feature importances.
    """
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df["target"]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y, verbose=False)

    # Feature importance
    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return model, importances


def main() -> None:
    parser = argparse.ArgumentParser(description="Train wind generation model")
    parser.add_argument("--features", default=None, help="Path to feature CSV")
    parser.add_argument("--min-train", type=int, default=365,
                        help="Minimum training days")
    parser.add_argument("--step", type=int, default=30,
                        help="Days per CV fold step")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load features
    df = load_features(args.features)

    # Drop rows with all-NaN features
    n_before = len(df)
    df = df.dropna(thresh=len(df.columns) // 2)
    if len(df) < n_before:
        logger.info("Dropped %d rows with excessive NaN → %d remaining",
                     n_before - len(df), len(df))

    logger.info("=" * 60)
    logger.info("EXPANDING-WINDOW CROSS-VALIDATION")
    logger.info("=" * 60)

    results = expanding_window_cv(
        df, min_train_days=args.min_train, step_days=args.step,
    )

    overall = results["overall"]
    logger.info("=" * 60)
    logger.info("OVERALL RESULTS (%d folds, %d test days)",
                overall["n_folds"], overall["total_test_days"])
    logger.info("  R²:   %.4f", overall["r2"])
    logger.info("  MAPE: %.2f%%", overall["mape"])
    logger.info("  MAE:  %.0f MWh", overall["mae"])
    logger.info("=" * 60)

    # Fold-by-fold summary
    fold_df = pd.DataFrame(results["fold_metrics"])
    print("\nFold-by-fold metrics:")
    print(fold_df[["fold", "test_start", "test_end", "r2", "mape", "mae"]].to_string(index=False))

    # Train final model
    logger.info("\nTraining final model on all %d rows...", len(df))
    model, importances = train_final_model(df)

    # Save model
    model_path = MODELS_DIR / "xgb_wind_gen.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model: %s", model_path)

    # Save feature importances
    print("\nTop 20 wind gen features:")
    for feat, imp in importances.head(20).items():
        print(f"  {feat:40s} {imp:.4f}")

    imp_path = MODELS_DIR / "wind_gen_feature_importance.csv"
    importances.to_csv(imp_path, header=["importance"])
    logger.info("Saved importances: %s", imp_path)

    # Save predictions for analysis
    pred_df = pd.DataFrame({
        "date": results["dates"],
        "actual": results["actuals"],
        "predicted": results["predictions"],
    })
    pred_df["error"] = pred_df["predicted"] - pred_df["actual"]
    pred_df["pct_error"] = (pred_df["error"] / pred_df["actual"].clip(lower=1)) * 100
    pred_path = PROCESSED_DIR / "wind_gen_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info("Saved predictions: %s", pred_path)


if __name__ == "__main__":
    main()
