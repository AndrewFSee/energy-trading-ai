#!/usr/bin/env python
"""Train NG production direction forecasting model.

Predicts whether US natural gas production will be higher or lower
in 3 months (90 days) based on rig counts, production trends,
price signals, and seasonal patterns.

This is a binary classification problem:
  - Target = 1: Production increases over next 90 days
  - Target = 0: Production decreases or flat

Uses XGBoost classifier with expanding-window cross-validation.

Usage::

    python scripts/train_ng_production_model.py
    python scripts/train_ng_production_model.py --horizon 180
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

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
    """Load NG production feature matrix."""
    if path:
        fp = Path(path)
    else:
        fp = PROCESSED_DIR / "ng_production_features.csv"

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
    min_train_days: int = 730,
    step_days: int = 90,
) -> dict:
    """Expanding-window walk-forward validation for classification.

    Uses longer minimum training period (2 years) and larger steps
    because production data is monthly → less effective daily granularity.

    Args:
        df: Feature matrix with ``target`` column.
        min_train_days: Minimum training set size.
        step_days: Days to advance per fold.

    Returns:
        Dictionary with predictions, actuals, metrics per fold, and overall.
    """
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df["target"]

    all_preds = []
    all_probs = []
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

        if len(X_test) == 0 or y_train.nunique() < 2:
            continue

        # Handle class imbalance
        class_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=10,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=class_ratio,
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        actuals = y_test.values

        acc = accuracy_score(actuals, preds)
        f1 = f1_score(actuals, preds, zero_division=0)
        try:
            auc = roc_auc_score(actuals, probs)
        except ValueError:
            auc = np.nan

        fold_metrics.append({
            "fold": fold,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "base_rate": y_test.mean(),
            "pred_rate": preds.mean(),
            "test_start": df.index[train_end].strftime("%Y-%m-%d"),
            "test_end": df.index[test_end - 1].strftime("%Y-%m-%d"),
        })

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_actuals.extend(actuals)
        all_dates.extend(df.index[train_end:test_end])

        if fold % 3 == 0:
            logger.info(
                "Fold %d: Acc=%.1f%%  F1=%.3f  AUC=%.3f  base=%.1f%%  train=%d",
                fold, acc * 100, f1, auc, y_test.mean() * 100, len(X_train),
            )

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_actuals = np.array(all_actuals)

    overall_acc = accuracy_score(all_actuals, all_preds)
    overall_f1 = f1_score(all_actuals, all_preds, zero_division=0)
    try:
        overall_auc = roc_auc_score(all_actuals, all_probs)
    except ValueError:
        overall_auc = np.nan

    overall = {
        "accuracy": overall_acc,
        "f1": overall_f1,
        "auc": overall_auc,
        "base_rate": all_actuals.mean(),
        "pred_rate": all_preds.mean(),
        "n_folds": len(fold_metrics),
        "total_test_days": len(all_preds),
    }

    return {
        "predictions": all_preds,
        "probabilities": all_probs,
        "actuals": all_actuals,
        "dates": all_dates,
        "fold_metrics": fold_metrics,
        "overall": overall,
    }


def train_final_model(df: pd.DataFrame) -> tuple:
    """Train final model on all data. Returns model + importances."""
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df["target"]

    class_ratio = (y == 0).sum() / max((y == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=class_ratio,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y, verbose=False)

    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return model, importances


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NG production model")
    parser.add_argument("--features", default=None, help="Path to feature CSV")
    parser.add_argument("--min-train", type=int, default=730,
                        help="Minimum training days (default 2 years)")
    parser.add_argument("--step", type=int, default=90,
                        help="Days per CV fold step")
    parser.add_argument("--horizon", type=int, default=90,
                        help="Forecast horizon days (used at feature build time)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load features
    df = load_features(args.features)

    # Drop rows with excessive NaN
    n_before = len(df)
    df = df.dropna(thresh=len(df.columns) // 2)
    if len(df) < n_before:
        logger.info("Dropped %d rows → %d remaining", n_before - len(df), len(df))

    # Report class balance
    logger.info("Class balance: %.1f%% increases, %.1f%% decreases",
                df["target"].mean() * 100, (1 - df["target"].mean()) * 100)

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
    logger.info("  Accuracy:  %.1f%%", overall["accuracy"] * 100)
    logger.info("  F1 Score:  %.3f", overall["f1"])
    logger.info("  AUC:       %.3f", overall["auc"])
    logger.info("  Base Rate: %.1f%% (always-increase baseline)",
                overall["base_rate"] * 100)
    logger.info("  Pred Rate: %.1f%% (model predicts increase)",
                overall["pred_rate"] * 100)
    logger.info("  Lift:      %.1f pp over base rate",
                (overall["accuracy"] - overall["base_rate"]) * 100)
    logger.info("=" * 60)

    # Fold summary
    fold_df = pd.DataFrame(results["fold_metrics"])
    print("\nFold-by-fold metrics:")
    cols = ["fold", "test_start", "test_end", "accuracy", "f1", "auc", "base_rate"]
    print(fold_df[cols].to_string(index=False))

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        results["actuals"], results["predictions"],
        target_names=["Decrease", "Increase"],
    ))

    # Train final model
    logger.info("Training final model on all %d rows...", len(df))
    model, importances = train_final_model(df)

    # Save model
    model_path = MODELS_DIR / "xgb_ng_production.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model: %s", model_path)

    # Feature importances
    print("\nTop 20 features for NG production direction:")
    for feat, imp in importances.head(20).items():
        print(f"  {feat:40s} {imp:.4f}")

    imp_path = MODELS_DIR / "ng_production_feature_importance.csv"
    importances.to_csv(imp_path, header=["importance"])

    # Save predictions
    pred_df = pd.DataFrame({
        "date": results["dates"],
        "actual": results["actuals"],
        "predicted": results["predictions"],
        "probability": results["probabilities"],
    })
    pred_path = PROCESSED_DIR / "ng_production_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info("Saved predictions: %s", pred_path)


if __name__ == "__main__":
    main()
