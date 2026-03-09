#!/usr/bin/env python3
"""Run a grid of experiments across horizons, targets, and models.

Tests: {1-day, 5-day, 20-day} × {regression, classification} × {Ridge/Logistic, XGBoost}
All with non-overlapping targets for honest evaluation.

Usage:
    python scripts/run_experiments.py [OPTIONS]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from src.models.base import BaseModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_classification_target(y_regression: np.ndarray) -> np.ndarray:
    """Convert regression targets to binary direction (1 = positive return)."""
    return (y_regression > 0).astype(int)


@click.command()
@click.option("--features-dir", default="data/processed", help="Processed features directory")
@click.option("--instrument", default="wti", help="Instrument to evaluate")
@click.option("--folds", default=5, help="Walk-forward validation folds")
def main(features_dir: str, instrument: str, folds: int) -> None:
    """Run experiment grid and print comparison table."""
    from rich.console import Console
    from rich.table import Table

    from src.models.training import ModelTrainer

    console = Console()
    features_path = Path(features_dir)

    console.print("[bold green]Experiment Grid: Horizons × Targets × Models[/bold green]\n")

    # Load the base feature data (has 5-day target, but we'll recompute)
    feature_file = features_path / f"features_{instrument}.csv"
    if not feature_file.exists():
        console.print(f"[red]Feature data not found: {feature_file}[/red]")
        return

    full_df = pd.read_csv(feature_file, index_col=0, parse_dates=True)

    # We need Close prices to compute targets at different horizons
    if "Close" not in full_df.columns:
        console.print("[red]Close column not found — cannot compute multi-horizon targets[/red]")
        return

    feature_cols = [
        c for c in full_df.columns
        if c not in {"target", "Open", "High", "Low", "Close", "Volume"}
    ]
    console.print(f"  Features: {len(feature_cols)}")
    console.print(f"  Base rows: {len(full_df)}\n")

    horizons = [1, 5, 20]
    target_types = ["regression", "classification"]

    results = []

    for horizon in horizons:
        # Compute target for this horizon
        close = full_df["Close"]
        target_reg = np.log(close.shift(-horizon) / close)

        df_h = full_df.copy()
        df_h["target_reg"] = target_reg
        df_h = df_h.dropna(subset=["target_reg"])

        # Non-overlapping subsample
        df_h = df_h.iloc[::horizon]
        n_samples = len(df_h)

        X = df_h[feature_cols].fillna(0).values
        y_reg = df_h["target_reg"].values
        y_cls = make_classification_target(y_reg)

        # Check class balance
        pos_pct = y_cls.mean() * 100

        console.print(
            f"  [cyan]Horizon {horizon}d:[/cyan] {n_samples} non-overlapping samples "
            f"(class balance: {pos_pct:.1f}% positive)"
        )

        for target_type in target_types:
            y = y_reg if target_type == "regression" else y_cls

            # Select models based on target type
            if target_type == "regression":
                from src.models.linear_model import RidgeModel
                from src.models.xgboost_model import XGBoostModel

                models = [
                    ("Ridge", RidgeModel()),
                    ("XGBoost", XGBoostModel()),
                ]
            else:
                from src.models.linear_model import LogisticModel

                # XGBoost classifier
                from xgboost import XGBClassifier

                models = [
                    ("Logistic", LogisticModel()),
                    ("XGB-Cls", _XGBClassifierWrapper()),
                ]

            for model_name, model in models:
                trainer = ModelTrainer(model, n_folds=folds)
                try:
                    result = trainer.walk_forward_validate(X, y)
                    metrics = result.mean_metrics
                    std = result.std_metrics
                except Exception as exc:
                    logger.warning("Failed: %s/%s/%dd — %s", model_name, target_type, horizon, exc)
                    metrics = {}
                    std = {}

                row = {
                    "horizon": f"{horizon}d",
                    "target": target_type,
                    "model": model_name,
                    "samples": n_samples,
                }

                if target_type == "regression":
                    row["MAE"] = f"{metrics.get('mae', 0):.4f} ± {std.get('mae', 0):.4f}"
                    row["R²"] = f"{metrics.get('r2', 0):.4f}"
                    row["Dir.Acc"] = f"{metrics.get('directional_accuracy', 0):.3f}"
                    row["metric_sort"] = metrics.get("r2", -999)
                else:
                    row["Accuracy"] = f"{metrics.get('accuracy', metrics.get('directional_accuracy', 0)):.3f}"
                    row["Precision"] = f"{metrics.get('precision', 0):.3f}"
                    row["F1"] = f"{metrics.get('f1', 0):.3f}"
                    row["metric_sort"] = metrics.get("accuracy", metrics.get("directional_accuracy", 0))

                results.append(row)
                console.print(f"    ✓ {model_name}/{target_type}/{horizon}d done")

        console.print()

    # Print regression results table
    console.print()
    reg_table = Table(title="Regression Results (non-overlapping)")
    reg_table.add_column("Horizon", style="cyan")
    reg_table.add_column("Model", style="green")
    reg_table.add_column("Samples", style="dim")
    reg_table.add_column("MAE", style="yellow")
    reg_table.add_column("R²", style="red")
    reg_table.add_column("Dir. Accuracy", style="magenta")

    for r in results:
        if r["target"] == "regression":
            reg_table.add_row(
                r["horizon"], r["model"], str(r["samples"]),
                r["MAE"], r["R²"], r["Dir.Acc"],
            )

    console.print(reg_table)

    # Print classification results table
    console.print()
    cls_table = Table(title="Classification Results (non-overlapping)")
    cls_table.add_column("Horizon", style="cyan")
    cls_table.add_column("Model", style="green")
    cls_table.add_column("Samples", style="dim")
    cls_table.add_column("Accuracy", style="yellow")
    cls_table.add_column("Precision", style="red")
    cls_table.add_column("F1", style="magenta")

    for r in results:
        if r["target"] == "classification":
            cls_table.add_row(
                r["horizon"], r["model"], str(r["samples"]),
                r["Accuracy"], r["Precision"], r["F1"],
            )

    console.print(cls_table)

    # Summary
    console.print("\n[bold]Best regression R²:[/bold]", style="green")
    reg_results = [r for r in results if r["target"] == "regression"]
    if reg_results:
        best = max(reg_results, key=lambda x: x["metric_sort"])
        console.print(f"  {best['model']} @ {best['horizon']}: R² = {best['R²']}")

    console.print("[bold]Best classification accuracy:[/bold]", style="green")
    cls_results = [r for r in results if r["target"] == "classification"]
    if cls_results:
        best = max(cls_results, key=lambda x: x["metric_sort"])
        console.print(f"  {best['model']} @ {best['horizon']}: Acc = {best['Accuracy']}")

    console.print("\n[bold green]Experiment grid complete![/bold green]")


class _XGBClassifierWrapper(BaseModel):
    """XGBoost classifier wrapped in BaseModel interface for walk-forward."""

    def __init__(self) -> None:
        super().__init__(name="XGB-Cls", config=None)
        from sklearn.preprocessing import StandardScaler as _SS
        self._scaler = _SS()

    @property
    def default_config(self) -> dict:
        return {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 50,
            "random_state": 42,
        }

    def train(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.preprocessing import StandardScaler as _SS
        from xgboost import XGBClassifier

        self._scaler = _SS()
        X_train_s = self._scaler.fit_transform(X_train)

        fit_params = {}
        if X_val is not None and y_val is not None:
            X_val_s = self._scaler.transform(X_val)
            fit_params["eval_set"] = [(X_val_s, y_val)]
            fit_params["verbose"] = False

        params = {k: v for k, v in self.config.items() if k != "early_stopping_rounds"}
        self._model = XGBClassifier(
            **params,
            early_stopping_rounds=self.config["early_stopping_rounds"] if X_val is not None else None,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self._model.fit(X_train_s, y_train, **fit_params)
        self.is_fitted = True
        logger.info("XGB-Cls trained (%d estimators)", self.config["n_estimators"])
        return self

    def predict(self, X):
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model not fitted")
        X_s = self._scaler.transform(X)
        return self._model.predict(X_s)

    def evaluate(self, X, y):
        preds = self.predict(X)
        accuracy = float(np.mean(preds == y))
        tp = float(np.sum((preds == 1) & (y == 1)))
        fp = float(np.sum((preds == 1) & (y == 0)))
        fn = float(np.sum((preds == 0) & (y == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "directional_accuracy": accuracy,
        }
        logger.info("Evaluation (%s): %s", self.name, metrics)
        return metrics


if __name__ == "__main__":
    main()
