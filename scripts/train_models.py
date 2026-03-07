#!/usr/bin/env python3
"""CLI script to train all forecasting models.

Loads processed feature data, performs walk-forward validation,
and saves trained models to disk.

Usage:
    python scripts/train_models.py [OPTIONS]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--features-dir", default="data/processed", help="Processed features directory")
@click.option("--models-dir", default="models", help="Directory to save trained models")
@click.option("--instrument", default="wti", help="Instrument to train on")
@click.option(
    "--model", default="all", type=click.Choice(["all", "xgboost", "lstm", "transformer"])
)
@click.option("--folds", default=5, help="Walk-forward validation folds")
def main(
    features_dir: str,
    models_dir: str,
    instrument: str,
    model: str,
    folds: int,
) -> None:
    """Train forecasting models with walk-forward validation."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    features_path = Path(features_dir)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    feature_file = features_path / f"features_{instrument}.csv"
    if not feature_file.exists():
        console.print(f"[red]Feature data not found: {feature_file}[/red]")
        console.print("Run: python scripts/build_features.py first")
        return

    console.print(f"[bold green]Training Models for {instrument.upper()}[/bold green]\n")

    df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
    target_col = "target"
    if target_col not in df.columns:
        console.print(f"[red]Target column '{target_col}' not found in features[/red]")
        return

    feature_cols = [
        c for c in df.columns if c not in {target_col, "Open", "High", "Low", "Close", "Volume"}
    ]
    df = df.dropna(subset=[target_col])
    X = df[feature_cols].fillna(0).values
    y = df[target_col].values

    console.print(f"  Dataset: {len(X)} samples, {len(feature_cols)} features")
    console.print(f"  Walk-forward folds: {folds}\n")

    models_to_train = []
    if model in ("all", "xgboost"):
        from src.models.xgboost_model import XGBoostModel

        models_to_train.append(XGBoostModel())

    if model in ("all", "lstm"):
        from src.models.lstm_model import LSTMModel

        models_to_train.append(LSTMModel())

    if model in ("all", "transformer"):
        from src.models.transformer_model import TransformerModel

        models_to_train.append(TransformerModel())

    from src.models.training import ModelTrainer

    results_table = Table(title="Walk-Forward Validation Results")
    results_table.add_column("Model", style="cyan")
    results_table.add_column("MAE", style="green")
    results_table.add_column("RMSE", style="green")
    results_table.add_column("Dir. Accuracy", style="yellow")

    for m in models_to_train:
        console.print(f"  Training {m.name}...")
        trainer = ModelTrainer(m, n_folds=folds)
        result = trainer.walk_forward_validate(X, y)

        results_table.add_row(
            m.name,
            f"{result.mean_metrics.get('mae', 0):.4f} ± {result.std_metrics.get('mae', 0):.4f}",
            f"{result.mean_metrics.get('rmse', 0):.4f} ± {result.std_metrics.get('rmse', 0):.4f}",
            f"{result.mean_metrics.get('directional_accuracy', 0):.3f}",
        )

        # Final training on full dataset
        trainer.final_train(X, y)
        save_path = models_path / f"{m.name.lower()}_{instrument}.pkl"
        m.save(save_path)
        console.print(f"    ✓ Saved to {save_path}")

    console.print()
    console.print(results_table)
    console.print("\n[bold green]Training complete![/bold green]")


if __name__ == "__main__":
    main()
