"""Feature pipeline orchestrator.

Combines all feature sets (technical, fundamental, macro, seasonal, NLP sentiment)
into a single model-ready feature matrix.  Handles missing data imputation,
feature scaling, and train/validation/test splitting with strict temporal ordering.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.fundamental import FundamentalFeatures
from src.features.macro import MacroFeatures
from src.features.seasonal import SeasonalFeatures
from src.features.technical import TechnicalFeatures

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Orchestrates all feature engineering steps into a unified pipeline.

    Sequentially applies technical, fundamental, macro, and seasonal
    feature transformations, then handles scaling and target creation.

    Attributes:
        technical: ``TechnicalFeatures`` instance.
        fundamental: ``FundamentalFeatures`` instance.
        macro: ``MacroFeatures`` instance.
        seasonal: ``SeasonalFeatures`` instance.
        scaler: ``StandardScaler`` fitted on training data.
        feature_columns: Column names of the final feature matrix.
    """

    def __init__(
        self,
        technical: TechnicalFeatures | None = None,
        fundamental: FundamentalFeatures | None = None,
        macro: MacroFeatures | None = None,
        seasonal: SeasonalFeatures | None = None,
    ) -> None:
        """Initialise the feature pipeline with optional sub-transformers.

        Args:
            technical: Pre-configured ``TechnicalFeatures`` instance.
            fundamental: Pre-configured ``FundamentalFeatures`` instance.
            macro: Pre-configured ``MacroFeatures`` instance.
            seasonal: Pre-configured ``SeasonalFeatures`` instance.
        """
        self.technical = technical or TechnicalFeatures()
        self.fundamental = fundamental or FundamentalFeatures()
        self.macro = macro or MacroFeatures()
        self.seasonal = seasonal or SeasonalFeatures()
        self.scaler: StandardScaler | None = None
        self.feature_columns: list[str] = []

    def build(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame | None = None,
        storage_df: pd.DataFrame | None = None,
        macro_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build the complete feature matrix from raw data inputs.

        Args:
            price_df: OHLCV price DataFrame with ``DatetimeIndex``.
            sentiment_df: Optional daily sentiment index DataFrame with a
                ``sentiment_score`` column.
            storage_df: Optional weekly EIA storage DataFrame.
            macro_df: Optional daily macro features DataFrame from FRED.

        Returns:
            Feature matrix ``pd.DataFrame`` with all engineered features.
            Missing values are imputed and rows with insufficient history
            are dropped.
        """
        logger.info("Building feature matrix from price data (shape=%s)", price_df.shape)
        df = price_df.copy()

        # Technical features
        df = self.technical.add_all(df)

        # Seasonal features
        df = self.seasonal.add_all(df)

        # Fundamental features (optional)
        if storage_df is not None and not storage_df.empty:
            df = self.fundamental.add_storage_features(df, storage_df)

        # Macro features (optional)
        if macro_df is not None and not macro_df.empty:
            macro_cols = [c for c in macro_df.columns if c not in df.columns]
            if macro_cols:
                df = df.join(macro_df[macro_cols].reindex(df.index).ffill(), how="left")

        # Sentiment features (optional)
        if sentiment_df is not None and not sentiment_df.empty:
            sentiment_cols = [c for c in sentiment_df.columns if c not in df.columns]
            if sentiment_cols:
                df = df.join(sentiment_df[sentiment_cols].reindex(df.index).ffill(), how="left")

        # Drop rows dominated by NaN (early history without enough lookback)
        df = df.dropna(thresh=int(len(df.columns) * 0.5))
        df = df.ffill().bfill()

        self.feature_columns = [
            c for c in df.columns if c not in {"Open", "High", "Low", "Close", "Volume"}
        ]

        logger.info(
            "Feature matrix built: shape=%s, features=%d", df.shape, len(self.feature_columns)
        )
        return df

    def create_targets(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        target_col: str = "Close",
        method: str = "log_return",
    ) -> pd.Series:
        """Create forward-looking prediction targets.

        Args:
            df: Feature DataFrame with price columns.
            horizon: Forecast horizon in days.
            target_col: Column to compute targets from (typically ``"Close"``).
            method: Target type — ``"log_return"`` for log return, ``"direction"``
                for binary up/down, ``"pct_change"`` for percentage return.

        Returns:
            ``pd.Series`` of target values aligned to the feature DataFrame.
        """
        price = df[target_col]
        if method == "log_return":
            target = np.log(price.shift(-horizon) / price)
        elif method == "pct_change":
            target = price.pct_change(horizon).shift(-horizon)
        elif method == "direction":
            target = (price.shift(-horizon) > price).astype(int)
        else:
            raise ValueError(f"Unknown target method: '{method}'")
        target.name = f"target_{method}_{horizon}d"
        return target

    def split_train_val_test(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> tuple[
        tuple[pd.DataFrame, pd.Series],
        tuple[pd.DataFrame, pd.Series],
        tuple[pd.DataFrame, pd.Series],
    ]:
        """Split data into train, validation, and test sets (temporal order).

        Args:
            df: Feature DataFrame.
            target: Target series aligned to ``df``.
            train_ratio: Fraction of data for training.
            val_ratio: Fraction of data for validation.

        Returns:
            Tuple of (train, val, test) where each is a (X, y) tuple.
        """
        combined = df.copy()
        combined["__target__"] = target
        combined = combined.dropna(subset=["__target__"])

        n = len(combined)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = combined.iloc[:n_train]
        val = combined.iloc[n_train : n_train + n_val]
        test = combined.iloc[n_train + n_val :]

        feature_cols = [c for c in combined.columns if c != "__target__"]

        logger.info(
            "Data split: train=%d, val=%d, test=%d",
            len(train),
            len(val),
            len(test),
        )
        return (
            (train[feature_cols], train["__target__"]),
            (val[feature_cols], val["__target__"]),
            (test[feature_cols], test["__target__"]),
        )

    def fit_scaler(self, X_train: pd.DataFrame) -> FeaturePipeline:
        """Fit the StandardScaler on training data.

        Args:
            X_train: Training feature matrix.

        Returns:
            Self (for method chaining).
        """
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        logger.info("Scaler fitted on %d training samples", len(X_train))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted scaler to feature data.

        Args:
            X: Feature DataFrame to scale.

        Returns:
            Scaled feature array.

        Raises:
            RuntimeError: If scaler has not been fitted.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit_scaler() first.")
        return self.scaler.transform(X)  # type: ignore[return-value]
