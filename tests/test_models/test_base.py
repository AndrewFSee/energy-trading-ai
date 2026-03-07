"""Tests for the BaseModel abstract class."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.base import BaseModel


class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""

    @property
    def default_config(self) -> dict:
        return {"learning_rate": 0.01, "epochs": 10}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val=None,
        y_val=None,
    ) -> ConcreteModel:
        """Simple passthrough training for testing."""
        self._model = "trained"
        self.is_fitted = True
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return zeros as predictions (for testing)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return np.zeros(len(X))


class TestBaseModel:
    """Unit tests for BaseModel via ConcreteModel."""

    def test_init_sets_name(self) -> None:
        """Model name should be set on initialisation."""
        model = ConcreteModel(name="TestModel")
        assert model.name == "TestModel"

    def test_init_not_fitted(self) -> None:
        """New model should not be fitted."""
        model = ConcreteModel(name="TestModel")
        assert not model.is_fitted

    def test_default_config_merged(self) -> None:
        """Default config should be applied when none provided."""
        model = ConcreteModel(name="TestModel")
        assert model.config["learning_rate"] == 0.01
        assert model.config["epochs"] == 10

    def test_custom_config_overrides_default(self) -> None:
        """Custom config values should override defaults."""
        model = ConcreteModel(name="TestModel", config={"learning_rate": 0.001})
        assert model.config["learning_rate"] == 0.001
        assert model.config["epochs"] == 10  # Still uses default

    def test_train_sets_is_fitted(self) -> None:
        """Training should set is_fitted to True."""
        model = ConcreteModel(name="TestModel")
        X = np.random.default_rng(0).random((50, 5))
        y = np.random.default_rng(0).random(50)
        model.train(X, y)
        assert model.is_fitted

    def test_predict_before_train_raises(self) -> None:
        """predict() before training should raise RuntimeError."""
        model = ConcreteModel(name="TestModel")
        X = np.random.default_rng(0).random((10, 5))
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_predict_returns_array(self) -> None:
        """predict() should return a numpy array."""
        model = ConcreteModel(name="TestModel")
        X = np.random.default_rng(0).random((50, 5))
        y = np.random.default_rng(0).random(50)
        model.train(X, y)
        preds = model.predict(np.random.default_rng(1).random((10, 5)))
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 10

    def test_evaluate_returns_metrics(self) -> None:
        """evaluate() should return a dictionary with standard metrics."""
        model = ConcreteModel(name="TestModel")
        n = 50
        X = np.random.default_rng(0).random((n, 5))
        y = np.zeros(n)  # Matches predict() which returns zeros
        model.train(X, y)
        metrics = model.evaluate(X, y)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "directional_accuracy" in metrics

    def test_evaluate_mae_correct(self) -> None:
        """MAE should be 0.0 when predictions equal true values."""
        model = ConcreteModel(name="TestModel")
        n = 20
        X = np.ones((n, 3))
        y = np.zeros(n)  # predict returns zeros
        model.train(X, y)
        metrics = model.evaluate(X, y)
        assert abs(metrics["mae"]) < 1e-9

    def test_repr_contains_name(self) -> None:
        """repr() should include the model name."""
        model = ConcreteModel(name="MyModel")
        assert "MyModel" in repr(model)

    def test_save_and_load(self, tmp_path) -> None:
        """Model can be saved to disk and loaded back."""
        model = ConcreteModel(name="SaveTest")
        X = np.random.default_rng(0).random((30, 4))
        y = np.random.default_rng(0).random(30)
        model.train(X, y)
        save_path = tmp_path / "model.pkl"
        model.save(save_path)
        loaded = ConcreteModel.load(save_path)
        assert loaded.is_fitted
        assert loaded.name == "SaveTest"
