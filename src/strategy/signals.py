"""Trading signal generation from model outputs.

Converts raw model prediction values (forecast returns) into discrete
trading signals: LONG (+1), FLAT (0), SHORT (-1).  Includes signal
filtering, confirmation windows, and LLM signal integration.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Converts model forecast outputs into actionable trading signals.

    Applies configurable thresholds and optional confirmation windows
    to reduce noise and avoid whipsawing.

    Attributes:
        long_threshold: Minimum predicted return to generate a LONG signal.
        short_threshold: Maximum predicted return to generate a SHORT signal.
        confirmation_window: Bars signal must persist before acting.
        signal_smoothing: If True, apply EMA smoothing to raw predictions.
    """

    LONG = 1
    FLAT = 0
    SHORT = -1

    def __init__(
        self,
        long_threshold: float = 0.015,
        short_threshold: float = -0.015,
        confirmation_window: int = 2,
        signal_smoothing: bool = True,
        smoothing_span: int = 3,
    ) -> None:
        """Initialise the signal generator.

        Args:
            long_threshold: Predicted return above this generates LONG (+1).
            short_threshold: Predicted return below this generates SHORT (-1).
            confirmation_window: Bars the raw signal must persist before
                the final signal is flipped.
            signal_smoothing: Whether to apply EMA smoothing to predictions.
            smoothing_span: EMA span for prediction smoothing.
        """
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.confirmation_window = confirmation_window
        self.signal_smoothing = signal_smoothing
        self.smoothing_span = smoothing_span

    def generate(
        self,
        predictions: np.ndarray | pd.Series,
        llm_signal: float | None = None,
        llm_weight: float = 0.2,
    ) -> pd.Series:
        """Convert predictions to trading signals.

        Args:
            predictions: Array or Series of predicted returns.
            llm_signal: Optional LLM qualitative signal in [-1, +1] range.
                When provided, blends with quantitative signal.
            llm_weight: Weight of the LLM signal in the combined signal.

        Returns:
            ``pd.Series`` of integer signals (LONG=1, FLAT=0, SHORT=-1).
        """
        if isinstance(predictions, np.ndarray):
            preds = pd.Series(predictions)
        else:
            preds = predictions.copy()

        # Smooth predictions to reduce noise
        if self.signal_smoothing:
            preds = preds.ewm(span=self.smoothing_span, adjust=False).mean()

        # Blend with LLM signal if provided
        if llm_signal is not None:
            quant_weight = 1.0 - llm_weight
            preds = preds * quant_weight + llm_signal * llm_weight
            logger.debug("Blended LLM signal (weight=%.2f) into predictions", llm_weight)

        # Apply thresholds to generate raw signals
        raw_signal = pd.Series(
            np.where(
                preds > self.long_threshold,
                self.LONG,
                np.where(preds < self.short_threshold, self.SHORT, self.FLAT),
            ),
            index=preds.index,
            dtype=int,
        )

        # Apply confirmation window (signal must persist N bars)
        if self.confirmation_window > 1:
            confirmed = self._apply_confirmation(raw_signal)
        else:
            confirmed = raw_signal

        logger.info(
            "Signals generated: LONG=%d, FLAT=%d, SHORT=%d",
            (confirmed == self.LONG).sum(),
            (confirmed == self.FLAT).sum(),
            (confirmed == self.SHORT).sum(),
        )
        return confirmed

    def _apply_confirmation(self, raw_signal: pd.Series) -> pd.Series:
        """Require the signal to persist for ``confirmation_window`` bars.

        Args:
            raw_signal: Raw integer signal series.

        Returns:
            Confirmed signal series.
        """
        confirmed = raw_signal.copy()
        window = self.confirmation_window
        values = raw_signal.values
        out = np.zeros_like(values)

        for i in range(window, len(values)):
            window_vals = values[i - window : i + 1]
            if np.all(window_vals == self.LONG):
                out[i] = self.LONG
            elif np.all(window_vals == self.SHORT):
                out[i] = self.SHORT
            else:
                out[i] = self.FLAT

        confirmed = pd.Series(out, index=raw_signal.index, dtype=int)
        return confirmed

    def signal_to_positions(
        self,
        signals: pd.Series,
        max_position: float = 1.0,
    ) -> pd.Series:
        """Convert discrete signals to continuous position sizes.

        Args:
            signals: Integer signal series (LONG/FLAT/SHORT).
            max_position: Maximum absolute position size (fraction of capital).

        Returns:
            Continuous position series in range [-max_position, max_position].
        """
        return signals.astype(float) * max_position
