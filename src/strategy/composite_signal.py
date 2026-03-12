"""Composite Natural Gas Trading Signal System.

Combines multiple structural edges into a LONG / FLAT / SHORT signal:

1. **Regime detection** -- Hidden Markov Model classifies market into
   low-vol (risk-on) / high-vol (risk-off) states.
2. **Storage anomaly** -- NG storage vs 5-year seasonal average. Large
   deficits are bullish; large surpluses are bearish.
3. **Seasonal positioning** -- statistically significant monthly return
   patterns (buy spring shoulder, short pre-winter).
4. **Technical trend / momentum** -- trend-following via MA crossover,
   RSI, and 20-day return momentum.
5. **Mean reversion** -- z-score vs 200-day SMA. NG is well-documented
   as mean-reverting (OU process). Buy oversold, short overbought.
6. **Composite score** -- weighted sum of sub-signals with
   conviction-scaled position sizing and trailing stops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CompositeSignalConfig:
    """Tunable parameters for the composite signal."""

    # Regime detection -- 2-state (low-vol / high-vol)
    hmm_n_states: int = 2
    hmm_lookback: int = 504  # 2 years of training data for rolling fit
    hmm_vol_window: int = 21  # realised vol window

    # Storage anomaly
    storage_seasonal_years: int = 5  # 5-year average baseline
    storage_zscore_bull: float = -1.0  # deficit threshold (z-score)
    storage_zscore_bear: float = 1.0  # surplus threshold

    # Seasonal
    seasonal_lookback_years: int = 15  # how many years to estimate pattern

    # Technical
    fast_ma: int = 20
    slow_ma: int = 60
    rsi_period: int = 14
    rsi_ob: float = 70.0  # overbought
    rsi_os: float = 30.0  # oversold
    momentum_window: int = 20  # return momentum lookback

    # Mean reversion (NEW)
    mr_lookback: int = 200  # SMA lookback for mean-reversion z-score
    mr_zscore_long: float = -1.0  # buy when price 1 sigma below SMA
    mr_zscore_short: float = 1.0  # sell when price 1 sigma above SMA

    # Composite weights (must sum to 1)
    w_regime: float = 0.15
    w_storage: float = 0.15
    w_seasonal: float = 0.15
    w_technical: float = 0.30
    w_mean_reversion: float = 0.25

    # Signal thresholds
    long_threshold: float = 0.05  # low bar -- conviction sizing does the work
    short_threshold: float = -0.15  # short only when clearly bearish
    long_only: bool = False  # LONG / SHORT / FLAT

    # Risk management
    vol_target: float = 0.40  # annualised vol target for position sizing
    vol_lookback: int = 63  # 3-month realised vol for sizing
    max_short_size: float = 0.50  # cap short positions at 50% of long
    vol_risk_off_pctile: float = 0.93  # go FLAT above this vol percentile
    min_position_frac: float = 0.40  # floor: even weak signals get 40% sizing

    # Trailing stops (0 = disabled)
    trailing_stop_long: float = 0.0
    trailing_stop_short: float = 0.0
    stop_cooldown_days: int = 10


# ---------------------------------------------------------------------------
# Sub-signal: Regime Detection (HMM)
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Hidden Markov Model regime classification.

    Fits a 2-state Gaussian HMM on [returns, realised_vol] to label
    each day as low-volatility (risk-on) or high-volatility (risk-off).
    Low-vol regime → small positive bias; high-vol regime → FLAT/defensive.
    """

    def __init__(self, n_states: int = 2, vol_window: int = 21,
                 lookback: int = 504) -> None:
        self.n_states = n_states
        self.vol_window = vol_window
        self.lookback = lookback

    def fit_predict(self, prices: pd.Series) -> pd.DataFrame:
        """Rolling HMM regime detection.

        Returns DataFrame with columns: regime (0/1), regime_signal (-1/0/+1).
        """
        log_ret = np.log(prices / prices.shift(1)).dropna()
        vol = log_ret.rolling(self.vol_window).std().dropna()

        common = log_ret.index.intersection(vol.index)
        log_ret = log_ret.reindex(common)
        vol = vol.reindex(common)

        regime = pd.Series(np.nan, index=common, name="regime")
        regime_signal = pd.Series(0.0, index=common, name="regime_signal")

        # Rolling fit — retrain every quarter (63 days)
        refit_every = 63
        model = None
        state_map = None

        for i in range(self.lookback, len(common), refit_every):
            train_start = max(0, i - self.lookback)
            train_end = i
            pred_end = min(i + refit_every, len(common))

            X_train = np.column_stack([
                log_ret.iloc[train_start:train_end].values,
                vol.iloc[train_start:train_end].values,
            ])

            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="full",
                    n_iter=100,
                    random_state=42,
                    verbose=False,
                )
                model.fit(X_train)

                # Map states: lower vol state → risk-on (+0.5), higher vol → risk-off (-0.5)
                state_vols = model.means_[:, 1]  # vol dimension
                low_vol_state = int(np.argmin(state_vols))
                state_map = {}
                for s in range(self.n_states):
                    state_map[s] = 0.5 if s == low_vol_state else -0.5

            except Exception:
                logger.debug("HMM fit failed at index %d, reusing last model", i)
                if model is None:
                    continue

            # Predict on forward window
            X_pred = np.column_stack([
                log_ret.iloc[train_end:pred_end].values,
                vol.iloc[train_end:pred_end].values,
            ])
            if len(X_pred) == 0:
                continue

            try:
                states = model.predict(X_pred)
                idx = common[train_end:pred_end]
                regime.loc[idx] = states
                regime_signal.loc[idx] = [state_map.get(s, 0.0) for s in states]
            except Exception:
                logger.debug("HMM predict failed at index %d", i)

        return pd.DataFrame({"regime": regime, "regime_signal": regime_signal})


# ---------------------------------------------------------------------------
# Sub-signal: Storage Anomaly
# ---------------------------------------------------------------------------

class StorageAnomalySignal:
    """NG storage vs 5-year seasonal average.

    Computes a z-score of current storage vs the same-week-of-year average
    over the prior N years.  Large deficits → bullish, surpluses → bearish.
    """

    def __init__(self, seasonal_years: int = 5,
                 z_bull: float = -1.0, z_bear: float = 1.0) -> None:
        self.seasonal_years = seasonal_years
        self.z_bull = z_bull
        self.z_bear = z_bear

    def compute(self, storage: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.Series:
        """Compute storage anomaly signal aligned to daily price index.

        Args:
            storage: DataFrame with columns ['date', 'w'] (weekly BCF).
            price_index: DatetimeIndex of daily prices.

        Returns:
            Series of signal values (-1, 0, +1) on price_index.
        """
        df = storage.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        df["week"] = df.index.isocalendar().week.astype(int)
        df["year"] = df.index.year

        # 5-year seasonal average and std by week-of-year (rolling)
        records = []
        for idx_date, row in df.iterrows():
            wk = row["week"]
            yr = row["year"]
            hist = df[
                (df["week"] == wk)
                & (df["year"] >= yr - self.seasonal_years)
                & (df["year"] < yr)
            ]["w"]
            if len(hist) >= 3:
                mu = hist.mean()
                sigma = hist.std()
                z = (row["w"] - mu) / sigma if sigma > 0 else 0.0
            else:
                z = 0.0
            records.append({"date": idx_date, "storage_z": z})

        z_df = pd.DataFrame(records).set_index("date")
        z_df = z_df.sort_index()

        # Forward-fill to daily
        z_daily = z_df.reindex(price_index, method="ffill")["storage_z"].fillna(0.0)

        # Convert to signal
        signal = pd.Series(0.0, index=price_index, name="storage_signal")
        signal[z_daily <= self.z_bull] = 1.0   # deficit → bullish
        signal[z_daily >= self.z_bear] = -1.0  # surplus → bearish

        return signal


# ---------------------------------------------------------------------------
# Sub-signal: Seasonal Positioning
# ---------------------------------------------------------------------------

class SeasonalSignal:
    """Calendar-based seasonal NG signal.

    Computes average monthly returns from history and converts to a
    signal: months with historically positive returns → LONG,
    negative → SHORT, near zero → FLAT.
    """

    def __init__(self, lookback_years: int = 15) -> None:
        self.lookback_years = lookback_years

    def compute(self, prices: pd.Series) -> pd.Series:
        """Compute rolling seasonal signal.

        Uses expanding window — at each point, only uses data up to that date
        to avoid lookahead.
        """
        log_ret = np.log(prices / prices.shift(1)).dropna()
        signal = pd.Series(0.0, index=prices.index, name="seasonal_signal")

        # Compute monthly average returns expanding window
        monthly_ret = log_ret.resample("ME").sum()

        for i, date in enumerate(prices.index):
            if date.year - prices.index[0].year < 5:
                # Not enough history
                continue

            cutoff = date - pd.Timedelta(days=30)
            hist = monthly_ret[monthly_ret.index <= cutoff]
            if len(hist) < 12:
                continue

            # Average return for current month across all years
            month = date.month
            month_returns = hist[hist.index.month == month]
            if len(month_returns) < 3:
                continue

            avg = month_returns.mean()
            se = month_returns.std() / np.sqrt(len(month_returns))

            # t-stat significance — must be > 1.0 to generate signal
            if se > 0:
                t = avg / se
                if t > 1.0:
                    signal.iloc[i] = 1.0
                elif t < -1.0:
                    signal.iloc[i] = -1.0

        return signal


# ---------------------------------------------------------------------------
# Sub-signal: Technical Trend / Momentum
# ---------------------------------------------------------------------------

class TechnicalSignal:
    """Trend-following and momentum signal.

    - MA crossover: fast MA > slow MA → bullish
    - RSI filter: overbought dampens long, oversold dampens short
    - Momentum: 20-day return direction
    """

    def __init__(self, fast_ma: int = 20, slow_ma: int = 60,
                 rsi_period: int = 14, rsi_ob: float = 70.0,
                 rsi_os: float = 30.0, momentum_window: int = 20) -> None:
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.rsi_period = rsi_period
        self.rsi_ob = rsi_ob
        self.rsi_os = rsi_os
        self.momentum_window = momentum_window

    def compute(self, prices: pd.Series) -> pd.Series:
        """Compute technical composite signal."""
        fast = prices.rolling(self.fast_ma).mean()
        slow = prices.rolling(self.slow_ma).mean()

        # MA crossover component
        ma_signal = pd.Series(0.0, index=prices.index)
        ma_signal[fast > slow] = 1.0
        ma_signal[fast < slow] = -1.0

        # RSI
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        rsi_modifier = pd.Series(0.0, index=prices.index)
        rsi_modifier[rsi > self.rsi_ob] = -0.5  # overbought → dampen long
        rsi_modifier[rsi < self.rsi_os] = 0.5   # oversold → dampen short

        # Momentum
        mom = prices.pct_change(self.momentum_window)
        mom_signal = pd.Series(0.0, index=prices.index)
        mom_signal[mom > 0.02] = 0.5
        mom_signal[mom < -0.02] = -0.5

        # Blend: MA (50%) + Momentum (30%) + RSI modifier (20%)
        tech = 0.50 * ma_signal + 0.30 * mom_signal + 0.20 * rsi_modifier

        # Clip to [-1, 1]
        tech = tech.clip(-1.0, 1.0)
        tech.name = "technical_signal"
        return tech


# ---------------------------------------------------------------------------
# Sub-signal: Mean Reversion (z-score vs SMA)
# ---------------------------------------------------------------------------

class MeanReversionSignal:
    """Mean-reversion signal based on z-score vs long-term SMA.

    NG is well-documented as mean-reverting (OU process).  When price
    deviates significantly from its 200-day SMA, it tends to revert.
    The signal is the *negative* z-score, smoothly scaled:
    - price well below SMA (z < -1) -> bullish (buy the dip)
    - price well above SMA (z > +1) -> bearish (fade the spike)
    """

    def __init__(self, lookback: int = 200,
                 z_long: float = -1.0, z_short: float = 1.0) -> None:
        self.lookback = lookback
        self.z_long = z_long
        self.z_short = z_short

    def compute(self, prices: pd.Series) -> pd.Series:
        """Compute mean-reversion signal."""
        sma = prices.rolling(self.lookback).mean()
        std = prices.rolling(self.lookback).std()
        zscore = (prices - sma) / std.replace(0, np.nan)
        zscore = zscore.fillna(0.0)

        # Smooth response: negative z-score = bullish, positive = bearish
        # Scale so z = -2 -> signal +1, z = +2 -> signal -1
        signal = (-zscore / 2.0).clip(-1.0, 1.0)
        signal.name = "mean_reversion_signal"
        return signal


# ---------------------------------------------------------------------------
# Composite Signal Aggregator
# ---------------------------------------------------------------------------

class CompositeSignalEngine:
    """Aggregates sub-signals into a final LONG / FLAT / SHORT signal.

    All sub-signals are computed using only data available at each point
    (no lookahead). The composite score drives conviction-scaled position
    sizing, with trailing stops and a cooldown after stop-outs.
    """

    def __init__(self, config: CompositeSignalConfig | None = None) -> None:
        self.cfg = config or CompositeSignalConfig()

    def generate_signals(
        self,
        prices: pd.Series,
        storage: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate composite trading signals.

        Args:
            prices: Daily NG close prices (DatetimeIndex).
            storage: Optional weekly storage DataFrame (['date','w']).
                If None, storage signal is zeroed and weight redistributed.

        Returns:
            DataFrame with all sub-signals, composite score, discrete signal,
            and risk-adjusted position size.
        """
        logger.info("Generating composite NG trading signals (%d prices)", len(prices))

        # --- Regime ---
        regime_det = RegimeDetector(
            n_states=self.cfg.hmm_n_states,
            vol_window=self.cfg.hmm_vol_window,
            lookback=self.cfg.hmm_lookback,
        )
        regime_df = regime_det.fit_predict(prices)

        # --- Storage ---
        if storage is not None and len(storage) > 0:
            storage_sig = StorageAnomalySignal(
                seasonal_years=self.cfg.storage_seasonal_years,
                z_bull=self.cfg.storage_zscore_bull,
                z_bear=self.cfg.storage_zscore_bear,
            )
            storage_signal = storage_sig.compute(storage, prices.index)
            w_storage = self.cfg.w_storage
        else:
            storage_signal = pd.Series(0.0, index=prices.index, name="storage_signal")
            w_storage = 0.0
            logger.warning("No storage data -- storage weight redistributed")

        # --- Seasonal ---
        seasonal_sig = SeasonalSignal(lookback_years=self.cfg.seasonal_lookback_years)
        seasonal_signal = seasonal_sig.compute(prices)

        # --- Technical ---
        tech_sig = TechnicalSignal(
            fast_ma=self.cfg.fast_ma,
            slow_ma=self.cfg.slow_ma,
            rsi_period=self.cfg.rsi_period,
            rsi_ob=self.cfg.rsi_ob,
            rsi_os=self.cfg.rsi_os,
            momentum_window=self.cfg.momentum_window,
        )
        tech_signal = tech_sig.compute(prices)

        # --- Mean Reversion ---
        mr_sig = MeanReversionSignal(
            lookback=self.cfg.mr_lookback,
            z_long=self.cfg.mr_zscore_long,
            z_short=self.cfg.mr_zscore_short,
        )
        mr_signal = mr_sig.compute(prices)

        # --- Align all to price index ---
        regime_signal = regime_df["regime_signal"].reindex(prices.index).fillna(0.0)

        # --- Weights ---
        w_regime = self.cfg.w_regime
        w_seasonal = self.cfg.w_seasonal
        w_technical = self.cfg.w_technical
        w_mr = self.cfg.w_mean_reversion

        # Redistribute storage weight if absent
        if w_storage == 0.0:
            total_other = w_regime + w_seasonal + w_technical + w_mr
            if total_other > 0:
                scale = 1.0 / total_other
                w_regime *= scale
                w_seasonal *= scale
                w_technical *= scale
                w_mr *= scale

        # --- Composite score ---
        composite = (
            w_regime * regime_signal
            + w_storage * storage_signal
            + w_seasonal * seasonal_signal
            + w_technical * tech_signal
            + w_mr * mr_signal
        )

        # --- Map to discrete signal ---
        signal = pd.Series(0, index=prices.index, dtype=int, name="signal")
        signal[composite >= self.cfg.long_threshold] = 1
        if not self.cfg.long_only:
            signal[composite <= self.cfg.short_threshold] = -1

        # --- Risk management: volatility targeting ---
        log_ret = np.log(prices / prices.shift(1))
        realised_vol = log_ret.rolling(self.cfg.vol_lookback).std() * np.sqrt(252)
        realised_vol = realised_vol.bfill().clip(lower=0.05)

        # Vol-target scalar
        vol_scalar = (self.cfg.vol_target / realised_vol).clip(upper=1.5)

        # Risk-off filter: go FLAT when realised vol exceeds threshold
        vol_rank = realised_vol.expanding(min_periods=252).rank(pct=True)
        risk_off_mask = vol_rank > self.cfg.vol_risk_off_pctile
        signal[risk_off_mask] = 0

        # --- Conviction-scaled position sizing ---
        # Map |composite_score| to [min_position_frac, 1.0] range so even
        # marginal signals get a meaningful position, strong signals go full.
        raw_conviction = composite.abs().clip(upper=0.8)
        floor = self.cfg.min_position_frac
        conviction = floor + (1.0 - floor) * (raw_conviction / 0.8)
        position_size = signal.astype(float) * conviction * vol_scalar
        if not self.cfg.long_only:
            position_size = position_size.clip(
                lower=-self.cfg.max_short_size, upper=1.0,
            )
        else:
            position_size = position_size.clip(lower=0.0, upper=1.0)

        # --- Trailing stops with cooldown ---
        if self.cfg.trailing_stop_long > 0 or self.cfg.trailing_stop_short > 0:
            signal, position_size = self._apply_trailing_stops(
                prices, signal, position_size,
            )

        # --- Assemble output ---
        result = pd.DataFrame({
            "price": prices,
            "regime_signal": regime_signal,
            "storage_signal": storage_signal,
            "seasonal_signal": seasonal_signal,
            "technical_signal": tech_signal,
            "mean_reversion_signal": mr_signal,
            "composite_score": composite,
            "signal": signal,
            "realised_vol": realised_vol,
            "vol_scalar": vol_scalar,
            "position_size": position_size,
        })

        # Log summary
        n_long = (signal == 1).sum()
        n_short = (signal == -1).sum()
        n_flat = (signal == 0).sum()
        logger.info(
            "Composite signals: LONG=%d (%.1f%%), SHORT=%d (%.1f%%), FLAT=%d (%.1f%%)",
            n_long, 100 * n_long / len(signal),
            n_short, 100 * n_short / len(signal),
            n_flat, 100 * n_flat / len(signal),
        )

        return result

    # ------------------------------------------------------------------
    # Trailing stops
    # ------------------------------------------------------------------

    def _apply_trailing_stops(
        self,
        prices: pd.Series,
        signal: pd.Series,
        position_size: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """Apply trailing stops with cooldown after stop-out.

        When a position's unrealised loss from its peak (long) or trough
        (short) exceeds the trailing stop threshold, the position is closed
        and no new position is taken for ``stop_cooldown_days``.
        """
        sig = signal.copy()
        ps = position_size.copy()
        cfg = self.cfg

        current_dir = 0  # 0=flat, 1=long, -1=short
        peak = 0.0
        trough = float("inf")
        cooldown_remaining = 0

        for i in range(len(sig)):
            price = prices.iloc[i]
            raw_signal = sig.iloc[i]

            # Cooldown: force flat
            if cooldown_remaining > 0:
                sig.iloc[i] = 0
                ps.iloc[i] = 0.0
                cooldown_remaining -= 1
                if cooldown_remaining == 0:
                    current_dir = 0
                continue

            # Check trailing stop on existing position
            if current_dir == 1:
                peak = max(peak, price)
                dd = (price - peak) / peak
                if dd < -cfg.trailing_stop_long:
                    sig.iloc[i] = 0
                    ps.iloc[i] = 0.0
                    current_dir = 0
                    cooldown_remaining = cfg.stop_cooldown_days
                    continue
            elif current_dir == -1:
                trough = min(trough, price)
                runup = (price - trough) / trough if trough > 0 else 0.0
                if runup > cfg.trailing_stop_short:
                    sig.iloc[i] = 0
                    ps.iloc[i] = 0.0
                    current_dir = 0
                    cooldown_remaining = cfg.stop_cooldown_days
                    continue

            # Track direction changes
            new_dir = int(raw_signal)
            if new_dir != current_dir:
                current_dir = new_dir
                peak = price
                trough = price

        return sig, ps
