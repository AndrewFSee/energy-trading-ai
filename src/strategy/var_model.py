"""Portfolio Value-at-Risk and Stress-Testing Suite.

Implements a professional-grade risk analytics engine with multiple
VaR methodologies, expected shortfall (CVaR), and energy-specific
stress scenarios:

* **Historical Simulation** — Non-parametric VaR/CVaR from actual returns.
* **Parametric (Gaussian)** — Closed-form VaR under normality assumption.
* **Parametric (Student-t)** — Captures fat tails in energy returns.
* **Cornish-Fisher** — Adjusts Gaussian VaR for skewness and kurtosis.
* **EWMA** — Exponentially weighted volatility for responsive VaR.
* **Monte Carlo** — Simulated VaR with optional GARCH volatility.
* **Component VaR** — Marginal contribution of each portfolio position.
* **Stress Testing** — Energy-specific scenario analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VaRResult:
    """VaR and CVaR computation result.

    Attributes:
        method: Methodology name.
        confidence: Confidence level (e.g. 0.99).
        horizon_days: Risk horizon in trading days.
        var: Value at Risk (positive = loss).
        cvar: Conditional VaR / Expected Shortfall.
        var_dollar: VaR in dollar terms.
        cvar_dollar: CVaR in dollar terms.
    """

    method: str = ""
    confidence: float = 0.99
    horizon_days: int = 1
    var: float = 0.0
    cvar: float = 0.0
    var_dollar: float = 0.0
    cvar_dollar: float = 0.0


@dataclass
class StressResult:
    """Stress-test scenario result.

    Attributes:
        scenario: Scenario name.
        description: Brief description of the scenario.
        portfolio_pnl: Portfolio P&L under the scenario.
        portfolio_pnl_pct: P&L as percentage of portfolio value.
        asset_impacts: Per-asset P&L breakdown.
        worst_asset: Name of the worst-performing asset.
        worst_loss_pct: Loss on the worst-performing asset.
    """

    scenario: str = ""
    description: str = ""
    portfolio_pnl: float = 0.0
    portfolio_pnl_pct: float = 0.0
    asset_impacts: dict[str, float] = field(default_factory=dict)
    worst_asset: str = ""
    worst_loss_pct: float = 0.0


@dataclass
class RiskDashboard:
    """Comprehensive risk dashboard.

    Attributes:
        portfolio_value: Current portfolio notional.
        var_results: VaR by each methodology.
        component_var: Marginal VaR contribution by asset.
        stress_results: Stress test outcomes.
        tail_statistics: Distributional statistics of returns.
    """

    portfolio_value: float = 0.0
    var_results: list[VaRResult] = field(default_factory=list)
    component_var: dict[str, float] = field(default_factory=dict)
    stress_results: list[StressResult] = field(default_factory=list)
    tail_statistics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Energy-specific stress scenarios
# ---------------------------------------------------------------------------

ENERGY_STRESS_SCENARIOS = {
    "polar_vortex": {
        "description": "Polar vortex — extreme cold snap driving heating demand spike "
                       "(modelled on Jan 2014 / Feb 2021 events)",
        "shocks": {
            "natural_gas": +0.40,   # +40% gas price spike
            "power": +0.60,         # +60% power price spike
            "crude_oil": +0.05,     # modest crude impact
            "renewables": -0.15,    # wind/solar output drops (icing)
        },
    },
    "hurricane_gulf": {
        "description": "Category 4 hurricane hitting Gulf Coast — disrupts offshore production "
                       "and refining (modelled on Hurricanes Harvey/Ida)",
        "shocks": {
            "natural_gas": +0.25,
            "power": +0.10,
            "crude_oil": +0.15,
            "renewables": -0.05,
        },
    },
    "pipeline_outage": {
        "description": "Major pipeline outage — reduces gas deliverability to Northeast "
                       "(modelled on Algonquin basis blowout events)",
        "shocks": {
            "natural_gas": +0.15,   # Henry Hub moderate
            "power": +0.35,         # NE power spikes
            "crude_oil": +0.02,
            "renewables": 0.00,
        },
    },
    "demand_collapse": {
        "description": "Severe demand collapse — warm winter + economic downturn "
                       "(modelled on COVID-19 March 2020 + weather)",
        "shocks": {
            "natural_gas": -0.30,
            "power": -0.25,
            "crude_oil": -0.35,
            "renewables": +0.05,
        },
    },
    "lng_export_surge": {
        "description": "LNG export surge — European energy crisis drives US exports "
                       "to record levels (modelled on 2022 events)",
        "shocks": {
            "natural_gas": +0.50,
            "power": +0.20,
            "crude_oil": +0.10,
            "renewables": 0.00,
        },
    },
    "renewables_intermittency": {
        "description": "Renewables intermittency crisis — extended cloud cover + low wind "
                       "across multiple ISOs simultaneously",
        "shocks": {
            "natural_gas": +0.10,
            "power": +0.30,
            "crude_oil": 0.00,
            "renewables": -0.40,
        },
    },
}


# ---------------------------------------------------------------------------
# VaR Engine
# ---------------------------------------------------------------------------

class VaREngine:
    """Multi-method Value-at-Risk computation engine.

    Computes portfolio risk using six different methodologies,
    providing a comprehensive view of risk from parametric,
    non-parametric, and simulation perspectives.
    """

    def __init__(
        self,
        confidence: float = 0.99,
        horizon_days: int = 1,
        ewma_lambda: float = 0.94,
        mc_simulations: int = 10000,
    ) -> None:
        """Initialise the VaR engine.

        Args:
            confidence: VaR confidence level (e.g. 0.99 = 99%).
            horizon_days: Risk horizon (1 = daily, 10 ≈ 2 weeks).
            ewma_lambda: Decay factor for EWMA volatility.
            mc_simulations: Number of Monte Carlo simulations.
        """
        self.confidence = confidence
        self.horizon = horizon_days
        self.ewma_lambda = ewma_lambda
        self.n_sim = mc_simulations

    # --- Historical Simulation ---

    def historical_var(
        self,
        returns: pd.Series | pd.DataFrame,
        portfolio_value: float = 1e6,
    ) -> VaRResult:
        """Historical simulation VaR.

        Non-parametric: uses the actual empirical return distribution.
        Makes no distributional assumptions — excellent for fat-tailed
        energy returns.

        Args:
            returns: Daily return series or multi-asset returns.
            portfolio_value: Portfolio notional for dollar VaR.

        Returns:
            ``VaRResult`` with historical VaR/CVaR.
        """
        if isinstance(returns, pd.DataFrame):
            rets = returns.sum(axis=1).dropna().values
        else:
            rets = returns.dropna().values

        # Scale to horizon
        if self.horizon > 1:
            # Use overlapping multi-day returns
            multi = pd.Series(rets).rolling(self.horizon).sum().dropna().values
        else:
            multi = rets

        alpha = 1 - self.confidence
        var = -float(np.percentile(multi, alpha * 100))
        tail = multi[multi <= -var]
        cvar = -float(tail.mean()) if len(tail) > 0 else var

        return VaRResult(
            method="Historical Simulation",
            confidence=self.confidence,
            horizon_days=self.horizon,
            var=var,
            cvar=cvar,
            var_dollar=var * portfolio_value,
            cvar_dollar=cvar * portfolio_value,
        )

    # --- Parametric (Gaussian) ---

    def parametric_normal_var(
        self,
        returns: pd.Series | pd.DataFrame,
        portfolio_value: float = 1e6,
    ) -> VaRResult:
        """Parametric VaR under Gaussian assumption.

        Closed-form: VaR = μ + z_α × σ × √h

        Fast but understates tail risk for energy commodities.

        Args:
            returns: Daily returns.
            portfolio_value: Portfolio notional.

        Returns:
            ``VaRResult`` with parametric normal VaR.
        """
        if isinstance(returns, pd.DataFrame):
            rets = returns.sum(axis=1).dropna().values
        else:
            rets = returns.dropna().values

        mu = rets.mean()
        sigma = rets.std()
        z = stats.norm.ppf(1 - self.confidence)  # negative

        var = -(mu * self.horizon + z * sigma * np.sqrt(self.horizon))
        # CVaR for normal: E[X | X < -VaR]
        cvar = -(mu * self.horizon - sigma * np.sqrt(self.horizon)
                 * stats.norm.pdf(z) / (1 - self.confidence))

        return VaRResult(
            method="Parametric (Normal)",
            confidence=self.confidence,
            horizon_days=self.horizon,
            var=var,
            cvar=cvar,
            var_dollar=var * portfolio_value,
            cvar_dollar=cvar * portfolio_value,
        )

    # --- Parametric (Student-t) ---

    def parametric_t_var(
        self,
        returns: pd.Series | pd.DataFrame,
        portfolio_value: float = 1e6,
    ) -> VaRResult:
        """Parametric VaR with Student-t distribution.

        Better captures fat tails typical in energy commodity returns.
        Degrees of freedom are estimated via MLE fit.

        Args:
            returns: Daily returns.
            portfolio_value: Portfolio notional.

        Returns:
            ``VaRResult`` with Student-t VaR.
        """
        if isinstance(returns, pd.DataFrame):
            rets = returns.sum(axis=1).dropna().values
        else:
            rets = returns.dropna().values

        # Fit Student-t distribution
        df_t, loc_t, scale_t = stats.t.fit(rets)
        z_t = stats.t.ppf(1 - self.confidence, df_t)

        var = -(loc_t * self.horizon + z_t * scale_t * np.sqrt(self.horizon))

        # CVaR for Student-t
        alpha = 1 - self.confidence
        x_alpha = stats.t.ppf(alpha, df_t)
        cvar_standardized = -(
            stats.t.pdf(x_alpha, df_t) / alpha
            * (df_t + x_alpha ** 2) / (df_t - 1)
        )
        cvar = -(loc_t * self.horizon + cvar_standardized * scale_t * np.sqrt(self.horizon))

        return VaRResult(
            method=f"Parametric (Student-t, df={df_t:.1f})",
            confidence=self.confidence,
            horizon_days=self.horizon,
            var=var,
            cvar=cvar,
            var_dollar=var * portfolio_value,
            cvar_dollar=cvar * portfolio_value,
        )

    # --- Cornish-Fisher ---

    def cornish_fisher_var(
        self,
        returns: pd.Series | pd.DataFrame,
        portfolio_value: float = 1e6,
    ) -> VaRResult:
        """Cornish-Fisher expansion VaR.

        Adjusts the Gaussian quantile for skewness and excess kurtosis,
        providing a semi-parametric correction without full distributional
        assumptions.

        VaR_CF = μ + z_CF × σ × √h

        where z_CF = z + (z²-1)S/6 + (z³-3z)K/24 - (2z³-5z)S²/36

        Args:
            returns: Daily returns.
            portfolio_value: Portfolio notional.

        Returns:
            ``VaRResult`` with Cornish-Fisher VaR.
        """
        if isinstance(returns, pd.DataFrame):
            rets = returns.sum(axis=1).dropna().values
        else:
            rets = returns.dropna().values

        mu = rets.mean()
        sigma = rets.std()
        skew = float(pd.Series(rets).skew())
        kurt = float(pd.Series(rets).kurtosis())  # excess kurtosis

        z = stats.norm.ppf(1 - self.confidence)

        # Cornish-Fisher adjustment
        z_cf = (
            z
            + (z ** 2 - 1) * skew / 6
            + (z ** 3 - 3 * z) * kurt / 24
            - (2 * z ** 3 - 5 * z) * skew ** 2 / 36
        )

        var = -(mu * self.horizon + z_cf * sigma * np.sqrt(self.horizon))
        # Approximate CVaR as 1.1-1.3× VaR for Cornish-Fisher
        cvar = var * (1 + 0.3 * (1 - self.confidence))

        return VaRResult(
            method="Cornish-Fisher",
            confidence=self.confidence,
            horizon_days=self.horizon,
            var=var,
            cvar=cvar,
            var_dollar=var * portfolio_value,
            cvar_dollar=cvar * portfolio_value,
        )

    # --- EWMA ---

    def ewma_var(
        self,
        returns: pd.Series | pd.DataFrame,
        portfolio_value: float = 1e6,
    ) -> VaRResult:
        """EWMA (RiskMetrics) VaR.

        Uses exponentially weighted moving average volatility (λ = 0.94)
        for a more responsive VaR estimate that adapts quickly to
        volatility regime changes — crucial for energy markets.

        Args:
            returns: Daily returns.
            portfolio_value: Portfolio notional.

        Returns:
            ``VaRResult`` with EWMA VaR.
        """
        if isinstance(returns, pd.DataFrame):
            rets = returns.sum(axis=1).dropna().values
        else:
            rets = returns.dropna().values

        # EWMA variance
        lam = self.ewma_lambda
        var_t = np.zeros(len(rets))
        var_t[0] = rets[0] ** 2
        for i in range(1, len(rets)):
            var_t[i] = lam * var_t[i - 1] + (1 - lam) * rets[i - 1] ** 2

        # Use the last EWMA volatility
        sigma_ewma = np.sqrt(var_t[-1])
        mu = rets.mean()
        z = stats.norm.ppf(1 - self.confidence)

        var = -(mu * self.horizon + z * sigma_ewma * np.sqrt(self.horizon))
        cvar = -(mu * self.horizon - sigma_ewma * np.sqrt(self.horizon)
                 * stats.norm.pdf(z) / (1 - self.confidence))

        return VaRResult(
            method=f"EWMA (λ={lam})",
            confidence=self.confidence,
            horizon_days=self.horizon,
            var=var,
            cvar=cvar,
            var_dollar=var * portfolio_value,
            cvar_dollar=cvar * portfolio_value,
        )

    # --- Monte Carlo ---

    def monte_carlo_var(
        self,
        returns: pd.Series | pd.DataFrame,
        portfolio_value: float = 1e6,
        seed: int = 42,
    ) -> VaRResult:
        """Monte Carlo VaR with bootstrapped returns.

        Resamples from the historical return distribution with
        replacement, capturing empirical fat tails and auto-correlation.

        Args:
            returns: Daily returns.
            portfolio_value: Portfolio notional.
            seed: Random seed.

        Returns:
            ``VaRResult`` with Monte Carlo VaR.
        """
        rng = np.random.default_rng(seed)

        if isinstance(returns, pd.DataFrame):
            rets = returns.sum(axis=1).dropna().values
        else:
            rets = returns.dropna().values

        # Bootstrap simulation
        simulated = np.zeros(self.n_sim)
        for i in range(self.n_sim):
            sampled = rng.choice(rets, size=self.horizon, replace=True)
            simulated[i] = sampled.sum()

        alpha = 1 - self.confidence
        var = -float(np.percentile(simulated, alpha * 100))
        tail = simulated[simulated <= -var]
        cvar = -float(tail.mean()) if len(tail) > 0 else var

        return VaRResult(
            method="Monte Carlo (Bootstrap)",
            confidence=self.confidence,
            horizon_days=self.horizon,
            var=var,
            cvar=cvar,
            var_dollar=var * portfolio_value,
            cvar_dollar=cvar * portfolio_value,
        )

    # --- Component VaR ---

    def component_var(
        self,
        returns: pd.DataFrame,
        weights: dict[str, float],
        portfolio_value: float = 1e6,
    ) -> dict[str, float]:
        """Compute component VaR — each asset's marginal contribution.

        Component VaR decomposes portfolio VaR into additive contributions
        from each position:
            ComponentVaR_i = w_i × β_i × PortfolioVaR

        where β_i = Cov(r_i, r_p) / Var(r_p)

        Args:
            returns: DataFrame with per-asset daily returns.
            weights: Portfolio weights per asset.
            portfolio_value: Portfolio notional.

        Returns:
            Dictionary of {asset: component_var_dollar}.
        """
        assets = [a for a in weights if a in returns.columns]
        if not assets:
            return {}

        w = np.array([weights[a] for a in assets])
        R = returns[assets].dropna().values
        port_returns = R @ w

        cov_matrix = np.cov(R.T)
        port_var_return = np.dot(w, np.dot(cov_matrix, w))

        if port_var_return < 1e-12:
            return {a: 0.0 for a in assets}

        # Beta of each asset to portfolio
        cov_with_port = cov_matrix @ w
        beta = cov_with_port / port_var_return

        # Portfolio VaR
        z = stats.norm.ppf(self.confidence)
        port_sigma = np.sqrt(port_var_return)
        port_var = z * port_sigma * np.sqrt(self.horizon)

        components = {}
        for i, asset in enumerate(assets):
            components[asset] = float(w[i] * beta[i] * port_var * portfolio_value)

        return components

    # --- Tail Statistics ---

    def tail_statistics(self, returns: pd.Series | pd.DataFrame) -> dict[str, float]:
        """Compute distributional statistics relevant for risk.

        Args:
            returns: Daily returns.

        Returns:
            Dictionary of tail statistics.
        """
        if isinstance(returns, pd.DataFrame):
            rets = returns.sum(axis=1).dropna().values
        else:
            rets = returns.dropna().values

        return {
            "mean_daily": float(np.mean(rets)),
            "std_daily": float(np.std(rets)),
            "annualized_vol": float(np.std(rets) * np.sqrt(252)),
            "skewness": float(pd.Series(rets).skew()),
            "excess_kurtosis": float(pd.Series(rets).kurtosis()),
            "min_daily_return": float(np.min(rets)),
            "max_daily_return": float(np.max(rets)),
            "worst_5_avg": float(np.sort(rets)[:5].mean()),
            "jarque_bera_stat": float(stats.jarque_bera(rets).statistic),
            "jarque_bera_pval": float(stats.jarque_bera(rets).pvalue),
            "pct_negative_days": float((rets < 0).mean() * 100),
            "max_drawdown": float(self._max_drawdown(rets)),
        }

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        """Compute maximum drawdown from return series."""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return float(drawdown.min())

    # --- Full Analysis ---

    def full_analysis(
        self,
        returns: pd.Series | pd.DataFrame,
        portfolio_value: float = 1e6,
    ) -> list[VaRResult]:
        """Run all VaR methodologies and return results list.

        Args:
            returns: Daily returns.
            portfolio_value: Portfolio notional.

        Returns:
            List of ``VaRResult`` from each methodology.
        """
        methods = [
            self.historical_var,
            self.parametric_normal_var,
            self.parametric_t_var,
            self.cornish_fisher_var,
            self.ewma_var,
            self.monte_carlo_var,
        ]
        results = []
        for method in methods:
            try:
                res = method(returns, portfolio_value)
                results.append(res)
            except Exception as e:
                logger.warning("VaR method failed: %s", e)
        return results


# ---------------------------------------------------------------------------
# Stress Testing Engine
# ---------------------------------------------------------------------------

class StressTestEngine:
    """Energy-specific stress testing.

    Applies predefined and custom stress scenarios to a portfolio,
    computing scenario P&L and identifying concentration risks.
    """

    def __init__(
        self,
        scenarios: dict[str, dict] | None = None,
    ) -> None:
        """Initialise with scenario definitions.

        Args:
            scenarios: Custom scenarios.  If ``None``, uses built-in
                energy stress scenarios.
        """
        self.scenarios = scenarios or ENERGY_STRESS_SCENARIOS

    def run_scenario(
        self,
        scenario_name: str,
        positions: dict[str, float],
        asset_exposures: dict[str, str],
        portfolio_value: float = 1e6,
    ) -> StressResult:
        """Apply a single stress scenario to the portfolio.

        Args:
            scenario_name: Key into the scenarios dictionary.
            positions: {asset_name: notional_exposure} in dollars.
            asset_exposures: {asset_name: risk_factor} mapping each
                position to its primary risk driver (e.g. "natural_gas").
            portfolio_value: Total portfolio value for percentage calc.

        Returns:
            ``StressResult`` with scenario P&L breakdown.
        """
        if scenario_name not in self.scenarios:
            available = ", ".join(self.scenarios.keys())
            raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")

        scenario = self.scenarios[scenario_name]
        shocks = scenario["shocks"]

        asset_pnls = {}
        total_pnl = 0.0

        for asset, exposure in positions.items():
            risk_factor = asset_exposures.get(asset, "natural_gas")
            shock = shocks.get(risk_factor, 0.0)
            pnl = exposure * shock
            asset_pnls[asset] = pnl
            total_pnl += pnl

        worst_asset = min(asset_pnls, key=asset_pnls.get) if asset_pnls else ""
        worst_loss = asset_pnls.get(worst_asset, 0.0) / abs(positions.get(worst_asset, 1)) * 100 if worst_asset else 0.0

        return StressResult(
            scenario=scenario_name,
            description=scenario["description"],
            portfolio_pnl=total_pnl,
            portfolio_pnl_pct=total_pnl / portfolio_value * 100 if portfolio_value else 0,
            asset_impacts=asset_pnls,
            worst_asset=worst_asset,
            worst_loss_pct=worst_loss,
        )

    def run_all(
        self,
        positions: dict[str, float],
        asset_exposures: dict[str, str],
        portfolio_value: float = 1e6,
    ) -> list[StressResult]:
        """Run all stress scenarios.

        Args:
            positions: Portfolio positions.
            asset_exposures: Risk factor mappings.
            portfolio_value: Portfolio value.

        Returns:
            List of ``StressResult`` for each scenario.
        """
        results = []
        for name in self.scenarios:
            res = self.run_scenario(name, positions, asset_exposures, portfolio_value)
            results.append(res)
        return sorted(results, key=lambda r: r.portfolio_pnl)

    def reverse_stress_test(
        self,
        positions: dict[str, float],
        asset_exposures: dict[str, str],
        portfolio_value: float = 1e6,
        loss_threshold: float = 0.10,
    ) -> dict[str, float]:
        """Reverse stress test — find shocks that cause target loss.

        For each risk factor, compute the shock size needed to cause
        the specified loss threshold (e.g. 10% of portfolio).

        Args:
            positions: Portfolio positions.
            asset_exposures: Risk factor mappings.
            portfolio_value: Portfolio value.
            loss_threshold: Target loss as fraction (e.g. 0.10 = 10%).

        Returns:
            {risk_factor: required_shock} to cause the threshold loss.
        """
        target_loss = portfolio_value * loss_threshold

        # Group positions by risk factor
        factor_exposure = {}
        for asset, exposure in positions.items():
            factor = asset_exposures.get(asset, "natural_gas")
            factor_exposure.setdefault(factor, 0.0)
            factor_exposure[factor] += exposure

        result = {}
        for factor, exposure in factor_exposure.items():
            if abs(exposure) > 1e-6:
                required_shock = -target_loss / exposure
                result[factor] = required_shock

        return result


# ---------------------------------------------------------------------------
# BacktestVaR — validate VaR model accuracy
# ---------------------------------------------------------------------------

class VaRBacktester:
    """Backtest VaR model accuracy via coverage test.

    Computes the number of VaR exceedances (days where actual loss
    exceeded VaR estimate) and performs Kupiec's POF test.
    """

    @staticmethod
    def backtest(
        returns: pd.Series,
        var_estimates: pd.Series,
        confidence: float = 0.99,
    ) -> dict[str, float]:
        """Compute VaR backtesting statistics.

        Args:
            returns: Actual daily returns.
            var_estimates: Rolling VaR estimates (positive = loss).
            confidence: VaR confidence level.

        Returns:
            Dictionary of backtesting metrics.
        """
        aligned = pd.concat([returns, var_estimates], axis=1, join="inner").dropna()
        if len(aligned) == 0:
            return {}
        aligned.columns = ["return", "var"]

        # Exceedance = when actual loss exceeds VaR
        exceedances = aligned["return"] < -aligned["var"]
        n_exc = int(exceedances.sum())
        n_total = len(aligned)
        exc_rate = n_exc / n_total if n_total > 0 else 0
        expected_rate = 1 - confidence

        # Kupiec POF test (proportion of failures)
        if 0 < exc_rate < 1 and n_total > 0:
            lr_stat = -2 * (
                n_exc * np.log(expected_rate / exc_rate)
                + (n_total - n_exc) * np.log((1 - expected_rate) / (1 - exc_rate))
            )
            p_value = 1 - stats.chi2.cdf(lr_stat, 1)
        else:
            lr_stat = 0.0
            p_value = 1.0

        return {
            "n_observations": n_total,
            "n_exceedances": n_exc,
            "exceedance_rate": exc_rate,
            "expected_rate": expected_rate,
            "kupiec_lr_stat": lr_stat,
            "kupiec_p_value": p_value,
            "model_accepted": p_value > 0.05,
        }

    @staticmethod
    def rolling_var(
        returns: pd.Series,
        window: int = 252,
        confidence: float = 0.99,
    ) -> pd.Series:
        """Compute rolling historical VaR for backtesting.

        Args:
            returns: Daily returns.
            window: Rolling window size.
            confidence: VaR confidence level.

        Returns:
            Series of rolling VaR estimates.
        """
        alpha = 1 - confidence
        return -returns.rolling(window, min_periods=60).quantile(alpha)


# ---------------------------------------------------------------------------
# Convenience: run full risk dashboard
# ---------------------------------------------------------------------------

def build_risk_dashboard(
    returns: pd.DataFrame | pd.Series,
    positions: dict[str, float] | None = None,
    asset_exposures: dict[str, str] | None = None,
    portfolio_value: float = 1e6,
    confidence: float = 0.99,
    horizon_days: int = 1,
) -> RiskDashboard:
    """Build a complete risk dashboard in one call.

    Args:
        returns: Daily returns (Series for single asset, DataFrame for multi).
        positions: Portfolio positions (notional per asset).
        asset_exposures: {asset: risk_factor} for stress testing.
        portfolio_value: Total portfolio value.
        confidence: VaR confidence level.
        horizon_days: Risk horizon.

    Returns:
        ``RiskDashboard`` with VaR, component VaR, stress tests, and stats.
    """
    engine = VaREngine(confidence=confidence, horizon_days=horizon_days)

    # VaR methods
    var_results = engine.full_analysis(returns, portfolio_value)

    # Component VaR (if multi-asset)
    comp_var = {}
    if isinstance(returns, pd.DataFrame) and positions:
        weights = {}
        total_pos = sum(abs(v) for v in positions.values())
        if total_pos > 0:
            weights = {k: v / total_pos for k, v in positions.items()}
        comp_var = engine.component_var(returns, weights, portfolio_value)

    # Stress tests
    stress_results = []
    if positions and asset_exposures:
        stress_engine = StressTestEngine()
        stress_results = stress_engine.run_all(positions, asset_exposures, portfolio_value)

    # Tail stats
    tail_stats = engine.tail_statistics(returns)

    return RiskDashboard(
        portfolio_value=portfolio_value,
        var_results=var_results,
        component_var=comp_var,
        stress_results=stress_results,
        tail_statistics=tail_stats,
    )
