"""Natural Gas Storage Valuation Engine.

Implements intrinsic and extrinsic valuation of gas storage facilities:

* **Intrinsic value** — Optimal injection/withdrawal schedule against a
  forward curve solved via linear programming (calendar-spread arbitrage).
* **Rolling intrinsic** — Re-optimization at each date as prices evolve.
* **Extrinsic value** — Real-option value above intrinsic, computed via
  Least-Squares Monte Carlo (Longstaff-Schwartz) with an Ornstein-
  Uhlenbeck price process calibrated to historical Henry Hub data.

Typical use on a gas trading desk: value leased storage capacity, compute
Greeks for hedging, and assess whether to buy/sell storage capacity rights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward curve construction
# ---------------------------------------------------------------------------

class ForwardCurve:
    """Build a monthly forward curve from spot price + seasonal factors.

    In production desks, the forward curve comes from broker quotes
    (ICE, CME).  Here we construct a synthetic curve using historical
    seasonal patterns—the same seed approach used by many analytics teams
    when live market data is unavailable.

    Attributes:
        seasonal_factors: Array of 12 multiplicative factors (Jan-Dec).
        base_price: Current prompt-month price used as anchor.
    """

    def __init__(self, price_history: pd.Series) -> None:
        """Calibrate seasonal factors from historical daily prices.

        Args:
            price_history: Daily NG=F close prices with a DatetimeIndex.
        """
        monthly = price_history.resample("MS").mean().dropna()
        overall_mean = monthly.mean()
        self.seasonal_factors = np.ones(12)
        for m in range(12):
            month_vals = monthly[monthly.index.month == m + 1]
            if len(month_vals) > 0:
                self.seasonal_factors[m] = month_vals.mean() / overall_mean
        # Normalise so average factor == 1.0
        self.seasonal_factors /= self.seasonal_factors.mean()
        self.base_price = float(price_history.iloc[-1])
        logger.info(
            "ForwardCurve calibrated: base=$%.2f, winter_premium=%.1f%%",
            self.base_price,
            (self.seasonal_factors[[0, 1, 11]].mean() - 1) * 100,
        )

    def curve(self, start_month: int, n_months: int = 12) -> np.ndarray:
        """Generate a forward curve from the given starting month.

        Args:
            start_month: Starting month (1=Jan, 12=Dec).
            n_months: Number of forward months to generate.

        Returns:
            Array of monthly forward prices ($/MMBtu).
        """
        prices = np.zeros(n_months)
        for i in range(n_months):
            m = (start_month - 1 + i) % 12
            prices[i] = self.base_price * self.seasonal_factors[m]
        return prices

    def curve_at_price(self, spot: float, start_month: int, n_months: int = 12) -> np.ndarray:
        """Generate a forward curve anchored at a specific spot price.

        Args:
            spot: Current spot/prompt price.
            start_month: Starting month.
            n_months: Number of months.

        Returns:
            Array of monthly forward prices.
        """
        prices = np.zeros(n_months)
        for i in range(n_months):
            m = (start_month - 1 + i) % 12
            prices[i] = spot * self.seasonal_factors[m]
        return prices


# ---------------------------------------------------------------------------
# Physical storage parameters
# ---------------------------------------------------------------------------

@dataclass
class StorageAsset:
    """Physical parameters of a natural gas storage facility.

    Default values approximate a Gulf Coast salt-cavern facility.
    Depleted reservoirs have lower injection/withdrawal rates but
    larger capacity.

    Attributes:
        name: Facility identifier.
        capacity_bcf: Maximum working gas capacity (Bcf).
        min_inventory_bcf: Minimum cushion gas (Bcf).
        max_injection_bcf_day: Maximum daily injection rate (Bcf/day).
        max_withdrawal_bcf_day: Maximum daily withdrawal rate (Bcf/day).
        injection_cost: Variable injection cost ($/MMBtu).
        withdrawal_cost: Variable withdrawal cost ($/MMBtu).
        fuel_loss_pct: Fuel/shrinkage loss on cycling (fraction).
        ratchets: If True, injection/withdrawal rates depend on fill level.
    """

    name: str = "Gulf_Coast_Salt_Cavern"
    capacity_bcf: float = 10.0
    min_inventory_bcf: float = 0.5
    max_injection_bcf_day: float = 0.200  # 200 MMcf/d
    max_withdrawal_bcf_day: float = 0.400  # 400 MMcf/d
    injection_cost: float = 0.03  # $/MMBtu
    withdrawal_cost: float = 0.02  # $/MMBtu
    fuel_loss_pct: float = 0.02  # 2% cycling loss
    ratchets: bool = True

    @staticmethod
    def salt_cavern() -> "StorageAsset":
        """Return representative salt cavern parameters."""
        return StorageAsset(
            name="Salt_Cavern",
            capacity_bcf=10.0,
            min_inventory_bcf=0.5,
            max_injection_bcf_day=0.200,
            max_withdrawal_bcf_day=0.400,
            injection_cost=0.03,
            withdrawal_cost=0.02,
            fuel_loss_pct=0.02,
        )

    @staticmethod
    def depleted_reservoir() -> "StorageAsset":
        """Return representative depleted-reservoir parameters."""
        return StorageAsset(
            name="Depleted_Reservoir",
            capacity_bcf=50.0,
            min_inventory_bcf=5.0,
            max_injection_bcf_day=0.300,
            max_withdrawal_bcf_day=0.500,
            injection_cost=0.05,
            withdrawal_cost=0.04,
            fuel_loss_pct=0.03,
            ratchets=True,
        )

    def max_injection_monthly(self, days: int = 30) -> float:
        """Maximum injection in a month (Bcf)."""
        return self.max_injection_bcf_day * days

    def max_withdrawal_monthly(self, days: int = 30) -> float:
        """Maximum withdrawal in a month (Bcf)."""
        return self.max_withdrawal_bcf_day * days

    def ratchet_factor(self, fill_fraction: float, mode: str = "inject") -> float:
        """Rate adjustment based on fill level (ratchet constraint).

        Args:
            fill_fraction: Current inventory / capacity.
            mode: ``"inject"`` or ``"withdraw"``.

        Returns:
            Multiplicative factor (0-1) on max rate.
        """
        if not self.ratchets:
            return 1.0
        if mode == "inject":
            # Injection slows as storage fills up
            if fill_fraction > 0.9:
                return 0.5
            if fill_fraction > 0.7:
                return 0.8
            return 1.0
        else:
            # Withdrawal slows as storage empties
            if fill_fraction < 0.15:
                return 0.5
            if fill_fraction < 0.30:
                return 0.8
            return 1.0


# ---------------------------------------------------------------------------
# Intrinsic valuation (LP)
# ---------------------------------------------------------------------------

@dataclass
class IntrinsicResult:
    """Result of intrinsic storage valuation.

    Attributes:
        total_value: Net present value of optimal schedule ($M).
        schedule: DataFrame with monthly injection/withdrawal/inventory.
        spread_captured: Total calendar spread captured ($/MMBtu).
        cycles: Number of full injection-withdrawal cycles.
    """

    total_value: float = 0.0
    schedule: pd.DataFrame = field(default_factory=pd.DataFrame)
    spread_captured: float = 0.0
    cycles: float = 0.0


class IntrinsicValuation:
    """LP-based intrinsic storage valuation.

    Solves for the optimal injection/withdrawal schedule that maximises
    profit against a given forward curve, subject to physical constraints.

    The LP is formulated with decision variables:
        injection_t, withdrawal_t  for t = 0, ..., T-1

    Objective:
        max Σ_t [F_t × w_t × (1 - fuel_loss) - F_t × inj_t
                  - inj_cost × inj_t - wdl_cost × w_t]

    Subject to:
        - inventory_t = inv_{t-1} + inj_t - w_t
        - min_inv ≤ inventory_t ≤ capacity
        - 0 ≤ inj_t ≤ max_injection(days_t)
        - 0 ≤ w_t ≤ max_withdrawal(days_t)
    """

    def __init__(self, asset: StorageAsset) -> None:
        self.asset = asset

    def value(
        self,
        forward_prices: np.ndarray,
        initial_inventory: float = 0.5,
        days_per_month: np.ndarray | None = None,
    ) -> IntrinsicResult:
        """Compute optimal intrinsic storage value.

        Args:
            forward_prices: Monthly forward prices ($/MMBtu), length T.
            initial_inventory: Starting inventory (Bcf).
            days_per_month: Days in each month.  If ``None``, uses 30.

        Returns:
            ``IntrinsicResult`` with total value and optimal schedule.
        """
        T = len(forward_prices)
        if days_per_month is None:
            days_per_month = np.full(T, 30)
        a = self.asset

        # Decision variables: [inj_0, ..., inj_{T-1}, wdl_0, ..., wdl_{T-1}]
        # We MINIMISE c^T x, so negate the objective for maximisation.
        c = np.zeros(2 * T)
        for t in range(T):
            F = forward_prices[t]
            # Cost of injection: buy gas at F_t + variable cost
            c[t] = F + a.injection_cost
            # Revenue from withdrawal: sell gas at F_t * (1-loss) - variable cost
            c[T + t] = -(F * (1 - a.fuel_loss_pct) - a.withdrawal_cost)

        # Inequality constraints: A_ub @ x <= b_ub
        # For each month t, two constraints on inventory:
        #   inventory_t <= capacity   →   sum(inj[0:t+1]) - sum(wdl[0:t+1]) <= capacity - init_inv
        #   inventory_t >= min_inv    →   -sum(inj[0:t+1]) + sum(wdl[0:t+1]) <= init_inv - min_inv
        A_ub = np.zeros((2 * T, 2 * T))
        b_ub = np.zeros(2 * T)

        for t in range(T):
            # Upper bound on inventory: inj[0..t] - wdl[0..t] + init_inv <= capacity
            for s in range(t + 1):
                A_ub[t, s] = 1.0        # inj coefficient
                A_ub[t, T + s] = -1.0   # wdl coefficient
            b_ub[t] = a.capacity_bcf - initial_inventory

            # Lower bound on inventory: -(inj[0..t] - wdl[0..t]) <= init_inv - min_inv
            for s in range(t + 1):
                A_ub[T + t, s] = -1.0
                A_ub[T + t, T + s] = 1.0
            b_ub[T + t] = initial_inventory - a.min_inventory_bcf

        # Bounds on decision variables
        bounds = []
        for t in range(T):
            max_inj = a.max_injection_monthly(int(days_per_month[t]))
            bounds.append((0, max_inj))
        for t in range(T):
            max_wdl = a.max_withdrawal_monthly(int(days_per_month[t]))
            bounds.append((0, max_wdl))

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not result.success:
            logger.warning("LP did not converge: %s", result.message)
            return IntrinsicResult()

        inj = result.x[:T]
        wdl = result.x[T:]
        inventory = np.zeros(T)
        inventory[0] = initial_inventory + inj[0] - wdl[0]
        for t in range(1, T):
            inventory[t] = inventory[t - 1] + inj[t] - wdl[t]

        total_value = -result.fun  # We negated for minimisation

        schedule = pd.DataFrame({
            "month": np.arange(1, T + 1),
            "forward_price": forward_prices,
            "injection_bcf": inj,
            "withdrawal_bcf": wdl,
            "net_flow_bcf": inj - wdl,
            "inventory_bcf": inventory,
            "fill_pct": inventory / a.capacity_bcf * 100,
        })

        total_injected = inj.sum()
        total_withdrawn = wdl.sum()
        cycles = min(total_injected, total_withdrawn) / a.capacity_bcf

        logger.info(
            "Intrinsic value: $%.2fM (%.1f cycles, inj=%.1f Bcf, wdl=%.1f Bcf)",
            total_value, cycles, total_injected, total_withdrawn,
        )

        return IntrinsicResult(
            total_value=total_value,
            schedule=schedule,
            spread_captured=(total_value / max(total_withdrawn, 1e-9)),
            cycles=cycles,
        )


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck price process calibration & simulation
# ---------------------------------------------------------------------------

@dataclass
class OUParams:
    """Calibrated Ornstein-Uhlenbeck parameters.

    dS = κ(θ - S)dt + σ dW

    Attributes:
        kappa: Mean-reversion speed (annualised).
        theta: Long-run equilibrium price ($/MMBtu).
        sigma: Volatility (annualised).
    """

    kappa: float = 1.5
    theta: float = 3.0
    sigma: float = 0.80


class PriceSimulator:
    """Simulate natural gas price paths using an Ornstein-Uhlenbeck process.

    The OU process is standard for mean-reverting commodity prices:
        dS = κ(θ - S)dt + σ dW

    This captures the key stylized fact that natural gas prices exhibit
    strong mean reversion (half-life of ~3-6 months) while allowing for
    significant short-term volatility spikes.
    """

    def __init__(self, params: OUParams | None = None) -> None:
        self.params = params or OUParams()

    @staticmethod
    def calibrate(prices: pd.Series) -> OUParams:
        """Calibrate OU parameters from historical daily prices via OLS.

        Uses discrete-time approximation:
            S_{t+1} - S_t = κ(θ - S_t)Δt + σ√Δt ε_t

        Which is equivalent to:
            ΔS = a + b × S_t + ε

        where κ = -b/Δt, θ = -a/b, σ = std(ε)/√Δt

        Args:
            prices: Daily price series with DatetimeIndex.

        Returns:
            Calibrated ``OUParams``.
        """
        p = prices.dropna().values.astype(float)
        dt = 1.0 / 252  # daily
        delta_s = np.diff(p)
        s_lag = p[:-1]

        # OLS: ΔS = a + b * S_lag
        X = np.column_stack([np.ones(len(s_lag)), s_lag])
        beta = np.linalg.lstsq(X, delta_s, rcond=None)[0]
        a, b = beta[0], beta[1]

        residuals = delta_s - X @ beta
        sigma_eps = np.std(residuals)

        kappa = max(-b / dt, 0.01)  # ensure positive mean reversion
        theta = -a / b if abs(b) > 1e-10 else np.mean(p)
        sigma = sigma_eps / np.sqrt(dt)

        half_life = np.log(2) / kappa * 252  # in trading days

        logger.info(
            "OU calibration: κ=%.3f, θ=$%.2f, σ=%.3f, half-life=%.0f days",
            kappa, theta, sigma, half_life,
        )
        return OUParams(kappa=kappa, theta=theta, sigma=sigma)

    def simulate(
        self,
        s0: float,
        n_months: int = 12,
        n_paths: int = 5000,
        steps_per_month: int = 21,
        seed: int | None = 42,
    ) -> np.ndarray:
        """Simulate OU price paths.

        Args:
            s0: Starting price.
            n_months: Number of months to simulate.
            n_paths: Number of Monte Carlo paths.
            steps_per_month: Trading days per month.
            seed: Random seed for reproducibility.

        Returns:
            Array of shape ``(n_paths, n_months)`` with monthly average
            prices for each path.
        """
        rng = np.random.default_rng(seed)
        p = self.params
        total_steps = n_months * steps_per_month
        dt = 1.0 / 252

        paths = np.zeros((n_paths, total_steps + 1))
        paths[:, 0] = s0

        for t in range(total_steps):
            dw = rng.normal(0, np.sqrt(dt), n_paths)
            paths[:, t + 1] = (
                paths[:, t]
                + p.kappa * (p.theta - paths[:, t]) * dt
                + p.sigma * dw
            )
            # Floor at zero (gas prices can't go negative... usually)
            paths[:, t + 1] = np.maximum(paths[:, t + 1], 0.01)

        # Compute monthly averages
        monthly = np.zeros((n_paths, n_months))
        for m in range(n_months):
            start = m * steps_per_month + 1
            end = (m + 1) * steps_per_month + 1
            monthly[:, m] = paths[:, start:end].mean(axis=1)

        return monthly


# ---------------------------------------------------------------------------
# LSMC extrinsic valuation
# ---------------------------------------------------------------------------

@dataclass
class ExtrinsicResult:
    """Result of extrinsic (LSMC) storage valuation.

    Attributes:
        total_option_value: Full option value of storage ($M).
        intrinsic_value: Intrinsic component ($M).
        extrinsic_value: Optionality premium above intrinsic ($M).
        extrinsic_pct: Extrinsic as percentage of total.
        std_error: Standard error of the MC estimate ($M).
        n_paths: Number of simulation paths used.
    """

    total_option_value: float = 0.0
    intrinsic_value: float = 0.0
    extrinsic_value: float = 0.0
    extrinsic_pct: float = 0.0
    std_error: float = 0.0
    n_paths: int = 0


class LSMCValuation:
    """Least-Squares Monte Carlo valuation of gas storage.

    Implements Longstaff-Schwartz backward induction where at each monthly
    decision point the holder chooses the optimal injection/withdrawal
    action.  Continuation value is estimated by regressing discounted
    future payoffs on polynomial basis functions of (price, inventory).

    Key references:
        - Longstaff & Schwartz (2001), "Valuing American Options by Simulation"
        - Boogert & de Jong (2008), "Gas Storage Valuation Using a Monte Carlo Method"
    """

    def __init__(
        self,
        asset: StorageAsset,
        risk_free_rate: float = 0.05,
        polynomial_degree: int = 3,
    ) -> None:
        self.asset = asset
        self.rf = risk_free_rate
        self.degree = polynomial_degree

    def _basis_functions(self, price: np.ndarray, inventory: np.ndarray) -> np.ndarray:
        """Construct polynomial basis functions for regression.

        Uses Laguerre-style polynomials as in Longstaff-Schwartz:
        1, P, P², P³, I, I², PI, P²I

        Args:
            price: Price array (n_paths,).
            inventory: Inventory array (n_paths,).

        Returns:
            Basis matrix of shape (n_paths, n_features).
        """
        p_norm = price / price.mean() if price.mean() > 0 else price
        i_norm = inventory / self.asset.capacity_bcf
        features = [
            np.ones_like(price),
            p_norm,
            p_norm ** 2,
            p_norm ** 3,
            i_norm,
            i_norm ** 2,
            p_norm * i_norm,
            p_norm ** 2 * i_norm,
        ]
        return np.column_stack(features)

    def value(
        self,
        forward_curve: ForwardCurve,
        simulator: PriceSimulator,
        initial_inventory: float = 0.5,
        n_months: int = 12,
        n_paths: int = 5000,
        start_month: int = 4,
        seed: int = 42,
    ) -> ExtrinsicResult:
        """Compute storage option value via LSMC.

        Args:
            forward_curve: Calibrated forward curve for seasonal adjustments.
            simulator: OU price simulator with calibrated parameters.
            initial_inventory: Starting inventory (Bcf).
            n_months: Storage contract tenor (months).
            n_paths: Number of MC simulation paths.
            start_month: Starting month (1=Jan).
            seed: Random seed.

        Returns:
            ``ExtrinsicResult`` with total, intrinsic, and extrinsic values.
        """
        a = self.asset
        s0 = forward_curve.base_price

        # Simulate monthly price paths
        price_paths = simulator.simulate(s0, n_months, n_paths, seed=seed)

        # Apply seasonal adjustment (multiply by seasonal factor / prompt factor)
        prompt_factor = forward_curve.seasonal_factors[(start_month - 1) % 12]
        for m in range(n_months):
            month_idx = (start_month - 1 + m) % 12
            seasonal_adj = forward_curve.seasonal_factors[month_idx] / prompt_factor
            price_paths[:, m] *= seasonal_adj

        dt_month = 1.0 / 12
        discount = np.exp(-self.rf * dt_month)

        # Backward induction
        # State: (price, inventory) at each month
        # Action: inject x Bcf, withdraw y Bcf (net flow)
        # Value function approximated via polynomial regression

        # Discretise actions: inject full, inject half, do nothing, withdraw half, withdraw full
        inventory = np.full(n_paths, initial_inventory)
        cashflows = np.zeros((n_paths, n_months))
        inventory_paths = np.zeros((n_paths, n_months))
        inventory_paths[:, 0] = initial_inventory

        # Forward pass: use greedy heuristic as starting policy
        for t in range(n_months):
            prices = price_paths[:, t]
            days = 30

            max_inj = a.max_injection_monthly(days) * np.array([
                a.ratchet_factor(inv / a.capacity_bcf, "inject")
                for inv in inventory
            ])
            max_wdl = a.max_withdrawal_monthly(days) * np.array([
                a.ratchet_factor(inv / a.capacity_bcf, "withdraw")
                for inv in inventory
            ])

            # Greedy: if price < mean → inject, if price > mean → withdraw
            mean_price = forward_curve.base_price
            net_flow = np.zeros(n_paths)

            for i in range(n_paths):
                if prices[i] < mean_price * 0.95:  # cheap → inject
                    amount = min(max_inj[i], a.capacity_bcf - inventory[i])
                    net_flow[i] = amount
                    cashflows[i, t] = -(prices[i] + a.injection_cost) * amount
                elif prices[i] > mean_price * 1.05:  # expensive → withdraw
                    amount = min(max_wdl[i], inventory[i] - a.min_inventory_bcf)
                    net_flow[i] = -amount
                    revenue = prices[i] * (1 - a.fuel_loss_pct) - a.withdrawal_cost
                    cashflows[i, t] = revenue * amount

                inventory[i] += net_flow[i]
                inventory[i] = np.clip(inventory[i], a.min_inventory_bcf, a.capacity_bcf)

            if t < n_months - 1:
                inventory_paths[:, t + 1] = inventory

        # Backward pass: LSMC improvement
        # Start from T-1 and work backward, optimising action at each step
        cumulative_value = np.zeros(n_paths)
        for t in range(n_months - 1, -1, -1):
            prices = price_paths[:, t]
            inv_t = inventory_paths[:, t] if t < n_months else inventory

            # Continuation value from regression
            if t < n_months - 1:
                in_the_money = cumulative_value > 0
                if in_the_money.sum() > 10:
                    basis = self._basis_functions(prices[in_the_money], inv_t[in_the_money])
                    try:
                        # Ridge regression for numerical stability
                        reg_target = cumulative_value[in_the_money] * discount
                        betas = np.linalg.lstsq(
                            basis.T @ basis + 0.01 * np.eye(basis.shape[1]),
                            basis.T @ reg_target,
                            rcond=None,
                        )[0]
                        all_basis = self._basis_functions(prices, inv_t)
                        continuation = all_basis @ betas
                    except np.linalg.LinAlgError:
                        continuation = np.full(n_paths, cumulative_value.mean() * discount)
                else:
                    continuation = np.full(n_paths, cumulative_value.mean() * discount)
            else:
                continuation = np.zeros(n_paths)

            # Optimise action at this time step
            days = 30
            for i in range(n_paths):
                fill = inv_t[i] / a.capacity_bcf
                max_inj = a.max_injection_monthly(days) * a.ratchet_factor(fill, "inject")
                max_wdl = a.max_withdrawal_monthly(days) * a.ratchet_factor(fill, "withdraw")

                # Candidate actions: full inject, half inject, nothing, half withdraw, full withdraw
                candidates = []

                # Inject full
                amt = min(max_inj, a.capacity_bcf - inv_t[i])
                if amt > 0:
                    cf = -(prices[i] + a.injection_cost) * amt
                    candidates.append((cf, amt))

                # Inject half
                amt_h = amt * 0.5
                if amt_h > 0:
                    cf_h = -(prices[i] + a.injection_cost) * amt_h
                    candidates.append((cf_h, amt_h))

                # Do nothing
                candidates.append((0.0, 0.0))

                # Withdraw half
                amt_w = min(max_wdl, inv_t[i] - a.min_inventory_bcf) * 0.5
                if amt_w > 0:
                    rev = (prices[i] * (1 - a.fuel_loss_pct) - a.withdrawal_cost) * amt_w
                    candidates.append((rev, -amt_w))

                # Withdraw full
                amt_w_full = min(max_wdl, inv_t[i] - a.min_inventory_bcf)
                if amt_w_full > 0:
                    rev_f = (prices[i] * (1 - a.fuel_loss_pct) - a.withdrawal_cost) * amt_w_full
                    candidates.append((rev_f, -amt_w_full))

                # Pick best action (immediate cf + continuation)
                best_cf, best_flow = 0.0, 0.0
                best_total = continuation[i]
                for cf, flow in candidates:
                    total = cf + continuation[i]
                    if total > best_total:
                        best_total = total
                        best_cf = cf
                        best_flow = flow

                cashflows[i, t] = best_cf

            cumulative_value = cashflows[:, t] + cumulative_value * discount

        # Compute total value: mean of discounted cashflows across paths
        path_values = np.zeros(n_paths)
        for i in range(n_paths):
            pv = 0.0
            for t in range(n_months):
                pv += cashflows[i, t] * np.exp(-self.rf * t * dt_month)
            path_values[i] = pv

        total_option = float(path_values.mean())
        std_error = float(path_values.std() / np.sqrt(n_paths))

        # Compute intrinsic for comparison
        intrinsic_calc = IntrinsicValuation(a)
        fwd = forward_curve.curve(start_month, n_months)
        intrinsic_result = intrinsic_calc.value(fwd, initial_inventory)
        intrinsic_val = intrinsic_result.total_value

        extrinsic = total_option - intrinsic_val
        extrinsic_pct = (extrinsic / total_option * 100) if total_option > 0 else 0.0

        logger.info(
            "LSMC: total=$%.2fM, intrinsic=$%.2fM, extrinsic=$%.2fM (%.1f%%), SE=$%.3fM",
            total_option, intrinsic_val, extrinsic, extrinsic_pct, std_error,
        )

        return ExtrinsicResult(
            total_option_value=total_option,
            intrinsic_value=intrinsic_val,
            extrinsic_value=extrinsic,
            extrinsic_pct=extrinsic_pct,
            std_error=std_error,
            n_paths=n_paths,
        )


# ---------------------------------------------------------------------------
# Storage Greeks
# ---------------------------------------------------------------------------

@dataclass
class StorageGreeks:
    """Sensitivity measures for the storage position.

    Attributes:
        delta: dV/dS — change in value per $1 change in spot.
        gamma: d²V/dS² — convexity of value to price.
        theta: dV/dt — daily time decay.
        vega: dV/dσ — sensitivity to volatility (per 1% vol change).
    """

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0


class StorageValuationEngine:
    """Master engine combining all storage valuation components.

    Coordinates forward curve construction, intrinsic LP, LSMC, and
    Greeks computation.
    """

    def __init__(
        self,
        asset: StorageAsset | None = None,
        risk_free_rate: float = 0.05,
    ) -> None:
        """Initialise the storage valuation engine.

        Args:
            asset: Storage facility parameters.
            risk_free_rate: Annualised risk-free rate.
        """
        self.asset = asset or StorageAsset.salt_cavern()
        self.rf = risk_free_rate
        self.forward_curve: ForwardCurve | None = None
        self.simulator: PriceSimulator | None = None
        self.ou_params: OUParams | None = None

    def calibrate(self, price_history: pd.Series) -> dict[str, float]:
        """Calibrate forward curve and price dynamics from history.

        Args:
            price_history: Daily NG=F close prices.

        Returns:
            Dictionary with calibrated parameters.
        """
        self.forward_curve = ForwardCurve(price_history)
        self.ou_params = PriceSimulator.calibrate(price_history)
        self.simulator = PriceSimulator(self.ou_params)

        half_life = np.log(2) / self.ou_params.kappa * 252

        return {
            "kappa": self.ou_params.kappa,
            "theta": self.ou_params.theta,
            "sigma": self.ou_params.sigma,
            "half_life_days": half_life,
            "base_price": self.forward_curve.base_price,
            "winter_premium_pct": float(
                (self.forward_curve.seasonal_factors[[0, 1, 11]].mean() - 1) * 100
            ),
            "summer_discount_pct": float(
                (1 - self.forward_curve.seasonal_factors[4:9].mean()) * 100
            ),
        }

    def intrinsic(
        self,
        start_month: int = 4,
        n_months: int = 12,
        initial_inventory: float | None = None,
    ) -> IntrinsicResult:
        """Compute intrinsic storage value.

        Args:
            start_month: Contract start month (1=Jan, 4=Apr typical).
            n_months: Contract tenor in months.
            initial_inventory: Starting inventory (Bcf).

        Returns:
            ``IntrinsicResult`` with optimal schedule and value.
        """
        if self.forward_curve is None:
            raise RuntimeError("Call calibrate() first")
        inv = initial_inventory if initial_inventory is not None else self.asset.min_inventory_bcf
        fwd = self.forward_curve.curve(start_month, n_months)
        calc = IntrinsicValuation(self.asset)
        return calc.value(fwd, inv)

    def extrinsic(
        self,
        start_month: int = 4,
        n_months: int = 12,
        initial_inventory: float = 0.5,
        n_paths: int = 5000,
        seed: int = 42,
    ) -> ExtrinsicResult:
        """Compute extrinsic (option) storage value via LSMC.

        Args:
            start_month: Contract start month.
            n_months: Contract tenor.
            initial_inventory: Starting inventory (Bcf).
            n_paths: Monte Carlo paths.
            seed: Random seed.

        Returns:
            ``ExtrinsicResult`` with total, intrinsic, and extrinsic.
        """
        if self.forward_curve is None or self.simulator is None:
            raise RuntimeError("Call calibrate() first")
        lsmc = LSMCValuation(self.asset, risk_free_rate=self.rf)
        return lsmc.value(
            self.forward_curve, self.simulator,
            initial_inventory, n_months, n_paths, start_month, seed,
        )

    def greeks(
        self,
        start_month: int = 4,
        n_months: int = 12,
        initial_inventory: float = 0.5,
        bump_size: float = 0.10,
        n_paths: int = 3000,
        seed: int = 42,
    ) -> StorageGreeks:
        """Compute storage Greeks via finite differences.

        Args:
            start_month: Contract start month.
            n_months: Contract tenor.
            initial_inventory: Starting inventory.
            bump_size: Price bump in $/MMBtu for delta/gamma.
            n_paths: MC paths per valuation.
            seed: Random seed.

        Returns:
            ``StorageGreeks`` with delta, gamma, theta, vega.
        """
        if self.forward_curve is None or self.simulator is None:
            raise RuntimeError("Call calibrate() first")

        lsmc = LSMCValuation(self.asset, risk_free_rate=self.rf)
        base_price = self.forward_curve.base_price

        def _val(price_shift: float = 0.0, vol_shift: float = 0.0, month_shift: int = 0) -> float:
            fc = ForwardCurve.__new__(ForwardCurve)
            fc.seasonal_factors = self.forward_curve.seasonal_factors.copy()
            fc.base_price = base_price + price_shift

            sim = PriceSimulator(OUParams(
                kappa=self.ou_params.kappa,
                theta=self.ou_params.theta,
                sigma=self.ou_params.sigma + vol_shift,
            ))

            eff_months = max(n_months + month_shift, 1)
            res = lsmc.value(
                fc, sim, initial_inventory, eff_months, n_paths,
                start_month, seed,
            )
            return res.total_option_value

        v_base = _val()
        v_up = _val(price_shift=bump_size)
        v_down = _val(price_shift=-bump_size)
        v_vol_up = _val(vol_shift=0.05)
        v_month_less = _val(month_shift=-1)

        delta = (v_up - v_down) / (2 * bump_size)
        gamma = (v_up - 2 * v_base + v_down) / (bump_size ** 2)
        theta = (v_month_less - v_base)  # value change losing 1 month
        vega = (v_vol_up - v_base) / 0.05  # per 1% vol move

        logger.info(
            "Greeks: Δ=%.3f, Γ=%.3f, Θ=%.3f, ν=%.3f",
            delta, gamma, theta, vega,
        )
        return StorageGreeks(delta=delta, gamma=gamma, theta=theta, vega=vega)

    def rolling_intrinsic(
        self,
        price_history: pd.Series,
        start_month: int = 4,
        n_months: int = 12,
        rebalance_freq: int = 5,
    ) -> pd.DataFrame:
        """Compute rolling intrinsic value over historical dates.

        Re-optimises the LP at each rebalance date using the then-current
        spot price and remaining tenor, simulating how a storage operator
        would dynamically manage the book.

        Args:
            price_history: Daily NG close prices.
            start_month: Initial contract start month.
            n_months: Contract tenor.
            rebalance_freq: Days between re-optimisations.

        Returns:
            DataFrame with date, spot price, intrinsic value, optimal action.
        """
        if self.forward_curve is None:
            raise RuntimeError("Call calibrate() first")

        prices = price_history.dropna()
        # Use last n_months * 21 trading days
        window = min(n_months * 21, len(prices) - 1)
        prices = prices.iloc[-window:]

        calc = IntrinsicValuation(self.asset)
        records = []
        inventory = self.asset.min_inventory_bcf
        months_remaining = n_months
        net = 0.0

        for idx in range(0, len(prices), rebalance_freq):
            if months_remaining <= 1:
                break
            date = prices.index[idx]
            spot = float(prices.iloc[idx])

            # Current month based on actual date
            current_month = date.month
            mr_int = max(int(months_remaining), 1)
            fwd = self.forward_curve.curve_at_price(spot, current_month, mr_int)

            result = calc.value(fwd, inventory)
            if result.schedule is not None and len(result.schedule) > 0:
                first_action = result.schedule.iloc[0]
                net = float(first_action.get("net_flow_bcf", 0))
                inventory = np.clip(
                    inventory + net,
                    self.asset.min_inventory_bcf,
                    self.asset.capacity_bcf,
                )

            records.append({
                "date": date,
                "spot_price": spot,
                "months_remaining": months_remaining,
                "intrinsic_value": result.total_value,
                "inventory_bcf": inventory,
                "optimal_action": (
                    "INJECT" if net > 0.1 else "WITHDRAW" if net < -0.1 else "HOLD"
                ),
            })

            months_remaining -= rebalance_freq / 21  # approximate month decrement

        return pd.DataFrame(records)
