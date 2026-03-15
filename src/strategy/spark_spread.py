"""Spark-Spread & Heat-Rate Analysis for Gas-Fired Generation.

Models the profitability of gas-fired power plants by analysing the
relationship between wholesale electricity prices and natural gas costs:

    Spark Spread = Power Price ($/MWh) - Heat Rate (MMBtu/MWh) × Gas Price ($/MMBtu)

A positive spark spread indicates profitable generation; a negative spread
means the plant should be offline.  This module:

* Estimates implied heat rates from observed generation dispatch data
* Constructs regional power supply (merit-order) curves
* Models spark-spread dynamics and identifies profitable dispatch signals
* Computes clean spark spreads net of variable O&M and emissions costs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Typical heat rates by plant class (MMBtu/MWh)
HEAT_RATES = {
    "combined_cycle": 6.6,    # Modern CCGT
    "combustion_turbine": 10.0,  # Peaker CT
    "steam_turbine": 9.5,     # Older steam
    "average_gas": 7.5,       # Fleet average
}

# Variable O&M costs by technology ($/MWh)
VOM_COSTS = {
    "combined_cycle": 2.50,
    "combustion_turbine": 5.00,
    "steam_turbine": 4.00,
    "nuclear": 2.50,
    "coal": 3.50,
    "wind": 0.00,
    "solar": 0.00,
}

# CO2 emission rates (tons CO2 per MWh) - for clean spark spread
EMISSION_RATES = {
    "combined_cycle": 0.40,
    "combustion_turbine": 0.60,
    "coal": 0.95,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SparkSpreadResult:
    """Regional spark-spread analysis results.

    Attributes:
        region: ISO/RTO region name.
        implied_heat_rate: Estimated fleet-average heat rate.
        avg_spark_spread: Average spark spread over the period.
        clean_spark_spread: Spark spread net of VOM and carbon costs.
        dispatch_threshold: Gas price at which gas plants go offline.
        profitable_hours_pct: Percentage of hours with positive spread.
        seasonal_pattern: Monthly average spark spreads.
    """

    region: str = ""
    implied_heat_rate: float = 0.0
    avg_spark_spread: float = 0.0
    clean_spark_spread: float = 0.0
    dispatch_threshold: float = 0.0
    profitable_hours_pct: float = 0.0
    seasonal_pattern: dict[int, float] = field(default_factory=dict)


@dataclass
class MeritOrderResult:
    """Supply-curve (merit-order) estimation result.

    Attributes:
        region: Region name.
        marginal_fuel: Fuel on the margin most often.
        marginal_cost_estimate: Estimated marginal generation cost ($/MWh).
        base_load_capacity_mw: Estimated must-run baseload (nuclear + min renewables).
        gas_share_at_peak: Gas generation share during peak load.
        stack: DataFrame with the estimated supply stack.
    """

    region: str = ""
    marginal_fuel: str = ""
    marginal_cost_estimate: float = 0.0
    base_load_capacity_mw: float = 0.0
    gas_share_at_peak: float = 0.0
    stack: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Spark Spread Engine
# ---------------------------------------------------------------------------

class SparkSpreadModel:
    """Model spark spread and heat-rate dynamics from generation data.

    Uses observed generation mix data (EIA-930) along with natural gas
    prices to infer regional heat rates, estimate power prices, and
    model spark-spread dynamics.

    The key insight: when we observe how much gas generation dispatches
    at different gas price levels, we can infer the fleet's marginal
    heat rate and the implied power price.
    """

    def __init__(
        self,
        carbon_price: float = 0.0,
        assumed_power_price_premium: float = 5.0,
    ) -> None:
        """Initialise the spark spread model.

        Args:
            carbon_price: CO2 price in $/ton (for clean spark spread).
            assumed_power_price_premium: Markup above marginal cost
                that sets the approximate power price ($/MWh).
        """
        self.carbon_price = carbon_price
        self.power_premium = assumed_power_price_premium
        self._heat_rate_models: dict[str, LinearRegression] = {}

    @staticmethod
    def load_and_merge(
        generation_path: str,
        ng_price_path: str,
    ) -> pd.DataFrame:
        """Load and merge generation data with natural gas prices.

        Args:
            generation_path: Path to daily generation CSV.
            ng_price_path: Path to NG=F price CSV.

        Returns:
            Merged DataFrame indexed by date.
        """
        gen = pd.read_csv(generation_path, parse_dates=["date"]).set_index("date")
        price = pd.read_csv(ng_price_path, parse_dates=["Date"]).set_index("Date")
        price = price[["Close"]].rename(columns={"Close": "ng_price"})
        price["ng_price"] = pd.to_numeric(price["ng_price"], errors="coerce")

        merged = gen.join(price, how="inner").dropna(subset=["ng_price"])
        logger.info("Merged gen+price: %d rows, %d cols", len(merged), len(merged.columns))
        return merged

    def estimate_implied_heat_rate(
        self,
        data: pd.DataFrame,
        region: str = "PJM",
    ) -> float:
        """Estimate the implied fleet-average heat rate for a region.

        Uses the relationship: gas_generation ∝ f(gas_price, load)
        When gas prices rise, gas generation falls (substituted by coal).
        The price sensitivity reveals the fleet heat-rate distribution.

        Args:
            data: Merged generation + price data.
            region: Region prefix (used to find columns like ``PJM_gas_mwh``).

        Returns:
            Implied marginal heat rate (MMBtu/MWh).
        """
        gas_col = f"{region}_gas_mwh"
        total_col = f"{region}_total_mwh"

        if gas_col not in data.columns:
            logger.warning("Column %s not found, trying alternatives", gas_col)
            # Try to find a matching column
            gas_candidates = [c for c in data.columns if region.lower() in c.lower() and "gas" in c.lower()]
            if gas_candidates:
                gas_col = gas_candidates[0]
            else:
                return HEAT_RATES["average_gas"]

        total_candidates = [c for c in data.columns if region.lower() in c.lower() and "total" in c.lower()]
        if total_candidates:
            total_col = total_candidates[0]
        else:
            total_col = None

        df = data[[gas_col, "ng_price"]].dropna().copy()
        if total_col and total_col in data.columns:
            df["total"] = data[total_col]
        else:
            df["total"] = df[gas_col] * 3  # rough estimate

        df["gas_share"] = df[gas_col] / df["total"].replace(0, np.nan)
        df = df.dropna()

        if len(df) < 30:
            logger.warning("Insufficient data for %s heat rate estimation", region)
            return HEAT_RATES["average_gas"]

        # Regression: gas_share = a + b * ng_price + c * total
        # The heat rate relates to the price sensitivity of gas dispatch
        X = df[["ng_price"]].values
        y = df["gas_share"].values

        model = LinearRegression()
        model.fit(X, y)
        self._heat_rate_models[region] = model

        # Implied heat rate: at what gas price does gas generation share drop to 0?
        # gas_share = 0 → ng_price_threshold = -intercept / slope
        if abs(model.coef_[0]) > 1e-10:
            threshold_price = -model.intercept_ / model.coef_[0]
        else:
            threshold_price = 20.0

        # Implied fleet heat rate based on gas share level.
        # Higher gas share → more CTs running as marginal units → higher HR.
        # Lower gas share → gas is mostly efficient CCGTs → lower HR.
        mean_gas_share = df["gas_share"].mean()
        ccgt_hr = HEAT_RATES["combined_cycle"]   # 6.6
        ct_hr = HEAT_RATES["combustion_turbine"]  # 10.0
        # Interpolate: at 20% gas share assume mostly CCGT, at 60%+ assume mix
        share_frac = np.clip((mean_gas_share - 0.15) / 0.50, 0, 1)
        implied_hr = ccgt_hr + share_frac * (ct_hr - ccgt_hr) * 0.5

        # Also factor in regression sensitivity: steeper negative slope
        # means gas dispatches more at the margin under price pressure
        sensitivity_adj = 0.0
        if model.coef_[0] < -0.005:
            sensitivity_adj = min(abs(model.coef_[0]) * 10, 1.0)
        implied_hr += sensitivity_adj

        implied_hr = float(np.clip(implied_hr, 6.4, 10.5))

        logger.info(
            "%s: implied HR=%.1f MMBtu/MWh, gas_price_sensitivity=%.4f, threshold=$%.1f",
            region, implied_hr, model.coef_[0], threshold_price,
        )
        return float(implied_hr)

    def compute_spark_spreads(
        self,
        data: pd.DataFrame,
        region: str = "PJM",
        heat_rate: float | None = None,
    ) -> pd.DataFrame:
        """Compute daily spark spreads for a region.

        Estimates regional power prices from gas share, demand, seasonal
        patterns, and NG price volatility.  Uses actual regional demand
        data when available for a demand-driven scarcity premium.

        Args:
            data: Merged generation + price + optional demand data.
            region: Region identifier (PJM, MISO, NYISO, ISONE).
            heat_rate: Override heat rate.  If ``None``, uses fleet average.

        Returns:
            DataFrame with daily spark-spread metrics.
        """
        hr = heat_rate or self.estimate_implied_heat_rate(data, region)

        gas_col = f"{region}_gas_mwh"
        gas_candidates = [c for c in data.columns if region.lower() in c.lower() and "gas" in c.lower()]
        if gas_candidates:
            gas_col = gas_candidates[0]

        total_candidates = [c for c in data.columns if region.lower() in c.lower() and "total" in c.lower()]
        if total_candidates:
            total_col = total_candidates[0]
        else:
            total_col = gas_col

        df = data[["ng_price"]].copy()
        if gas_col in data.columns:
            df["gas_gen_mwh"] = data[gas_col]
        if total_col in data.columns:
            df["total_gen_mwh"] = data[total_col]
        df = df.dropna(subset=["ng_price"])

        # ── Gas share from generation data ──────────────────────────
        if "gas_gen_mwh" in df.columns and "total_gen_mwh" in df.columns:
            df["gas_share"] = (
                df["gas_gen_mwh"] / df["total_gen_mwh"].replace(0, np.nan)
            )
        else:
            df["gas_share"] = 0.5

        gs_mean = df["gas_share"].mean()
        gs_std = max(df["gas_share"].std(), 0.01)
        df["gs_z"] = (df["gas_share"] - gs_mean) / gs_std

        df["_month"] = df.index.month
        seasonal_premium = {
            1: 1.8, 2: 1.5, 3: 0.6, 4: 0.3, 5: 0.5, 6: 1.2,
            7: 2.0, 8: 1.8, 9: 0.8, 10: 0.4, 11: 0.7, 12: 1.4,
        }
        df["seasonal_adj"] = df["_month"].map(seasonal_premium)

        df["ng_ret"] = df["ng_price"].pct_change()
        df["ng_vol_20d"] = df["ng_ret"].rolling(20, min_periods=5).std() * np.sqrt(252)
        vol_med = df["ng_vol_20d"].median()
        df["vol_premium"] = (df["ng_vol_20d"] / max(vol_med, 0.01) - 1).clip(-1, 3) * 2

        # ── Demand-driven scarcity premium (if demand data available) ─
        demand_col = f"{region}_demand_mean"
        if demand_col in data.columns:
            df["_demand"] = data[demand_col]
            dm = df["_demand"].median()
            ds = max(df["_demand"].std(), 1)
            df["demand_z"] = (df["_demand"] - dm) / ds
            df["demand_premium"] = df["demand_z"].clip(-2, 3) * 2.0
            df.drop(columns=["_demand"], inplace=True)
        else:
            df["demand_z"] = 0
            df["demand_premium"] = 0

        df["scarcity_factor"] = (
            df["gs_z"] * 3.0
            + df["seasonal_adj"]
            + df["vol_premium"]
            + df["demand_premium"]
        )

        df["power_price"] = (
            df["ng_price"] * hr
            + self.power_premium
            + df["scarcity_factor"]
        )

        ccgt_floor = df["ng_price"] * HEAT_RATES["combined_cycle"] + VOM_COSTS["combined_cycle"]
        df["power_price"] = df["power_price"].clip(lower=ccgt_floor)

        df.drop(columns=["gs_z", "_month", "seasonal_adj", "ng_ret",
                         "ng_vol_20d", "vol_premium", "demand_z",
                         "demand_premium", "scarcity_factor"],
                inplace=True, errors="ignore")

        has_demand = demand_col in data.columns
        logger.info("%s: estimated power prices (gas_share=%.1f%%, demand=%s)",
                    region, gs_mean * 100,
                    "available" if has_demand else "not available")

        # ── Gas cost and spark spreads ──────────────────────────────
        # The marginal HR (hr) was used to estimate power price.
        # The primary "spark spread" uses CCGT HR — this is the standard
        # market definition (can a combined-cycle make money?).
        ccgt_hr = HEAT_RATES["combined_cycle"]
        df["gas_cost_mwh"] = df["ng_price"] * ccgt_hr
        df["est_power_price"] = df["power_price"]

        # Spark spreads by plant type
        for plant_type, plant_hr in HEAT_RATES.items():
            col_name = f"spark_{plant_type}"
            vom = VOM_COSTS.get(plant_type, 3.0)
            carbon = EMISSION_RATES.get(plant_type, 0.5) * self.carbon_price
            df[col_name] = df["power_price"] - plant_hr * df["ng_price"] - vom - carbon

        # Primary spark spread = CCGT spark (standard market convention)
        df["spark_spread"] = df["spark_combined_cycle"]
        df["clean_spark_spread"] = (
            df["spark_spread"]
            - VOM_COSTS["combined_cycle"]
            - EMISSION_RATES["combined_cycle"] * self.carbon_price
        )
        df["spark_ratio"] = df["power_price"] / (df["ng_price"] * hr).replace(0, np.nan)

        logger.info(
            "%s spark spreads: mean=$%.1f/MWh, clean=$%.1f/MWh, profitable=%.0f%%",
            region,
            df["spark_spread"].mean(),
            df["clean_spark_spread"].mean(),
            (df["spark_spread"] > 0).mean() * 100,
        )
        return df

    def estimate_merit_order(
        self,
        data: pd.DataFrame,
        region: str = "PJM",
    ) -> MeritOrderResult:
        """Estimate the generation supply stack (merit order) for a region.

        Analyses the generation fuel mix at different demand levels to
        reconstruct the merit-order dispatch curve.

        Args:
            data: Merged generation + price data.
            region: Region identifier.

        Returns:
            ``MeritOrderResult`` with supply stack estimates.
        """
        prefix = region.lower()

        fuel_cols = {}
        for fuel in ["nuclear", "gas", "coal", "wind", "solar", "hydro"]:
            candidates = [c for c in data.columns if prefix in c.lower() and fuel in c.lower() and "mwh" in c.lower()]
            if candidates:
                fuel_cols[fuel] = candidates[0]

        if not fuel_cols:
            logger.warning("No fuel columns found for %s", region)
            return MeritOrderResult(region=region)

        df = data[list(fuel_cols.values()) + ["ng_price"]].dropna().copy()
        df.columns = list(fuel_cols.keys()) + ["ng_price"]

        total_gen = df[list(fuel_cols.keys())].sum(axis=1)

        # Baseload = nuclear + hydro (roughly constant)
        baseload = 0
        if "nuclear" in df.columns:
            baseload += df["nuclear"].median()
        if "hydro" in df.columns:
            baseload += df["hydro"].median()

        # Compute gas share at different load quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        stack_records = []
        for q in quantiles:
            mask = total_gen > total_gen.quantile(q)
            if mask.sum() > 5:
                for fuel in fuel_cols.keys():
                    share = df.loc[mask, fuel].mean() / total_gen[mask].mean()
                    stack_records.append({
                        "load_quantile": q,
                        "fuel": fuel,
                        "share_pct": share * 100,
                        "avg_gen_mwh": df.loc[mask, fuel].mean(),
                    })

        stack_df = pd.DataFrame(stack_records)

        # Determine marginal fuel
        if "gas" in df.columns and "coal" in df.columns:
            gas_peak_share = df["gas"].quantile(0.9) / total_gen.quantile(0.9)
            coal_peak_share = df["coal"].quantile(0.9) / total_gen.quantile(0.9) if "coal" in df.columns else 0
            marginal = "gas" if gas_peak_share > coal_peak_share else "coal"
        else:
            marginal = "gas"

        # Marginal cost estimate
        avg_gas = df["ng_price"].mean()
        if marginal == "gas":
            mc = avg_gas * HEAT_RATES["combined_cycle"] + VOM_COSTS["combined_cycle"]
        else:
            mc = 25.0  # approximate coal marginal cost

        gas_share_peak = (df["gas"].sum() / total_gen.sum()) if "gas" in df.columns else 0

        result = MeritOrderResult(
            region=region,
            marginal_fuel=marginal,
            marginal_cost_estimate=mc,
            base_load_capacity_mw=baseload,
            gas_share_at_peak=gas_share_peak,
            stack=stack_df,
        )

        logger.info(
            "%s merit order: marginal=%s, MC=$%.1f/MWh, baseload=%.0f MWh/d",
            region, marginal, mc, baseload,
        )
        return result

    def dispatch_model(
        self,
        data: pd.DataFrame,
        region: str = "PJM",
    ) -> pd.DataFrame:
        """Build a gas dispatch prediction model.

        Predicts gas-fired generation volume as a function of gas price,
        total load, temperature, and season.  This is a simplified version
        of the dispatch models used on real trading desks to forecast
        gas burn.

        Args:
            data: Merged generation + price data.
            region: Region identifier.

        Returns:
            DataFrame with actual vs predicted gas generation.
        """
        gas_candidates = [c for c in data.columns if region.lower() in c.lower() and "gas" in c.lower()]
        total_candidates = [c for c in data.columns if region.lower() in c.lower() and "total" in c.lower()]

        if not gas_candidates:
            logger.warning("No gas generation column for %s", region)
            return pd.DataFrame()

        gas_col = gas_candidates[0]
        total_col = total_candidates[0] if total_candidates else None

        df = data[[gas_col, "ng_price"]].dropna().copy()
        df["gas_gen"] = df[gas_col]
        if total_col and total_col in data.columns:
            df["total_gen"] = data[total_col]
        else:
            df["total_gen"] = df["gas_gen"] * 3

        # Features
        df["month"] = df.index.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["ng_price_sq"] = df["ng_price"] ** 2
        df["total_x_price"] = df["total_gen"] * df["ng_price"]

        feature_cols = ["ng_price", "ng_price_sq", "total_gen", "month_sin", "month_cos", "total_x_price"]
        X = df[feature_cols].values
        y = df["gas_gen"].values

        # Expanding-window evaluation with monthly retraining
        # (retrain once per ~21 trading days for speed)
        min_train = 252
        retrain_freq = 21
        predictions = np.full(len(y), np.nan)

        current_model = None
        for i in range(min_train, len(y)):
            if current_model is None or (i - min_train) % retrain_freq == 0:
                current_model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
                )
                current_model.fit(X[:i], y[:i])
            predictions[i] = current_model.predict(X[i:i+1])[0]

        df["predicted_gas_gen"] = predictions
        df["residual"] = df["gas_gen"] - df["predicted_gas_gen"]

        valid = df["predicted_gas_gen"].notna()
        if valid.sum() > 0:
            from sklearn.metrics import r2_score, mean_absolute_percentage_error
            r2 = r2_score(y[valid], predictions[valid])
            mape = mean_absolute_percentage_error(y[valid], predictions[valid]) * 100
            logger.info(
                "%s gas dispatch model: R²=%.3f, MAPE=%.1f%%",
                region, r2, mape,
            )
            df.attrs["r2"] = r2
            df.attrs["mape"] = mape

        return df[["gas_gen", "predicted_gas_gen", "ng_price", "total_gen", "residual"]].dropna()

    def analyse_region(
        self,
        data: pd.DataFrame,
        region: str = "PJM",
    ) -> SparkSpreadResult:
        """Run full spark-spread analysis for a region.

        Args:
            data: Merged data.
            region: Region identifier.

        Returns:
            ``SparkSpreadResult`` with comprehensive analysis.
        """
        hr = self.estimate_implied_heat_rate(data, region)
        spreads = self.compute_spark_spreads(data, region, heat_rate=hr)

        # Seasonal pattern
        if hasattr(spreads.index, 'month'):
            monthly = spreads.groupby(spreads.index.month)["spark_spread"].mean()
            seasonal = monthly.to_dict()
        else:
            seasonal = {}

        # Dispatch threshold: gas price where spark spread = 0
        # 0 = power_price - HR * gas_price
        # power_price ≈ HR * gas_price + premium
        # Threshold where CCGT is uneconomic: gas_price > power_price / HR
        avg_premium = self.power_premium + VOM_COSTS["combined_cycle"]
        threshold = avg_premium / (hr - HEAT_RATES["combined_cycle"]) if hr > HEAT_RATES["combined_cycle"] else 99.0

        return SparkSpreadResult(
            region=region,
            implied_heat_rate=hr,
            avg_spark_spread=float(spreads["spark_spread"].mean()),
            clean_spark_spread=float(spreads["clean_spark_spread"].mean()),
            dispatch_threshold=threshold,
            profitable_hours_pct=float((spreads["spark_spread"] > 0).mean() * 100),
            seasonal_pattern=seasonal,
        )

    def multi_region_analysis(
        self,
        data: pd.DataFrame,
        regions: list[str] | None = None,
    ) -> dict[str, SparkSpreadResult]:
        """Run spark-spread analysis across multiple regions.

        Args:
            data: Merged generation + price data.
            regions: List of regions. If ``None``, auto-detects.

        Returns:
            Dictionary mapping region name to result.
        """
        if regions is None:
            # Auto-detect regions from column names
            regions = set()
            for col in data.columns:
                if "_gas_" in col.lower() or "_total_" in col.lower():
                    prefix = col.split("_")[0]
                    if prefix.isupper() or prefix.istitle():
                        regions.add(prefix)
            regions = sorted(regions)

        results = {}
        for region in regions:
            try:
                results[region] = self.analyse_region(data, region)
            except Exception as e:
                logger.warning("Failed to analyse %s: %s", region, e)

        return results
