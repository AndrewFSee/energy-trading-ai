"""Streamlit dashboard for the Energy Trading AI system.



Provides a professional energy-desk monitoring dashboard with:

- NG price chart with technicals

- NG Storage Valuation (intrinsic LP + LSMC extrinsic + Greeks)

- Spark Spread analysis across 4 ISOs

- Portfolio VaR (6 methods) + stress testing + backtesting

- LLM-generated morning research note synthesising all models

"""



from __future__ import annotations



import sys

import time

from datetime import datetime, timedelta

from pathlib import Path



import numpy as np

import pandas as pd

import streamlit as st



# Project root

ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(ROOT))



from dotenv import load_dotenv



load_dotenv()



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Page configuration

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.set_page_config(

    page_title="Energy Quant Dashboard",

    page_icon="⚡",

    layout="wide",

    initial_sidebar_state="expanded",

)



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Sidebar

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

with st.sidebar:

    st.title("⚡ Energy Quant Desk")

    st.markdown("---")

    st.markdown("**Market Data**")

    lookback_days = st.slider("Price lookback (days)", 60, 2000, 730)

    show_bollinger = st.checkbox("Bollinger Bands", False)

    show_volume = st.checkbox("Volume", True)



    st.markdown("---")

    st.markdown("**Storage Valuation**")

    facility_type = st.selectbox("Facility", ["Salt Cavern", "Depleted Reservoir"])

    lsmc_paths = st.select_slider("LSMC paths", [1000, 3000, 5000, 10000], value=5000)



    st.markdown("---")

    st.markdown("**VaR Settings**")

    portfolio_value = st.number_input("Portfolio $M", 1.0, 100.0, 10.0, 1.0) * 1_000_000

    var_confidence = st.selectbox("Confidence", [0.95, 0.99, 0.995], index=1)



    st.markdown("---")

    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if st.button("🔄 Refresh all"):

        st.cache_data.clear()

        st.rerun()





# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Data loaders (cached)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@st.cache_data(ttl=600)

def load_ng_prices() -> pd.DataFrame:

    path = ROOT / "data" / "raw" / "prices_natural_gas.csv"

    if not path.exists():

        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")

    for c in ["Close", "Open", "High", "Low", "Volume"]:

        if c in df.columns:

            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"]).sort_index()

    df = df.loc[~df.index.duplicated(keep="first")]

    return df





@st.cache_data(ttl=600)

def load_generation() -> pd.DataFrame:

    path = ROOT / "data" / "raw" / "generation_daily.csv"

    if not path.exists():

        return pd.DataFrame()

    return pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime").sort_index()





@st.cache_data(ttl=600)

def load_composite_signals() -> pd.DataFrame:

    """Load pre-computed composite investment signals."""

    path = ROOT / "data" / "processed" / "composite_signals.csv"

    if not path.exists():

        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()

    return df





@st.cache_data(ttl=3600, show_spinner="Running storage valuation ...")

def run_storage_valuation(facility: str, n_paths: int):

    """Run full storage valuation pipeline."""

    from src.strategy.storage_valuation import StorageValuationEngine, StorageAsset



    prices = load_ng_prices()

    if prices.empty:

        return None

    price_series = prices["Close"].dropna()



    asset = StorageAsset.salt_cavern() if facility == "Salt Cavern" else StorageAsset.depleted_reservoir()

    engine = StorageValuationEngine(asset=asset)

    params = engine.calibrate(price_series)



    # Forward curve

    fwd = engine.forward_curve.curve(1, 12)



    # Intrinsic (Apr -> Mar)

    intrinsic = engine.intrinsic(start_month=4, n_months=12)



    # LSMC extrinsic

    t0 = time.time()

    extrinsic = engine.extrinsic(start_month=4, n_months=12, n_paths=n_paths)

    lsmc_time = time.time() - t0



    # Greeks

    greeks = engine.greeks(start_month=4, n_months=12, n_paths=min(n_paths, 3000))



    return {

        "params": params,

        "fwd": fwd,

        "intrinsic": intrinsic,

        "extrinsic": extrinsic,

        "greeks": greeks,

        "lsmc_time": lsmc_time,

        "asset": asset,

        "current_price": float(price_series.iloc[-1]),

    }





@st.cache_data(ttl=3600, show_spinner="Computing spark spreads ...")

def run_spark_spread():

    """Run spark spread analysis for all regions."""

    from src.strategy.spark_spread import SparkSpreadModel



    gen = load_generation()

    prices = load_ng_prices()

    if gen.empty or prices.empty:

        return None



    # Merge generation + NG prices

    price_s = prices[["Close"]].rename(columns={"Close": "ng_price"})

    merged = gen.join(price_s, how="inner").dropna(subset=["ng_price"])



    # Region mapping: generation CSV uses lowercase prefixes

    REGION_MAP = {"PJM": "pjm", "MISO": "miso", "NYISO": "nyis", "ISONE": "isne"}



    # Rename columns so SparkSpreadModel finds {REGION}_gas_mwh

    for label, prefix in REGION_MAP.items():

        ng_col = f"{prefix}_ng_total_mwh"

        if ng_col in merged.columns:

            merged.rename(columns={ng_col: f"{label}_gas_mwh"}, inplace=True)

        # total = gas + wind + solar

        wnd_col = f"{prefix}_wnd_total_mwh"

        sun_col = f"{prefix}_sun_total_mwh"

        gas_col = f"{label}_gas_mwh"

        parts = []

        for c in [gas_col, wnd_col, sun_col]:

            if c in merged.columns:

                parts.append(merged[c])

        if parts:

            merged[f"{label}_total_mwh"] = sum(parts)



    model = SparkSpreadModel(assumed_power_price_premium=5.0)

    regions = [r for r in REGION_MAP if f"{r}_gas_mwh" in merged.columns]



    all_spreads = {}

    all_hrs = {}

    all_merit = {}

    for r in regions:

        hr = model.estimate_implied_heat_rate(merged, r)

        all_hrs[r] = hr

        all_spreads[r] = model.compute_spark_spreads(merged, r, heat_rate=hr)

        all_merit[r] = model.estimate_merit_order(merged, r)



    return {

        "regions": regions,

        "spreads": all_spreads,

        "hrs": all_hrs,

        "merit": all_merit,

        "n_days": len(merged),

        "date_range": (merged.index.min(), merged.index.max()),

    }





@st.cache_data(ttl=3600, show_spinner="Computing VaR ...")

def run_var_analysis(portfolio_val: float, confidence: float):

    """Run full VaR / stress / backtest suite."""

    from src.strategy.var_model import (

        VaREngine, StressTestEngine, VaRBacktester,

    )



    prices = load_ng_prices()

    if prices.empty:

        return None



    returns = prices["Close"].pct_change().dropna()



    engine = VaREngine(confidence=confidence, horizon_days=1)

    results_1d = engine.full_analysis(returns, portfolio_val)

    tail = engine.tail_statistics(returns)



    engine_10d = VaREngine(confidence=confidence, horizon_days=10)

    results_10d = engine_10d.full_analysis(returns, portfolio_val)



    # Confidence ladder

    ladder = []

    for c in [0.90, 0.95, 0.975, 0.99, 0.995]:

        eng = VaREngine(confidence=c, horizon_days=1)

        r = eng.historical_var(returns, portfolio_val)

        ladder.append({

            "Confidence": f"{c:.1%}",

            "VaR %": f"{r.var*100:.2f}%",

            "VaR $": f"${r.var_dollar:,.0f}",

            "CVaR %": f"{r.cvar*100:.2f}%",

            "CVaR $": f"${r.cvar_dollar:,.0f}",

        })



    # Stress tests

    stress_engine = StressTestEngine()

    positions = {"natural_gas": portfolio_val}

    asset_exp = {"natural_gas": "natural_gas"}

    stress = stress_engine.run_all(positions, asset_exp, portfolio_val)



    # Backtest

    bt_results = {}

    for c in [0.95, 0.99]:

        rolling = VaRBacktester.rolling_var(returns, window=252, confidence=c)

        bt = VaRBacktester.backtest(returns, rolling, confidence=c)

        if bt:

            bt_results[f"{c:.0%}"] = bt



    return {

        "results_1d": results_1d,

        "results_10d": results_10d,

        "tail": tail,

        "ladder": ladder,

        "stress": stress,

        "backtest": bt_results,

        "returns": returns,

    }





# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Load base data

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

price_df = load_ng_prices()

if price_df.empty:

    st.error("No NG price data found at data/raw/prices_natural_gas.csv")

    st.stop()



# Trim to lookback

cutoff = price_df.index.max() - pd.Timedelta(days=lookback_days)

chart_df = price_df.loc[price_df.index >= cutoff].copy()



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Title & KPI Row

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.title("⚡ Energy Quant Dashboard — Natural Gas")

st.markdown(

    "Storage valuation · Spark spreads · Portfolio VaR · LLM research synthesis"

)



cur = float(chart_df["Close"].iloc[-1])

prev = float(chart_df["Close"].iloc[-2]) if len(chart_df) > 1 else cur

chg_1d = (cur - prev) / prev * 100

ret = np.log(chart_df["Close"] / chart_df["Close"].shift(1)).dropna()

vol20 = float(ret.tail(20).std() * np.sqrt(252))

atr14 = float(

    (chart_df["High"] - chart_df["Low"]).rolling(14).mean().iloc[-1]

) if len(chart_df) >= 14 else 0.0

chg30 = (cur / float(chart_df["Close"].iloc[-min(30, len(chart_df))])) * 100 - 100



c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Henry Hub", f"${cur:.2f}", f"{chg_1d:+.2f}%")

c2.metric("30-Day Chg", f"{chg30:+.1f}%")

c3.metric("ATR-14", f"${atr14:.2f}")

c4.metric("Vol (20d ann.)", f"{vol20:.0%}")

c5.metric("Observations", f"{len(price_df):,}")



st.markdown("---")



# ======================================================================

# TABS

# ======================================================================

tab_price, tab_storage, tab_spark, tab_var, tab_signal, tab_note = st.tabs([

    "📈 Price & Technicals",

    "🗏️ Storage Valuation",

    "🔥 Spark Spreads",

    "⚠️ VaR & Risk",

    "🎯 NG Trading Signal",

    "🧠 AI Morning Note",

])



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# TAB 1 — Price & Technicals

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

with tab_price:

    import plotly.graph_objects as go

    from plotly.subplots import make_subplots



    n_rows = 2 if show_volume else 1

    heights = [0.75, 0.25] if show_volume else [1.0]

    fig = make_subplots(

        rows=n_rows, cols=1, shared_xaxes=True,

        row_heights=heights,

        subplot_titles=(

            ("Henry Hub Natural Gas (NG=F)", "Volume") if show_volume

            else ("Henry Hub Natural Gas (NG=F)",)

        ),

    )



    fig.add_trace(go.Candlestick(

        x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],

        low=chart_df["Low"], close=chart_df["Close"], name="Price",

        increasing_line_color="#00cc00", decreasing_line_color="#ff4444",

    ), row=1, col=1)



    if show_bollinger:

        sma = chart_df["Close"].rolling(20).mean()

        std = chart_df["Close"].rolling(20).std()

        fig.add_trace(go.Scatter(

            x=chart_df.index, y=sma + 2 * std, mode="lines",

            name="BB Upper", line=dict(color="rgba(0,200,255,0.3)"),

        ), row=1, col=1)

        fig.add_trace(go.Scatter(

            x=chart_df.index, y=sma - 2 * std, mode="lines",

            name="BB Lower", fill="tonexty",

            line=dict(color="rgba(0,200,255,0.3)"),

            fillcolor="rgba(0,200,255,0.05)",

        ), row=1, col=1)



    if show_volume and "Volume" in chart_df.columns:

        colors = [

            "green" if c >= o else "red"

            for c, o in zip(chart_df["Close"], chart_df["Open"], strict=False)

        ]

        fig.add_trace(go.Bar(

            x=chart_df.index, y=chart_df["Volume"], name="Volume",

            marker_color=colors,

        ), row=2, col=1)



    fig.update_layout(

        template="plotly_dark", height=600,

        xaxis_rangeslider_visible=False,

        margin=dict(l=50, r=50, t=30, b=30),

    )

    st.plotly_chart(fig, width='stretch')



    # Technical indicators

    with st.expander("📊 Technical Indicators (last 5 days)"):

        try:

            from src.features.technical import TechnicalFeatures

            tf = TechnicalFeatures()

            tech = tf.add_all(chart_df).tail(5)

            cols = [c for c in ["Close", "rsi", "macd", "bb_pct", "atr", "return_5d"]

                    if c in tech.columns]

            st.dataframe(tech[cols].round(4), width='stretch')

        except Exception as e:

            st.warning(f"Technical indicators unavailable: {e}")



    # Return distribution

    with st.expander("📉 Return Distribution"):

        fig_hist = go.Figure()

        fig_hist.add_trace(go.Histogram(

            x=ret.values * 100, nbinsx=100,

            marker_color="rgba(0,200,255,0.5)", name="Daily Returns (%)",

        ))

        fig_hist.update_layout(

            template="plotly_dark", height=300,

            xaxis_title="Daily Return (%)", yaxis_title="Count",

            title="NG Daily Return Distribution",

        )

        st.plotly_chart(fig_hist, width='stretch')





# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# TAB 2 — Storage Valuation

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

with tab_storage:

    sv = run_storage_valuation(facility_type, lsmc_paths)

    if sv is None:

        st.error("Cannot run storage valuation - no price data")

    else:

        import plotly.graph_objects as go



        params = sv["params"]

        asset = sv["asset"]

        MM = 1_000_000



        st.subheader(f"Storage Valuation — {asset.name}")

        st.caption(

            f"Capacity: {asset.capacity_bcf:.0f} Bcf  |  "

            f"Injection: {asset.max_injection_bcf_day*1000:.0f} MMcf/d  |  "

            f"Withdrawal: {asset.max_withdrawal_bcf_day*1000:.0f} MMcf/d"

        )



        # OU Parameters

        p1, p2, p3, p4 = st.columns(4)

        p1.metric("kappa (mean-rev)", f"{params['kappa']:.3f}")

        p2.metric("theta (long-run)", f"${params['theta']:.2f}")

        p3.metric("sigma (vol)", f"{params['sigma']:.3f}")

        p4.metric("Half-life", f"{params['half_life_days']:.0f} days")



        col_fwd, col_val = st.columns([1, 1])



        # Forward curve chart

        with col_fwd:

            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",

                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            fwd = sv["fwd"]

            fwd_mean = float(np.mean(fwd))

            colors = ["#ff6b6b" if p > fwd_mean else "#4ecdc4" for p in fwd]

            fig_fwd = go.Figure(go.Bar(

                x=months, y=fwd, marker_color=colors,

                text=[f"${p:.2f}" for p in fwd], textposition="outside",

            ))

            fig_fwd.add_hline(

                y=sv["current_price"], line_dash="dash", line_color="yellow",

                annotation_text=f"Spot ${sv['current_price']:.2f}",

            )

            fig_fwd.update_layout(

                template="plotly_dark", height=350,

                title="Seasonal Forward Curve", yaxis_title="$/MMBtu",

            )

            st.plotly_chart(fig_fwd, width='stretch')



        # Valuation summary

        with col_val:

            intr = sv["intrinsic"]

            extr = sv["extrinsic"]

            greeks = sv["greeks"]



            # Intrinsic

            intr_val = intr.total_value * MM

            st.markdown("#### Intrinsic (LP Optimisation)")

            i1, i2, i3 = st.columns(3)

            i1.metric("Value", f"${intr_val / MM:.2f}M")

            i2.metric("Cycles", f"{intr.cycles:.2f}")

            i3.metric("Spread", f"${intr.spread_captured:.2f}/MMBtu")



            # LSMC

            st.markdown("#### LSMC Extrinsic")

            e1, e2, e3 = st.columns(3)

            e1.metric("Total", f"${extr.total_option_value * MM / MM:.2f}M")

            e2.metric("Extrinsic", f"${extr.extrinsic_value * MM / MM:.2f}M")

            e3.metric("Extrinsic %", f"{extr.extrinsic_pct:.0f}%")

            st.caption(

                f"LSMC: {lsmc_paths:,} paths in {sv['lsmc_time']:.1f}s  |  "

                f"SE: ${extr.std_error * MM / MM:.2f}M"

            )



            # Greeks

            st.markdown("#### Storage Greeks")

            g1, g2, g3, g4 = st.columns(4)

            g1.metric("Delta", f"${greeks.delta * MM / MM:.2f}M")

            g2.metric("Gamma", f"${greeks.gamma * MM / MM:.2f}M")

            g3.metric("Theta", f"${greeks.theta * MM / MM:.2f}M/mo")

            g4.metric("Vega", f"${greeks.vega * MM / MM:.2f}M")



        # Injection/withdrawal schedule

        with st.expander("📋 Optimal Monthly Schedule"):

            sched = intr.schedule

            if isinstance(sched, pd.DataFrame) and not sched.empty:

                st.dataframe(sched.round(4), width='stretch')

            elif isinstance(sched, list):

                sched_df = pd.DataFrame(sched)

                st.dataframe(sched_df.round(4), width='stretch')

            else:

                st.info("Schedule not available in expected format")





# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# TAB 3 — Spark Spreads

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

with tab_spark:

    ss = run_spark_spread()

    if ss is None:

        st.error("Cannot run spark spread - missing generation or price data")

    else:

        import plotly.graph_objects as go



        st.subheader("Spark Spread & Heat Rate Analysis")

        st.caption(

            f"{ss['n_days']} days  "

            f"({ss['date_range'][0].date()} to {ss['date_range'][1].date()})"

        )



        regions = ss["regions"]

        spreads = ss["spreads"]

        hrs = ss["hrs"]



        # Cross-region comparison KPIs

        cols = st.columns(len(regions))

        for col, r in zip(cols, regions):

            s = spreads[r]

            avg_ss = s["spark_spread"].mean()

            prof = (s["spark_spread"] > 0).mean() * 100

            col.metric(r, f"${avg_ss:.1f}/MWh", f"{prof:.0f}% profitable")



        # Spark spread time series

        fig_ss = go.Figure()

        palette = {

            "PJM": "#00ff88", "MISO": "#ff6b6b",

            "NYISO": "#4ecdc4", "ISONE": "#ffd93d",

        }

        for r in regions:

            s = spreads[r]

            rolling = s["spark_spread"].rolling(21, min_periods=5).mean()

            fig_ss.add_trace(go.Scatter(

                x=rolling.index, y=rolling.values, mode="lines", name=r,

                line=dict(color=palette.get(r, "white"), width=2),

            ))

        fig_ss.add_hline(y=0, line_dash="dash", line_color="gray")

        fig_ss.update_layout(

            template="plotly_dark", height=400,

            title="Spark Spread (21-day rolling avg, $/MWh)",

            yaxis_title="$/MWh", xaxis_title="Date",

        )

        st.plotly_chart(fig_ss, width='stretch')



        # Seasonal heatmap & merit order side by side

        col_season, col_merit = st.columns([1, 1])



        with col_season:

            st.markdown("#### Seasonal Spark Spread by Region")

            month_names = [

                "Jan", "Feb", "Mar", "Apr", "May", "Jun",

                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",

            ]

            seasonal_data = {}

            for r in regions:

                s = spreads[r]

                monthly = s.groupby(s.index.month)["spark_spread"].mean()

                seasonal_data[r] = [monthly.get(m, 0) for m in range(1, 13)]



            z = [seasonal_data[r] for r in regions]

            fig_heat = go.Figure(go.Heatmap(

                z=z, x=month_names, y=regions,

                colorscale="RdYlGn", zmid=0,

                text=[[f"${v:.1f}" for v in row] for row in z],

                texttemplate="%{text}",

                colorbar_title="$/MWh",

            ))

            fig_heat.update_layout(

                template="plotly_dark", height=300,

                title="Monthly Avg Spark Spread",

            )

            st.plotly_chart(fig_heat, width='stretch')



        with col_merit:

            st.markdown("#### Generation Mix (Gas Share by Load Level)")

            selected_region = st.selectbox("Region", regions, key="merit_region")

            merit = ss["merit"].get(selected_region)

            if merit and not merit.stack.empty:

                stack = merit.stack.copy()

                pivoted = stack.pivot_table(

                    index="load_quantile", columns="fuel",

                    values="share_pct", aggfunc="first",

                )

                st.dataframe(pivoted.round(1), width='stretch')

                st.caption(

                    f"Marginal fuel: **{merit.marginal_fuel}**  |  "

                    f"Marginal cost: **${merit.marginal_cost_estimate:.1f}/MWh**"

                )

            else:

                st.info("Merit order data not available for this region")



        # Plant-type profitability

        with st.expander("🏭 Profitability by Plant Type"):

            from src.strategy.spark_spread import HEAT_RATES, VOM_COSTS

            rows = []

            for r in regions:

                s = spreads[r]

                for ptype in HEAT_RATES:

                    col_name = f"spark_{ptype}"

                    if col_name in s.columns:

                        avg = s[col_name].mean()

                        prof = (s[col_name] > 0).mean() * 100

                        rows.append({

                            "Region": r, "Plant Type": ptype,

                            "Avg Spark ($/MWh)": round(avg, 1),

                            "Profitable %": round(prof, 0),

                            "Heat Rate": HEAT_RATES[ptype],

                            "VOM ($/MWh)": VOM_COSTS.get(ptype, 0),

                        })

            st.dataframe(

                pd.DataFrame(rows), width='stretch', hide_index=True,

            )



        # Cross-region correlation

        with st.expander("🔗 Spark Spread Correlation Matrix"):

            ss_df = pd.DataFrame(

                {r: spreads[r]["spark_spread"] for r in regions}

            ).dropna()

            if len(ss_df) > 30:

                corr = ss_df.corr()

                fig_corr = go.Figure(go.Heatmap(

                    z=corr.values, x=corr.columns.tolist(),

                    y=corr.index.tolist(),

                    colorscale="RdBu", zmid=1, zmin=0, zmax=1,

                    text=[[f"{v:.2f}" for v in row] for row in corr.values],

                    texttemplate="%{text}",

                ))

                fig_corr.update_layout(

                    template="plotly_dark", height=350,

                    title="Cross-Region Spark Spread Correlation",

                )

                st.plotly_chart(fig_corr, width='stretch')





# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# TAB 4 — VaR & Risk

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

with tab_var:

    var_data = run_var_analysis(portfolio_value, var_confidence)

    if var_data is None:

        st.error("Cannot compute VaR - no price data")

    else:

        import plotly.graph_objects as go



        tail = var_data["tail"]

        st.subheader("Portfolio Value-at-Risk & Stress Testing")



        # Tail stats row

        t1, t2, t3, t4, t5, t6 = st.columns(6)

        t1.metric("Ann. Vol", f"{tail['annualized_vol']:.0%}")

        t2.metric("Skewness", f"{tail['skewness']:.2f}")

        t3.metric("Excess Kurt.", f"{tail['excess_kurtosis']:.1f}")

        t4.metric("Max Drawdown", f"{tail['max_drawdown']:.0%}")

        t5.metric("Worst Day", f"{tail['min_daily_return']:.0%}")

        t6.metric("JB p-val", f"{tail['jarque_bera_pval']:.4f}")



        st.markdown("---")



        # VaR comparison - 1 day & 10 day side by side

        col_1d, col_10d = st.columns(2)



        with col_1d:

            st.markdown(f"#### {var_confidence:.0%} VaR - 1 Day")

            rows_1d = []

            for r in var_data["results_1d"]:

                rows_1d.append({

                    "Method": r.method,

                    "VaR %": f"{r.var*100:.2f}%",

                    "VaR $": f"${r.var_dollar:,.0f}",

                    "CVaR %": f"{r.cvar*100:.2f}%",

                    "CVaR $": f"${r.cvar_dollar:,.0f}",

                })

            st.dataframe(

                pd.DataFrame(rows_1d), width='stretch', hide_index=True,

            )



        with col_10d:

            st.markdown(f"#### {var_confidence:.0%} VaR - 10 Day")

            rows_10d = []

            for r in var_data["results_10d"]:

                rows_10d.append({

                    "Method": r.method,

                    "VaR %": f"{r.var*100:.2f}%",

                    "VaR $": f"${r.var_dollar:,.0f}",

                    "CVaR %": f"{r.cvar*100:.2f}%",

                    "CVaR $": f"${r.cvar_dollar:,.0f}",

                })

            st.dataframe(

                pd.DataFrame(rows_10d), width='stretch', hide_index=True,

            )



        # VaR bar chart comparison

        fig_bar = go.Figure()

        methods_1d = [r.method for r in var_data["results_1d"]]

        var_pcts = [r.var * 100 for r in var_data["results_1d"]]

        cvar_pcts = [r.cvar * 100 for r in var_data["results_1d"]]

        fig_bar.add_trace(go.Bar(

            name="VaR", x=methods_1d, y=var_pcts, marker_color="#ff6b6b",

        ))

        fig_bar.add_trace(go.Bar(

            name="CVaR", x=methods_1d, y=cvar_pcts,

            marker_color="#ff4444", opacity=0.6,

        ))

        fig_bar.update_layout(

            template="plotly_dark", height=350, barmode="group",

            title=f"{var_confidence:.0%} 1-Day VaR & CVaR by Method (%)",

            yaxis_title="Loss %",

        )

        st.plotly_chart(fig_bar, width='stretch')



        # Confidence ladder

        with st.expander("📐 VaR Confidence Ladder"):

            st.dataframe(

                pd.DataFrame(var_data["ladder"]),

                width='stretch', hide_index=True,

            )



        # Stress testing

        st.markdown("---")

        st.markdown("#### 🌪️ Energy Stress Scenarios")



        stress = var_data["stress"]

        stress_rows = []

        for sr in stress:

            stress_rows.append({

                "Scenario": sr.scenario.replace("_", " ").title(),

                "Portfolio P&L": f"${sr.portfolio_pnl:+,.0f}",

                "P&L %": f"{sr.portfolio_pnl_pct:+.1f}%",

                "Worst Asset": sr.worst_asset,

                "Description": sr.description[:80],

            })

        st.dataframe(

            pd.DataFrame(stress_rows), width='stretch', hide_index=True,

        )



        # Stress test waterfall chart

        fig_stress = go.Figure(go.Waterfall(

            x=[s.scenario.replace("_", " ").title() for s in stress],

            y=[s.portfolio_pnl for s in stress],

            connector=dict(line=dict(color="rgba(63,63,63,0.5)")),

            increasing=dict(marker_color="#00cc00"),

            decreasing=dict(marker_color="#ff4444"),

            text=[f"${s.portfolio_pnl:+,.0f}" for s in stress],

        ))

        fig_stress.update_layout(

            template="plotly_dark", height=350,

            title="Stress Scenario P&L Impact", yaxis_title="P&L ($)",

        )

        st.plotly_chart(fig_stress, width='stretch')



        # VaR Backtest

        with st.expander("📊 VaR Model Backtesting (Kupiec Test)"):

            for label, bt in var_data["backtest"].items():

                status = "✅ Accepted" if bt["model_accepted"] else "❌ Rejected"

                st.markdown(f"**{label} VaR Backtest** — {status}")

                b1, b2, b3, b4 = st.columns(4)

                b1.metric("Exceedances", f"{bt['n_exceedances']} / {bt['n_observations']}")

                b2.metric("Rate", f"{bt['exceedance_rate']*100:.2f}%")

                b3.metric("Kupiec LR", f"{bt['kupiec_lr_stat']:.2f}")

                b4.metric("p-value", f"{bt['kupiec_p_value']:.4f}")









# -----------------------------------------------------------------------

# TAB 5 -- NG Investment Signal

# -----------------------------------------------------------------------

with tab_signal:

    import plotly.graph_objects as go

    from plotly.subplots import make_subplots as _make_subplots



    st.subheader("Composite NG Trading Signal")

    st.caption(

        "HMM regime detection + storage anomaly + seasonal patterns + "

        "technical trend/momentum + mean reversion.  "

        "Signal: LONG (green), SHORT (red), or FLAT (grey)."

    )



    sig_df = load_composite_signals()

    if sig_df.empty:

        st.warning(

            "No signal data found.  Run `python scripts/run_composite_signal.py` first."

        )

    else:

        # ---- Current signal banner ----

        latest = sig_df.iloc[-1]

        sig_val = int(latest["signal"])

        current_signal = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(sig_val, "FLAT")



        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Current Signal", current_signal)

        c2.metric("Composite Score", f"{latest['composite_score']:.3f}")

        c3.metric("Position Size", f"{latest['position_size']:.0%}")

        c4.metric("Realised Vol", f"{latest['realised_vol']:.0%}")



        # ---- Sub-signal gauges ----

        st.markdown("**Sub-signal values (latest):**")

        s1, s2, s3, s4, s5, s6, s7 = st.columns(7)

        s1.metric("Regime (HMM)", f"{latest['regime_signal']:+.2f}")

        s2.metric("Storage Anomaly", f"{latest['storage_signal']:+.2f}")

        s3.metric("Seasonal", f"{latest['seasonal_signal']:+.2f}")

        s4.metric("Technical", f"{latest['technical_signal']:+.2f}")

        mr_val = latest.get("mean_reversion_signal", 0.0)

        s5.metric("Mean Reversion", f"{mr_val:+.2f}")

        wth_val = latest.get("weather_signal", 0.0)

        s6.metric("Weather", f"{wth_val:+.2f}")

        sent_val = latest.get("sentiment_signal", 0.0)

        s7.metric("Sentiment/GPR", f"{sent_val:+.2f}")



        st.markdown("---")



        # ---- Price + signal overlay chart ----

        sig_years = st.slider("Signal lookback (years)", 1, 20, 5, key="sig_lb")

        cutoff = sig_df.index[-1] - pd.DateOffset(years=sig_years)

        plot_df = sig_df.loc[cutoff:]



        fig = _make_subplots(

            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,

            row_heights=[0.50, 0.25, 0.25],

            subplot_titles=["NG Price + Signal Overlay", "Composite Score", "Sub-Signals"],

        )



        # Price line

        fig.add_trace(

            go.Scatter(x=plot_df.index, y=plot_df["price"], mode="lines",

                       name="NG Close", line=dict(color="#636EFA", width=1.5)),

            row=1, col=1,

        )



        # Shade INVEST periods green

        invest_mask = plot_df["signal"] == 1

        invest_periods = []

        in_period = False

        for i, (dt, val) in enumerate(invest_mask.items()):

            if val and not in_period:

                start = dt

                in_period = True

            elif not val and in_period:

                invest_periods.append((start, invest_mask.index[i - 1]))

                in_period = False

        if in_period:

            invest_periods.append((start, invest_mask.index[-1]))



        for s, e in invest_periods:

            fig.add_vrect(

                x0=s, x1=e, fillcolor="rgba(0,200,0,0.12)", line_width=0,

                row=1, col=1,

            )



        # Shade SHORT periods red

        short_mask = plot_df["signal"] == -1

        short_periods = []

        in_period = False

        for i, (dt, val) in enumerate(short_mask.items()):

            if val and not in_period:

                start = dt

                in_period = True

            elif not val and in_period:

                short_periods.append((start, short_mask.index[i - 1]))

                in_period = False

        if in_period:

            short_periods.append((start, short_mask.index[-1]))



        for s, e in short_periods:

            fig.add_vrect(

                x0=s, x1=e, fillcolor="rgba(255,0,0,0.10)", line_width=0,

                row=1, col=1,

            )



        # Composite score

        fig.add_trace(

            go.Scatter(x=plot_df.index, y=plot_df["composite_score"], mode="lines",

                       name="Composite", line=dict(color="#AB63FA", width=1.2)),

            row=2, col=1,

        )

        fig.add_hline(y=0.05, line_dash="dot", line_color="green",

                       annotation_text="Long threshold", row=2, col=1)

        fig.add_hline(y=-0.15, line_dash="dot", line_color="red",

                       annotation_text="Short threshold", row=2, col=1)

        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)



        # Sub-signals

        sub_signal_cols = [

            ("regime_signal", "#636EFA"),

            ("storage_signal", "#EF553B"),

            ("seasonal_signal", "#00CC96"),

            ("technical_signal", "#FFA15A"),

            ("mean_reversion_signal", "#AB63FA"),

            ("weather_signal", "#19D3F3"),

            ("sentiment_signal", "#FF6692"),

        ]

        for col_name, clr in sub_signal_cols:

            if col_name not in plot_df.columns:

                continue

            fig.add_trace(

                go.Scatter(x=plot_df.index, y=plot_df[col_name], mode="lines",

                           name=col_name.replace("_signal", "").title(),

                           line=dict(width=1, color=clr), opacity=0.8),

                row=3, col=1,

            )



        fig.update_layout(height=750, showlegend=True, template="plotly_white",

                          legend=dict(orientation="h", y=-0.05))

        fig.update_yaxes(title_text="$/MMBtu", row=1)

        fig.update_yaxes(title_text="Score", row=2)

        fig.update_yaxes(title_text="Signal", row=3)

        st.plotly_chart(fig, use_container_width=True)



        # ---- Signal stats ----

        with st.expander("Signal Statistics"):

            long_days = (sig_df["signal"] == 1).sum()

            short_days = (sig_df["signal"] == -1).sum()

            flat_days = (sig_df["signal"] == 0).sum()

            total_days = len(sig_df)

            st.markdown(f"- **Period:** {sig_df.index[0].date()} to {sig_df.index[-1].date()}")

            st.markdown(f"- **LONG days:** {long_days:,} ({100*long_days/total_days:.1f}%)")

            st.markdown(f"- **SHORT days:** {short_days:,} ({100*short_days/total_days:.1f}%)")

            st.markdown(f"- **FLAT days:** {flat_days:,} ({100*flat_days/total_days:.1f}%)")

            st.markdown(f"- **Current regime signal:** {latest['regime_signal']:+.2f}")

            st.markdown(f"- **Current vol scalar:** {latest['vol_scalar']:.2f}")



            # Annual signal frequency

            annual = sig_df["signal"].resample("YE").mean()

            st.markdown("**Annual INVEST frequency:**")

            ann_df = pd.DataFrame({

                "Year": annual.index.year,

                "% Days Invested": (annual.values * 100).round(1),

            })

            st.dataframe(ann_df.set_index("Year").T, use_container_width=True)





# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# TAB 5 — AI Morning Note

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

with tab_note:

    st.subheader("🧠 AI-Generated Energy Market Morning Note")

    st.caption(

        "Uses GPT-4o to synthesise storage, spark spread, and risk analytics "

        "into a professional research note."

    )



    # Load saved notes

    notes_dir = ROOT / "data" / "processed"

    note_files = sorted(notes_dir.glob("morning_note_*.md"), reverse=True)



    # Offer to generate a new one on-demand

    if st.button("🔄 Generate New Morning Note (calls OpenAI)", key="gen_note"):

        with st.spinner("Generating morning note via GPT-4o ..."):

            try:

                from src.rag.llm_client import LLMClient



                # Gather data for the prompt from all models

                sv_data = run_storage_valuation(facility_type, lsmc_paths)

                ss_data = run_spark_spread()

                vr_data = run_var_analysis(portfolio_value, var_confidence)



                # --- Build comprehensive prompt ---

                prompt_parts = []

                prompt_parts.append(

                    f"Date: {datetime.now().strftime('%A, %d %B %Y')}"

                )

                prompt_parts.append(

                    f"Henry Hub NG spot: ${cur:.2f}/MMBtu ({chg_1d:+.2f}% today)"

                )

                prompt_parts.append(f"20-day annualised volatility: {vol20:.0%}")

                prompt_parts.append(f"ATR-14: ${atr14:.2f}")



                if sv_data:

                    p = sv_data["params"]

                    sv_intr = sv_data["intrinsic"]

                    sv_extr = sv_data["extrinsic"]

                    sv_gk = sv_data["greeks"]

                    prompt_parts.append(

                        f"\nSTORAGE VALUATION ({sv_data['asset'].name}):\n"

                        f"  OU kappa={p['kappa']:.3f}, theta=${p['theta']:.2f}, "

                        f"sigma={p['sigma']:.3f}, "

                        f"half-life={p['half_life_days']:.0f}d\n"

                        f"  Winter premium: +{p['winter_premium_pct']:.1f}%, "

                        f"Summer discount: -{p['summer_discount_pct']:.1f}%\n"

                        f"  Intrinsic (Apr->Mar): "

                        f"${sv_intr.total_value * 1e6 / 1e6:.2f}M, "

                        f"{sv_intr.cycles:.2f} cycles, "

                        f"spread=${sv_intr.spread_captured:.2f}/MMBtu\n"

                        f"  LSMC total: ${sv_extr.total_option_value * 1e6 / 1e6:.2f}M, "

                        f"extrinsic: "

                        f"${sv_extr.extrinsic_value * 1e6 / 1e6:.2f}M "

                        f"({sv_extr.extrinsic_pct:.0f}%)\n"

                        f"  Greeks: Delta=${sv_gk.delta * 1e6 / 1e6:.2f}M, "

                        f"Gamma=${sv_gk.gamma * 1e6 / 1e6:.2f}M, "

                        f"Theta=${sv_gk.theta * 1e6 / 1e6:.2f}M/mo, "

                        f"Vega=${sv_gk.vega * 1e6 / 1e6:.2f}M"

                    )



                if ss_data:

                    spark_lines = []

                    for r in ss_data["regions"]:

                        s = ss_data["spreads"][r]

                        avg_ss = s["spark_spread"].mean()

                        prof_pct = (s["spark_spread"] > 0).mean() * 100

                        hr = ss_data["hrs"][r]

                        spark_lines.append(

                            f"  {r}: avg ${avg_ss:.1f}/MWh, "

                            f"{prof_pct:.0f}% profitable, HR={hr:.1f}"

                        )

                    prompt_parts.append(

                        "\nSPARK SPREAD ANALYSIS:\n" + "\n".join(spark_lines)

                    )



                if vr_data:

                    vr_tail = vr_data["tail"]

                    hist_var = next(

                        (r for r in vr_data["results_1d"]

                         if "Historical" in r.method), None,

                    )

                    prompt_parts.append(

                        f"\nRISK METRICS "

                        f"(${portfolio_value/1e6:.0f}M portfolio):\n"

                        f"  Annualised vol: {vr_tail['annualized_vol']:.0%}\n"

                        f"  Excess kurtosis: "

                        f"{vr_tail['excess_kurtosis']:.1f} (fat tails)\n"

                        f"  Max drawdown: {vr_tail['max_drawdown']:.0%}\n"

                        f"  Historical VaR {var_confidence:.0%} 1d: "

                        + (

                            f"${hist_var.var_dollar:,.0f}"

                            if hist_var else "N/A"

                        )

                    )

                    worst = vr_data["stress"][0]

                    prompt_parts.append(

                        f"  Worst stress scenario: {worst.scenario} "

                        f"({worst.portfolio_pnl_pct:+.1f}%)"

                    )



                full_context = "\n".join(prompt_parts)



                system_prompt = (

                    "You are a senior energy market analyst on a natural gas "

                    "trading desk. Your audience is portfolio managers and "

                    "senior traders. Write with precision, cite specific "

                    "numbers, and be direct about risks. Use professional "

                    "trading desk language."

                )



                now_str = datetime.now().strftime("%d %B %Y")

                now_ts = datetime.now().strftime("%Y-%m-%d %H:%M")



                user_prompt = (

                    "Generate a professional morning research note using the "

                    "following quantitative analytics from our models.\n\n"

                    f"{full_context}\n\n"

                    "Structure the note as:\n\n"

                    f"## Energy Market Morning Note - {now_str}\n\n"

                    "### Market Snapshot\n"

                    "[Current price, vol regime, key overnight moves in "

                    "2-3 sentences]\n\n"

                    "### Storage Valuation Update\n"

                    "[Interpret the OU dynamics, forward curve shape, "

                    "intrinsic vs extrinsic value, and what the Greeks "

                    "tell us about positioning. What should the desk do?]\n\n"

                    "### Spark Spread & Generation Outlook\n"

                    "[Which regions show the best economics? What does "

                    "seasonality suggest for the coming month? Gas burn "

                    "implications for pipeline flows?]\n\n"

                    "### Risk Dashboard\n"

                    "[Interpret VaR numbers across methods - highlight "

                    "where Gaussian VaR understates risk vs Student-t / "

                    "Historical. Discuss the worst stress scenario and "

                    "what it means for position sizing.]\n\n"

                    "### Trade Ideas\n"

                    "[2 specific trade ideas with entry, target, stop "

                    "levels grounded in the analytics above]\n\n"

                    "### Risks to Watch\n"

                    "[3 bullet points on key risks for today's session]\n\n"

                    "---\n"

                    "*AI-generated research note. Not investment advice. "

                    f"Model outputs as of {now_ts}.*"

                )



                llm = LLMClient(

                    provider="openai", model="gpt-4o",

                    temperature=0.2, max_tokens=2000,

                )

                note_text = llm.complete(

                    user_prompt, system_prompt=system_prompt,

                )



                # Save to file

                fname = (

                    f"morning_note_"

                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

                )

                (notes_dir / fname).write_text(note_text, encoding="utf-8")



                st.markdown(note_text)

                st.success(f"Note saved to data/processed/{fname}")



            except Exception as e:

                st.error(f"Note generation failed: {e}")

                import traceback

                st.code(traceback.format_exc())

    else:

        # Show most recent saved note

        if note_files:

            latest = note_files[0]

            st.caption(f"Showing: {latest.name}")

            st.markdown(latest.read_text(encoding="utf-8"))

        else:

            st.info(

                "No saved morning notes found. Click the button above to "

                "generate one, or run:\n"

                "```bash\npython scripts/generate_report.py --print-only\n```"

            )





# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Footer

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.markdown("---")

st.caption(

    "Energy Quant Dashboard  |  "

    "Data: EIA, yfinance, NOAA  |  "

    "Models: OU/LSMC storage, spark spread, VaR/CVaR  |  "

    "NLP: GPT-4o  |  "

    "Stack: Streamlit + Plotly + scipy"

)



