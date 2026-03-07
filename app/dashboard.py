"""Streamlit dashboard for the Energy Trading AI system.

Provides a live monitoring dashboard with:
- Current price charts and model forecasts
- Sentiment index visualisation
- Backtest equity curve and performance metrics
- LLM morning research note
- Risk management status

Usage:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Trading AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Energy Trading AI")
    st.markdown("---")

    instrument = st.selectbox(
        "Instrument",
        options=["WTI Crude Oil", "Natural Gas", "Brent Crude", "Heating Oil", "RBOB Gasoline"],
        index=0,
    )
    ticker_map = {
        "WTI Crude Oil": "wti",
        "Natural Gas": "natural_gas",
        "Brent Crude": "brent",
        "Heating Oil": "heating_oil",
        "RBOB Gasoline": "rbob_gasoline",
    }
    instrument_key = ticker_map[instrument]

    lookback_days = st.slider("Lookback (days)", min_value=30, max_value=730, value=365)

    show_signals = st.checkbox("Show Trading Signals", value=True)
    show_bollinger = st.checkbox("Show Bollinger Bands", value=False)
    show_volume = st.checkbox("Show Volume", value=True)

    st.markdown("---")
    st.markdown("**Model Configuration**")
    forecast_horizon = st.selectbox("Forecast Horizon", [1, 5, 10, 20], index=1)

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_price_data(instrument_key: str, days: int) -> pd.DataFrame:
    """Load price data from yfinance (cached for 5 minutes)."""
    try:
        from src.data.price_fetcher import PriceFetcher

        fetcher = PriceFetcher()
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = fetcher.fetch_single(instrument_key, start=start)
        return df
    except Exception as e:
        st.warning(f"Could not load live data: {e}. Using simulated data.")
        rng = np.random.default_rng(42)
        n = days
        close = 70.0 + np.cumsum(rng.normal(0, 1, n))
        idx = pd.date_range(end=datetime.now(), periods=n, freq="B")
        return pd.DataFrame(
            {
                "Open": close * 0.998,
                "High": close * 1.005,
                "Low": close * 0.995,
                "Close": close,
                "Volume": rng.integers(10000, 100000, n).astype(float),
            },
            index=idx,
        )


@st.cache_data(ttl=300)
def load_backtest_results(instrument_key: str) -> pd.DataFrame | None:
    """Load backtest results if available."""
    result_file = Path("data/processed") / f"backtest_{instrument_key}.csv"
    if result_file.exists():
        return pd.read_csv(result_file, index_col=0, parse_dates=True)
    return None


@st.cache_data(ttl=3600)
def load_morning_note() -> str:
    """Load the most recent morning note."""
    notes_dir = Path("data/processed")
    note_files = sorted(notes_dir.glob("morning_note_*.md"), reverse=True)
    if note_files:
        return note_files[0].read_text()
    return "*No morning note available. Run `python scripts/generate_report.py` to generate one.*"


# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
st.title(f"⚡ Energy Trading AI — {instrument}")
st.markdown(
    "Real-time energy market dashboard combining deep learning forecasts, NLP sentiment, and RAG-powered analysis."
)

# Load data
price_df = load_price_data(instrument_key, lookback_days)

if price_df.empty:
    st.error("No price data available. Check your data connection.")
    st.stop()

# ─── Top KPI Cards ───
col1, col2, col3, col4, col5 = st.columns(5)
current_price = float(price_df["Close"].iloc[-1])
prev_price = float(price_df["Close"].iloc[-2]) if len(price_df) > 1 else current_price
price_change = (current_price - prev_price) / prev_price * 100
price_change_30d = (
    current_price / float(price_df["Close"].iloc[-min(30, len(price_df))])
) * 100 - 100
atr = float(
    (price_df["High"] - price_df["Low"]).rolling(14).mean().iloc[-1]
    if len(price_df) >= 14
    else price_df["High"].iloc[-1] - price_df["Low"].iloc[-1]
)
vol_20d = float(
    np.log(price_df["Close"] / price_df["Close"].shift(1)).dropna().tail(20).std() * np.sqrt(252)
)

col1.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f}%")
col2.metric("30-Day Change", f"{price_change_30d:+.1f}%")
col3.metric("ATR (14-day)", f"${atr:.2f}")
col4.metric("Realised Vol (20d)", f"{vol_20d:.1%}")
col5.metric("Lookback Period", f"{lookback_days}d")

st.markdown("---")

# ─── Price Chart ───
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Price & Signals", "📊 Backtest Results", "🧠 AI Morning Note", "⚠️ Risk Monitor"]
)

with tab1:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        n_rows = 2 if show_volume else 1
        row_heights = [0.75, 0.25] if show_volume else [1.0]
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            subplot_titles=(f"{instrument} Price", "Volume")
            if show_volume
            else (f"{instrument} Price",),
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=price_df.index,
                open=price_df["Open"],
                high=price_df["High"],
                low=price_df["Low"],
                close=price_df["Close"],
                name="Price",
                increasing_line_color="#00cc00",
                decreasing_line_color="#ff4444",
            ),
            row=1,
            col=1,
        )

        # Bollinger Bands
        if show_bollinger:
            bb_window = 20
            sma = price_df["Close"].rolling(bb_window).mean()
            std = price_df["Close"].rolling(bb_window).std()
            fig.add_trace(
                go.Scatter(
                    x=price_df.index,
                    y=sma + 2 * std,
                    fill=None,
                    mode="lines",
                    name="BB Upper",
                    line={"color": "rgba(0,200,255,0.3)"},
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=price_df.index,
                    y=sma - 2 * std,
                    fill="tonexty",
                    mode="lines",
                    name="BB Lower",
                    line={"color": "rgba(0,200,255,0.3)"},
                    fillcolor="rgba(0,200,255,0.05)",
                ),
                row=1,
                col=1,
            )

        # Volume
        if show_volume and "Volume" in price_df.columns:
            colors = [
                "green" if c >= o else "red"
                for c, o in zip(price_df["Close"], price_df["Open"], strict=False)
            ]
            fig.add_trace(
                go.Bar(
                    x=price_df.index,
                    y=price_df["Volume"],
                    name="Volume",
                    marker_color=colors,
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_rangeslider_visible=False,
            legend={"x": 0, "y": 1},
            margin={"l": 50, "r": 50, "t": 30, "b": 30},
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.line_chart(price_df["Close"])

    # Technical indicators table
    with st.expander("📊 Technical Indicators"):
        try:
            from src.features.technical import TechnicalFeatures

            tf = TechnicalFeatures()
            tech_df = tf.add_all(price_df).tail(5)
            display_cols = ["Close", "rsi", "macd", "bb_pct", "atr", "return_5d"]
            available_cols = [c for c in display_cols if c in tech_df.columns]
            st.dataframe(tech_df[available_cols].round(4), use_container_width=True)
        except Exception as e:
            st.warning(f"Technical indicators unavailable: {e}")

with tab2:
    backtest_df = load_backtest_results(instrument_key)
    if backtest_df is not None and not backtest_df.empty:
        try:
            from src.backtesting.analysis import BacktestAnalysis

            analysis = BacktestAnalysis(backtest_df)
            report = analysis.compute_metrics()

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{report.total_return:.1%}")
            m2.metric("Sharpe Ratio", f"{report.sharpe_ratio:.3f}")
            m3.metric("Max Drawdown", f"{report.max_drawdown:.1%}")
            m4.metric("Win Rate", f"{report.win_rate:.1%}")

            # Equity curve
            import plotly.graph_objects as go

            fig_equity = go.Figure()
            fig_equity.add_trace(
                go.Scatter(
                    x=backtest_df.index,
                    y=backtest_df["portfolio_value"],
                    name="Strategy",
                    mode="lines",
                    line={"color": "#00ff88", "width": 2},
                )
            )
            fig_equity.update_layout(
                title="Portfolio Equity Curve",
                template="plotly_dark",
                height=400,
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
            )
            st.plotly_chart(fig_equity, use_container_width=True)

            # Drawdown
            fig_dd = go.Figure()
            fig_dd.add_trace(
                go.Scatter(
                    x=backtest_df.index,
                    y=backtest_df["drawdown"] * 100,
                    fill="tozeroy",
                    name="Drawdown",
                    line={"color": "red"},
                    fillcolor="rgba(255,0,0,0.3)",
                )
            )
            fig_dd.update_layout(
                title="Drawdown (%)",
                template="plotly_dark",
                height=250,
                yaxis_title="Drawdown (%)",
                xaxis_title="Date",
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        except Exception as e:
            st.error(f"Backtest analysis error: {e}")
    else:
        st.info(
            "No backtest results found. Run the following to generate them:\n\n"
            "```bash\npython scripts/run_backtest.py\n```"
        )

with tab3:
    note = load_morning_note()
    st.markdown(note)
    if st.button("🔄 Generate New Note"):
        st.info(
            "To generate a new note, run:\n```bash\npython scripts/generate_report.py --print-only\n```"
        )

with tab4:
    st.subheader("Risk Management Status")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        returns = np.log(price_df["Close"] / price_df["Close"].shift(1)).dropna()
        var_95 = abs(float(np.percentile(returns, 5)))
        var_99 = abs(float(np.percentile(returns, 1)))
        st.metric("VaR 95% (Daily)", f"{var_95:.2%}")
        st.metric("VaR 99% (Daily)", f"{var_99:.2%}")
        st.metric("Annualised Volatility", f"{float(returns.std() * np.sqrt(252)):.1%}")
    with col_r2:
        recent_returns = returns.tail(252)
        cummax = (1 + recent_returns).cumprod().cummax()
        current_dd = float(((1 + recent_returns).cumprod().iloc[-1] / cummax.iloc[-1]) - 1)
        st.metric("Current Drawdown (1Y)", f"{current_dd:.2%}")
        st.metric(
            "Max Drawdown (1Y)", f"{float(((1 + recent_returns).cumprod() / cummax - 1).min()):.2%}"
        )
        trading_halted = current_dd < -0.15
        status = "🔴 HALTED" if trading_halted else "🟢 ACTIVE"
        st.metric("Trading Status", status)

# ─── Footer ───
st.markdown("---")
st.caption(
    "⚡ Energy Trading AI System | "
    "Data: yfinance, EIA, FRED, NOAA | "
    "Models: LSTM, Transformer, XGBoost | "
    "NLP: FinBERT | "
    "RAG: LangChain + ChromaDB"
)
