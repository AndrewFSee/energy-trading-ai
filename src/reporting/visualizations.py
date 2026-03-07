"""Plotting and visualisation utilities for the Energy Trading AI system.

Provides reusable Plotly-based chart functions for:
- Price and equity curves
- Feature importance bar charts
- Drawdown analysis
- Sentiment index overlay
- Backtest performance tearsheet
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Visualizer:
    """Creates interactive Plotly charts for the Energy Trading AI system.

    All methods return Plotly ``Figure`` objects that can be displayed
    in Jupyter notebooks, Streamlit dashboards, or saved as HTML/PNG.

    Attributes:
        theme: Plotly template name (e.g. ``"plotly_dark"``).
        default_height: Default chart height in pixels.
        default_width: Default chart width in pixels.
    """

    def __init__(
        self,
        theme: str = "plotly_dark",
        default_height: int = 500,
        default_width: int = 1000,
    ) -> None:
        """Initialise the visualizer.

        Args:
            theme: Plotly template for styling.
            default_height: Default chart height in pixels.
            default_width: Default chart width in pixels.
        """
        self.theme = theme
        self.default_height = default_height
        self.default_width = default_width

    def _check_plotly(self) -> None:
        """Ensure plotly is available."""
        try:
            import plotly  # noqa: F401
        except ImportError as e:
            raise ImportError("plotly is required for visualizations") from e

    def plot_price_with_signals(
        self,
        prices: pd.Series,
        signals: pd.Series | None = None,
        title: str = "Price Chart with Trading Signals",
    ):  # type: ignore[return]
        """Plot price series with overlaid buy/sell signals.

        Args:
            prices: Price series with ``DatetimeIndex``.
            signals: Optional signal series (LONG=1, FLAT=0, SHORT=-1).
            title: Chart title.

        Returns:
            Plotly ``Figure`` object.
        """
        self._check_plotly()
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices.values,
                mode="lines",
                name="Price",
                line={"color": "white", "width": 1.5},
            )
        )

        if signals is not None:
            long_idx = signals[signals == 1].index
            short_idx = signals[signals == -1].index

            if len(long_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=long_idx,
                        y=prices.reindex(long_idx),
                        mode="markers",
                        name="Long",
                        marker={"symbol": "triangle-up", "size": 10, "color": "green"},
                    )
                )
            if len(short_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=short_idx,
                        y=prices.reindex(short_idx),
                        mode="markers",
                        name="Short",
                        marker={"symbol": "triangle-down", "size": 10, "color": "red"},
                    )
                )

        fig.update_layout(
            title=title,
            template=self.theme,
            height=self.default_height,
            width=self.default_width,
            xaxis_title="Date",
            yaxis_title="Price ($)",
        )
        return fig

    def plot_equity_curve(
        self,
        portfolio_values: pd.Series,
        benchmark: pd.Series | None = None,
        title: str = "Strategy Equity Curve",
    ):  # type: ignore[return]
        """Plot portfolio equity curve with optional benchmark comparison.

        Args:
            portfolio_values: Portfolio value series.
            benchmark: Optional benchmark value series.
            title: Chart title.

        Returns:
            Plotly ``Figure`` object.
        """
        self._check_plotly()
        import plotly.graph_objects as go

        # Normalise to 100
        norm_port = portfolio_values / portfolio_values.iloc[0] * 100
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=norm_port.index,
                y=norm_port.values,
                mode="lines",
                name="Strategy",
                line={"color": "#00ff88", "width": 2},
            )
        )

        if benchmark is not None:
            norm_bench = benchmark.reindex(norm_port.index).ffill()
            norm_bench = norm_bench / norm_bench.iloc[0] * 100
            fig.add_trace(
                go.Scatter(
                    x=norm_bench.index,
                    y=norm_bench.values,
                    mode="lines",
                    name="Benchmark",
                    line={"color": "#aaaaaa", "width": 1.5, "dash": "dash"},
                )
            )

        fig.update_layout(
            title=title,
            template=self.theme,
            height=self.default_height,
            width=self.default_width,
            xaxis_title="Date",
            yaxis_title="Normalised Value (Base=100)",
        )
        return fig

    def plot_drawdown(self, drawdown: pd.Series, title: str = "Portfolio Drawdown"):  # type: ignore[return]
        """Plot drawdown chart with shaded area.

        Args:
            drawdown: Drawdown series (negative values).
            title: Chart title.

        Returns:
            Plotly ``Figure`` object.
        """
        self._check_plotly()
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode="lines",
                name="Drawdown",
                fill="tozeroy",
                line={"color": "red"},
                fillcolor="rgba(255,0,0,0.3)",
            )
        )
        fig.update_layout(
            title=title,
            template=self.theme,
            height=400,
            width=self.default_width,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
        )
        return fig

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance",
    ):  # type: ignore[return]
        """Plot horizontal bar chart of feature importances.

        Args:
            importance_df: DataFrame with ``feature`` and ``importance`` columns.
            top_n: Number of top features to display.
            title: Chart title.

        Returns:
            Plotly ``Figure`` object.
        """
        self._check_plotly()
        import plotly.express as px

        df = importance_df.head(top_n).sort_values("importance", ascending=True)
        fig = px.bar(
            df,
            x="importance",
            y="feature",
            orientation="h",
            title=title,
            template=self.theme,
            color="importance",
            color_continuous_scale="viridis",
        )
        fig.update_layout(height=max(400, top_n * 25), width=self.default_width)
        return fig

    def plot_sentiment_overlay(
        self,
        prices: pd.Series,
        sentiment: pd.Series,
        title: str = "Price vs Sentiment Index",
    ):  # type: ignore[return]
        """Plot price and sentiment index on dual y-axes.

        Args:
            prices: Price series.
            sentiment: Sentiment index series [-1, +1].
            title: Chart title.

        Returns:
            Plotly ``Figure`` object.
        """
        self._check_plotly()
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=prices.index, y=prices.values, name="Price", line={"color": "white"}),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=sentiment.index, y=sentiment.values, name="Sentiment", line={"color": "orange"}
            ),
            secondary_y=True,
        )
        fig.update_layout(title=title, template=self.theme, height=self.default_height)
        fig.update_yaxes(title_text="Price ($)", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Index", secondary_y=True)
        return fig

    def plot_monthly_returns_heatmap(
        self,
        monthly_returns_pivot: pd.DataFrame,
        title: str = "Monthly Returns Heatmap",
    ):  # type: ignore[return]
        """Plot a calendar heatmap of monthly returns.

        Args:
            monthly_returns_pivot: Pivot table from ``BacktestAnalysis.monthly_returns()``.
            title: Chart title.

        Returns:
            Plotly ``Figure`` object.
        """
        self._check_plotly()
        import plotly.graph_objects as go

        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        data = monthly_returns_pivot.values * 100
        text = np.where(
            ~np.isnan(data),
            [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in data],
            "",
        )

        fig = go.Figure(
            go.Heatmap(
                z=data,
                x=[str(c) for c in monthly_returns_pivot.columns],
                y=[month_names[i - 1] for i in monthly_returns_pivot.index],
                text=text,
                texttemplate="%{text}",
                colorscale="RdYlGn",
                zmid=0,
                showscale=True,
            )
        )
        fig.update_layout(title=title, template=self.theme, height=500, width=self.default_width)
        return fig
