"""Geopolitical Risk (GPR) Index fetcher.

Downloads the daily GPR index from Matteo Iacoviello's website (Federal Reserve).
The GPR index measures geopolitical risk derived from newspaper articles and
is available from 1985 to present.

Reference: Caldara, D. & Iacoviello, M. (2022), "Measuring Geopolitical Risk",
American Economic Review.

Source: https://www.matteoiacoviello.com/gpr.htm
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Daily GPR data (recent file includes full history)
GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
# Export file with monthly data + sub-indices
GPR_EXPORT_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"


def fetch_gpr_daily(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch the daily Geopolitical Risk Index.

    Downloads the daily GPR index and computes derived features useful
    for energy price forecasting (rolling stats, regime indicators).

    Args:
        start: Start date ``YYYY-MM-DD``.
        end: End date ``YYYY-MM-DD``.

    Returns:
        DataFrame with columns: ``gpr``, ``gpr_ma7``, ``gpr_ma30``,
        ``gpr_zscore``, ``gpr_elevated``, ``gpr_spike``, ``gpr_momentum``.
    """
    logger.info("Fetching daily GPR index from %s", GPR_DAILY_URL)
    try:
        df = pd.read_excel(GPR_DAILY_URL)
    except Exception:
        logger.warning("Primary GPR URL failed, trying export URL")
        try:
            df = pd.read_excel(GPR_EXPORT_URL, sheet_name=0)
        except Exception as exc:
            logger.error("Failed to download GPR data: %s", exc)
            return pd.DataFrame()

    # The daily file has columns like 'date' and 'GPRD' (or 'GPR')
    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Find the date column
    date_col = None
    for candidate in ("date", "day", "obs"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        # Use first column as date
        date_col = df.columns[0]

    # Find the GPR value column
    gpr_col = None
    for candidate in ("gprd", "gpr", "gpr_daily", "gpr daily"):
        if candidate in df.columns:
            gpr_col = candidate
            break
    if gpr_col is None:
        # Use the second numeric column
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            gpr_col = numeric_cols[0]
        else:
            logger.error("Could not find GPR value column in: %s", list(df.columns))
            return pd.DataFrame()

    result = pd.DataFrame()
    result.index = pd.to_datetime(df[date_col])
    result.index.name = "Date"
    result["gpr"] = df[gpr_col].values

    # Drop NaN
    result = result.dropna(subset=["gpr"])

    # Filter date range
    if start:
        result = result[result.index >= pd.Timestamp(start)]
    if end:
        result = result[result.index <= pd.Timestamp(end)]

    # Derived features
    result["gpr_ma7"] = result["gpr"].rolling(7, min_periods=1).mean()
    result["gpr_ma30"] = result["gpr"].rolling(30, min_periods=5).mean()

    roll60 = result["gpr"].rolling(60, min_periods=10)
    result["gpr_zscore"] = (result["gpr"] - roll60.mean()) / roll60.std()

    # Regime: GPR above its 75th percentile (expanding)
    expanding_q75 = result["gpr"].expanding(min_periods=60).quantile(0.75)
    result["gpr_elevated"] = (result["gpr"] > expanding_q75).astype(int)

    # Spike: GPR > 2 standard deviations above its 60-day mean
    result["gpr_spike"] = (result["gpr_zscore"] > 2.0).astype(int)

    # Momentum: 5-day change in GPR
    result["gpr_momentum"] = result["gpr"].diff(5)

    logger.info(
        "GPR index: %d daily observations (%s to %s), 8 features",
        len(result),
        result.index.min().strftime("%Y-%m-%d"),
        result.index.max().strftime("%Y-%m-%d"),
    )
    return result
