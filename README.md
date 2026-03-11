# ⚡ Energy Trading AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Quantitative analytics platform for power & gas markets** — storage valuation, spark-spread modelling, portfolio VaR, demand/generation forecasting, and AI-generated research notes, all backed by live EIA/FRED/weather data and served through an interactive Streamlit dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Models](#core-models)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Dashboard](#dashboard)
- [Data Sources](#data-sources)
- [RAG / LLM Setup](#rag--llm-setup)
- [License](#license)

---

## Overview

This project is a full-stack energy quantitative analytics platform covering the core competencies of a power/gas trading desk:

| Domain | What's Implemented |
|--------|--------------------|
| **Physical Asset Valuation** | Natural gas storage valuation — intrinsic (LP) + extrinsic (LSMC) + Greeks |
| **Spread Analytics** | Spark-spread modelling across 4 ISO regions with dispatch and merit-order analysis |
| **Risk Management** | Portfolio VaR (6 methods), energy stress scenarios, Kupiec backtesting |
| **Load Forecasting** | Electricity demand prediction (XGBoost, LSTM) |
| **Generation Forecasting** | Wind generation and NG production direction models |
| **Research Automation** | Agentic morning briefing — GPT-4o synthesises all model outputs into a research note |
| **Interactive Dashboard** | 5-tab Streamlit app with live data, charts, and LLM narrative |

Data is sourced from **EIA**, **FRED**, **yfinance**, **NOAA**, and **NewsAPI** — no static datasets.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   DATA INGESTION LAYER                    │
│  yfinance (NG/CL/UNG)  │  EIA v2 (storage, generation,  │
│  production)  │  FRED (macro)  │  NOAA (weather)  │ PDFs │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING & NLP                     │
│  Technical indicators  │  Macro features  │  Seasonal     │
│  FinBERT sentiment     │  Weather features │  Calendar    │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌─────────────────┐ ┌──────────────┐ ┌─────────────────────┐
│ QUANT MODELS    │ │  FORECASTING │ │   RAG / LLM         │
│ Storage Valuat. │ │  XGB Demand  │ │ 40k-doc VectorStore │
│ Spark Spreads   │ │  LSTM Demand │ │ GPT-4o Morning Note │
│ Portfolio VaR   │ │  Wind Gen    │ │ Signal Generation   │
│                 │ │  NG Prod Dir │ │                     │
└────────┬────────┘ └──────┬───────┘ └──────────┬──────────┘
         └─────────────────┼─────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│              STREAMLIT DASHBOARD (5 tabs)                  │
│  Price & Technicals │ Storage Valuation │ Spark Spreads   │
│  VaR & Risk         │ AI Morning Note                     │
└──────────────────────────────────────────────────────────┘
```

---

## Core Models

### 1. Natural Gas Storage Valuation

Values gas storage facilities using industry-standard methods:

- **Ornstein-Uhlenbeck calibration** on historical Henry Hub prices (κ=1.018, θ=$4.33, σ=3.143)
- **Intrinsic valuation** via linear programming — optimal injection/withdrawal schedule against the forward curve
- **Extrinsic valuation** via Least-Squares Monte Carlo (Longstaff-Schwartz) — captures optionality above intrinsic
- **Greeks** — delta, gamma, theta, vega for hedging
- **Two facility types** — salt cavern (high deliverability) and depleted reservoir (high working gas)

### 2. Spark-Spread Analytics

Models gas-fired generation profitability across **PJM, MISO, NYISO, and ISO-NE**:

- Gas-share-dependent power-price estimation with seasonal + volatility adjustments
- Regional dispatch model (gradient-boosted regression)
- Merit-order / supply-curve construction by plant type
- Cross-regional correlation analysis
- Clean spark spread net of variable O&M and carbon costs

### 3. Portfolio Value-at-Risk

Six VaR methodologies on a multi-asset energy portfolio (NG=F, CL=F, UNG):

- Historical Simulation, Parametric (Normal), Parametric (Student-t), Cornish-Fisher, EWMA, Monte Carlo
- Component VaR — marginal contribution of each position
- **6 energy-specific stress scenarios** — polar vortex, hurricane season, pipeline disruption, demand destruction, LNG export surge, storage surprise
- **Kupiec backtesting** to validate VaR model calibration

### 4. Demand & Generation Forecasting

- **Electricity demand** — XGBoost (R²=0.91) and LSTM (R²=0.70) using weather, calendar, and economic features
- **Wind generation** — XGBoost regressor (R²=0.50, 75-fold expanding-window CV) across 4 regions
- **NG production direction** — XGBoost classifier (81% accuracy, F1=0.83, 25pp above base rate)

### 5. RAG Pipeline & Agentic Morning Briefing

- **40,850 documents** ingested into a ChromaDB vector store (energy trading books, EIA reports, research papers)
- **6-step agentic workflow**: market snapshot → storage analysis → spark-spread summary → risk metrics → RAG context retrieval → GPT-4o synthesis
- Produces a professional morning research note with market views and trade ideas

---

## Key Results

| Model | Metric | Result |
|-------|--------|--------|
| Storage – Intrinsic (Salt Cavern) | NPV | $2.98M (1.55 cycles) |
| Storage – LSMC Total | NPV | $74.17M (96% extrinsic optionality) |
| Spark Spread – PJM Dispatch | R² / MAPE | 0.920 / 4.5% |
| Spark Spread – Seasonal Range | $/MWh | Jul $9.5–10 → Apr $2.5–3.3 |
| VaR – Historical (95%, 1-day) | $10M portfolio | $906K |
| VaR – Kupiec Backtest | 95% confidence | Accepted (p=0.09) |
| Demand Forecast – XGBoost | R² | 0.91 |
| Demand Forecast – LSTM | R² | 0.70 |
| Wind Gen Forecast | R² / MAPE | 0.50 / 34.5% |
| NG Production Direction | Accuracy / F1 | 81% / 0.83 |

**Notable negative result**: Direct price prediction across 5 energy instruments failed to beat a random walk. A two-stage demand→price ablation showed demand features *hurt* price accuracy by −1.31%. This is consistent with semi-strong market efficiency — public fundamental data is already priced in.

---

## Project Structure

```
energy-trading-ai/
├── app/
│   └── dashboard.py                  # 5-tab Streamlit dashboard
├── config/
│   └── settings.yaml                 # Central configuration
├── data/
│   ├── raw/                          # Downloaded data (gitignored)
│   ├── processed/                    # Feature matrices (gitignored)
│   └── documents/                    # PDFs for RAG (gitignored)
├── models/                           # Trained model artifacts (gitignored)
├── src/
│   ├── agents/
│   │   └── morning_briefing.py       # Agentic morning-note orchestrator
│   ├── data/
│   │   ├── price_fetcher.py          # yfinance wrapper
│   │   ├── eia_client.py             # EIA v2 API client (storage, prod)
│   │   ├── eia_generation_client.py  # EIA hourly→daily generation data
│   │   ├── fred_client.py            # FRED API client
│   │   ├── weather_client.py         # NOAA weather client
│   │   └── news_fetcher.py           # NewsAPI + RSS fetcher
│   ├── features/
│   │   ├── technical.py              # RSI, MACD, Bollinger, ATR, OBV, …
│   │   ├── fundamental.py            # Storage surprise, rig count, curve shape
│   │   ├── macro.py                  # DXY, VIX regime, yield curve
│   │   ├── seasonal.py               # Cyclical encodings, HDD/CDD seasons
│   │   └── pipeline.py               # Feature pipeline orchestrator
│   ├── models/
│   │   ├── load_forecaster.py        # XGBoost demand forecasting
│   │   ├── lstm_model.py             # LSTM demand forecasting
│   │   ├── xgboost_model.py          # XGBoost (general)
│   │   ├── transformer_model.py      # Transformer (PyTorch)
│   │   └── training.py               # Walk-forward trainer
│   ├── nlp/
│   │   ├── sentiment.py              # FinBERT sentiment analyser
│   │   ├── news_processor.py         # Article cleaning & dedup
│   │   └── sentiment_index.py        # Rolling sentiment index
│   ├── rag/
│   │   ├── document_loader.py        # PDF loader (PyMuPDF)
│   │   ├── chunker.py                # Text chunking
│   │   ├── embeddings.py             # Sentence-transformer embeddings
│   │   ├── vector_store.py           # ChromaDB / FAISS
│   │   ├── retriever.py              # Semantic retrieval
│   │   ├── llm_client.py             # OpenAI / Ollama LLM client
│   │   ├── qa_chain.py               # RAG QA chain
│   │   └── signal_generator.py       # LLM-based trading signals
│   ├── reporting/
│   │   ├── morning_note.py           # LLM morning note generator
│   │   └── visualizations.py         # Plotly chart utilities
│   └── strategy/
│       ├── storage_valuation.py      # OU, LP intrinsic, LSMC extrinsic, Greeks
│       ├── spark_spread.py           # Regional spark-spread & dispatch model
│       ├── var_model.py              # 6-method VaR, stress tests, Kupiec
│       ├── signals.py                # Trading signal generation
│       ├── position_sizing.py        # Vol-targeting, Kelly, fixed fractional
│       ├── risk_manager.py           # Risk limits & monitoring
│       └── portfolio.py              # Portfolio management
├── scripts/
│   ├── ingest_data.py                # Download price & fundamental data
│   ├── ingest_demand_data.py         # EIA demand data ingestion
│   ├── ingest_generation_data.py     # EIA generation data ingestion
│   ├── build_features.py             # Build feature matrices
│   ├── build_vector_store.py         # Ingest PDFs → ChromaDB
│   ├── train_load_model.py           # Train demand forecasting models
│   ├── train_wind_model.py           # Train wind generation model
│   ├── train_ng_production_model.py  # Train NG production direction model
│   ├── run_storage_valuation.py      # Run storage valuation engine
│   ├── run_spark_spread.py           # Run spark-spread analysis
│   ├── run_var_model.py              # Run VaR & stress testing
│   ├── generate_morning_briefing.py  # Generate agentic morning note
│   └── run_backtest.py               # Backtesting engine
├── notebooks/                        # Jupyter exploration notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_rag_pipeline.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_backtesting.ipynb
├── tests/                            # Unit tests
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## Setup

### Prerequisites
- Python 3.10+
- Git

### 1. Clone and Install

```bash
git clone https://github.com/AndrewFSee/energy-trading-ai.git
cd energy-trading-ai

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your keys
```

| Variable | Required | Registration |
|----------|----------|--------------|
| `EIA_API_KEY` | Yes | https://www.eia.gov/opendata/register.php |
| `FRED_API_KEY` | Yes | https://fred.stlouisfed.org/docs/api/api_key.html |
| `OPENAI_API_KEY` | For morning note | https://platform.openai.com/api-keys |
| `NEWS_API_KEY` | Optional | https://newsapi.org/register |
| `NOAA_API_KEY` | Optional | https://www.ncdc.noaa.gov/cdo-web/token |

> yfinance price data and RSS news feeds require no API key.

### 3. Docker (optional)

```bash
cd docker
docker compose up -d
# Dashboard → http://localhost:8501
```

---

## Quick Start

```bash
# Ingest price and fundamental data
python scripts/ingest_data.py --start 2015-01-01
python scripts/ingest_demand_data.py
python scripts/ingest_generation_data.py

# Build RAG vector store (place PDFs in data/documents/ first)
python scripts/build_vector_store.py

# Train forecasting models
python scripts/train_load_model.py
python scripts/train_wind_model.py
python scripts/train_ng_production_model.py

# Run quant models
python scripts/run_storage_valuation.py
python scripts/run_spark_spread.py
python scripts/run_var_model.py

# Generate AI morning note
python scripts/generate_morning_briefing.py

# Launch dashboard
streamlit run app/dashboard.py
```

---

## Dashboard

The Streamlit dashboard (`app/dashboard.py`) provides five tabs:

| Tab | Contents |
|-----|----------|
| **Price & Technicals** | Candlestick chart, Bollinger Bands, volume, return distribution |
| **Storage Valuation** | OU parameters, seasonal forward curve, intrinsic LP schedule, LSMC value, Greeks |
| **Spark Spreads** | 4-region KPIs, 21-day rolling spreads, seasonal heatmap, merit order, plant-type profitability, cross-region correlation |
| **VaR & Risk** | 6-method comparison (1d + 10d), confidence ladder, stress-scenario waterfall, Kupiec backtest results |
| **AI Morning Note** | GPT-4o-generated research note synthesising all model outputs with trade ideas |

Run with: `streamlit run app/dashboard.py`

---

## Data Sources

| Source | Data | API |
|--------|------|-----|
| **yfinance** | NG=F, CL=F, UNG, VIX, DXY, S&P 500 | No key needed |
| **EIA v2** | Storage, production, hourly generation (4 regions × 3 fuels) | Free key |
| **FRED** | Fed funds, CPI, GDP, industrial production, yield curve | Free key |
| **NOAA** | HDD/CDD, wind speed, temperature | Free key |
| **NewsAPI + RSS** | Energy market headlines | Free key (optional) |

---

## RAG / LLM Setup

The RAG pipeline supports three configurations:

**Local embeddings + OpenAI LLM** (recommended):
```bash
python scripts/build_vector_store.py --embedding-provider sentence-transformers
# Requires OPENAI_API_KEY for GPT-4o generation
```

**Fully local (Ollama)**:
```bash
ollama pull llama3
python scripts/generate_morning_briefing.py --llm-provider ollama
```

**Full OpenAI**:
```bash
python scripts/build_vector_store.py --embedding-provider openai
python scripts/generate_morning_briefing.py --llm-provider openai
```

---

## License

MIT — see [LICENSE](LICENSE).

---

> **Disclaimer**: This software is for educational and research purposes only. It is not financial advice. Energy markets are highly volatile and trading involves substantial risk of loss.
