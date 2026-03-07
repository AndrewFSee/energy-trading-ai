# ⚡ Energy Trading AI

[![CI](https://github.com/AndrewFSee/energy-trading-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/AndrewFSee/energy-trading-ai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **End-to-end energy trading AI system** — crude oil & natural gas price forecasting with deep learning, NLP sentiment analysis, RAG/LLM pipeline, and full algorithmic trading infrastructure.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Data Sources](#data-sources)
- [RAGLLM Setup](#ragllm-setup)
- [Recommended Books and Reports](#recommended-books-and-reports)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Energy Trading AI is a production-quality quantitative research and algorithmic trading platform for energy markets. It combines:

1. **Deep Learning Price Forecasting** — LSTM, Transformer, Temporal Fusion Transformer, XGBoost ensemble
2. **NLP Sentiment Analysis** — FinBERT-powered sentiment from energy news, aggregated into a daily trading signal
3. **RAG/LLM Pipeline** — Retrieval-Augmented Generation from energy trading PDFs and reports, powering qualitative signal generation and automated morning research notes
4. **Full Algorithmic Trading Infrastructure** — data ingestion → feature engineering → signal generation → portfolio construction → backtesting → risk management → live dashboard

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  DATA INGESTION LAYER                    │
│  yfinance │ EIA API │ FRED │ NewsAPI │ NOAA │ PDFs      │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│               FEATURE ENGINEERING                        │
│  Technical indicators │ Macro features │ Seasonal feats  │
│  Futures curve shape  │ Storage surprise │ Weather feats  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────┐  ┌──────────────────────────────────┐
│   NLP PIPELINE   │  │         RAG / LLM PIPELINE       │
│ FinBERT/News     │  │  PDFs → Chunks → VectorStore     │
│ Sentiment Index  │  │  News + Query → LLM → Context    │
│                  │  │  → Qualitative Signal + Reports   │
└────────┬─────────┘  └──────────────┬───────────────────┘
         ▼                           ▼
┌─────────────────────────────────────────────────────────┐
│              ML / DL FORECASTING ENGINE                   │
│  LSTM / Transformer / TFT / XGBoost Ensemble             │
│  Inputs: Quant features + Sentiment + LLM Context Score  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│            STRATEGY & RISK MANAGEMENT                    │
│  Signal → Position sizing → Risk limits → Execution      │
│  Sharpe, Max DD, VaR, Calmar ratio                       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              DASHBOARD & REPORTING                        │
│  Streamlit: Live forecasts, P&L, LLM morning note        │
└─────────────────────────────────────────────────────────┘
```

---

## Features

### Data Ingestion
- **yfinance**: WTI crude (`CL=F`), Brent (`BZ=F`), Natural Gas (`NG=F`), VIX (`^VIX`), DXY, S&P 500
- **EIA API**: Weekly crude oil & natural gas storage, production, imports/exports
- **FRED API**: Fed funds rate, CPI, GDP, industrial production, yield curve
- **NOAA API**: Heating/cooling degree days (HDD/CDD) for demand modelling
- **NewsAPI + RSS**: Energy market headlines from Reuters, EIA, S&P Global Platts

### Feature Engineering
- **Technical**: RSI, MACD, Bollinger Bands, ATR, Stochastic, EMA/SMA, OBV, realised volatility
- **Fundamental**: Storage surprise, rig count changes, futures curve shape (contango/backwardation)
- **Macro**: DXY correlation, S&P 500/VIX regime, yield curve slope, interest rate level
- **Seasonal**: Sin/cos cyclical encodings, heating/cooling/driving/hurricane season indicators

### NLP Sentiment
- **FinBERT** (`ProsusAI/finbert`) for financial text sentiment classification
- Daily rolling sentiment index aggregated from energy news headlines
- Sentiment momentum and regime features for ML model input

### RAG / LLM Pipeline
- **Document ingestion**: PDF loading (PyMuPDF), text chunking, embedding generation
- **Vector store**: ChromaDB (persistent) or FAISS (in-memory)
- **Retrieval**: Dense semantic search with optional cross-encoder reranking
- **Signal generation**: LLM-based qualitative BULLISH/BEARISH/NEUTRAL signal with confidence score
- **Morning notes**: Automated daily research notes combining market data + RAG context

### ML/DL Models

| Model | Type | Key Strengths |
|-------|------|---------------|
| **LSTM** | PyTorch | Sequence modelling, long-range dependencies |
| **Transformer** | PyTorch | Parallel attention, long lookback windows |
| **TFT** | pytorch-forecasting | Interpretable temporal attention |
| **XGBoost** | Gradient Boosting | Tabular features, fast, feature importance |
| **Ensemble** | Weighted Average / Stacking | Best out-of-sample performance |

### Strategy & Risk Management
- **Position sizing**: Volatility-targeting, Kelly criterion, fixed fractional
- **Risk limits**: Max drawdown halt, VaR monitoring, position concentration limits
- **ATR-based stops**: Automatic stop-loss and take-profit levels

### Backtesting
- Configurable transaction costs and slippage (basis points)
- Metrics: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Profit Factor
- Walk-forward validation to prevent lookahead bias
- Benchmark comparison (strategy vs buy-and-hold)

---

## Project Structure

```
energy-trading-ai/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── settings.yaml              # Central configuration
├── data/
│   ├── raw/                       # Downloaded data (gitignored)
│   ├── processed/                 # Feature matrices (gitignored)
│   └── documents/                 # PDFs for RAG (gitignored)
├── src/
│   ├── data/                      # Data ingestion modules
│   │   ├── price_fetcher.py       # yfinance wrapper
│   │   ├── eia_client.py          # EIA API client
│   │   ├── fred_client.py         # FRED API client
│   │   ├── weather_client.py      # NOAA weather client
│   │   └── news_fetcher.py        # NewsAPI + RSS fetcher
│   ├── features/                  # Feature engineering
│   │   ├── technical.py           # Technical indicators
│   │   ├── fundamental.py         # Fundamental features
│   │   ├── macro.py               # Macro features
│   │   ├── seasonal.py            # Seasonal features
│   │   └── pipeline.py            # Feature pipeline orchestrator
│   ├── nlp/                       # Sentiment analysis
│   │   ├── sentiment.py           # FinBERT sentiment analyser
│   │   ├── news_processor.py      # Article cleaning & dedup
│   │   └── sentiment_index.py     # Daily sentiment index
│   ├── rag/                       # RAG/LLM pipeline
│   │   ├── document_loader.py     # PDF loader
│   │   ├── chunker.py             # Text chunking
│   │   ├── embeddings.py          # Embedding generation
│   │   ├── vector_store.py        # ChromaDB/FAISS store
│   │   ├── retriever.py           # Semantic retrieval
│   │   ├── llm_client.py          # OpenAI/Ollama client
│   │   ├── qa_chain.py            # RAG QA chain
│   │   └── signal_generator.py    # LLM trading signals
│   ├── models/                    # ML/DL models
│   │   ├── base.py                # Abstract base model
│   │   ├── lstm_model.py          # LSTM (PyTorch)
│   │   ├── transformer_model.py   # Transformer (PyTorch)
│   │   ├── xgboost_model.py       # XGBoost
│   │   ├── ensemble.py            # Ensemble combiner
│   │   └── training.py            # Walk-forward trainer
│   ├── strategy/                  # Trading strategy
│   │   ├── signals.py             # Signal generation
│   │   ├── position_sizing.py     # Position sizing
│   │   ├── risk_manager.py        # Risk management
│   │   └── portfolio.py           # Portfolio management
│   ├── backtesting/               # Backtesting engine
│   │   ├── engine.py              # Core backtest engine
│   │   ├── metrics.py             # Performance metrics
│   │   └── analysis.py            # Backtest analysis
│   └── reporting/                 # Reports & visualisations
│       ├── morning_note.py        # LLM morning note generator
│       └── visualizations.py      # Plotly chart utilities
├── notebooks/                     # Jupyter exploration notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_rag_pipeline.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_backtesting.ipynb
├── scripts/                       # CLI automation scripts
│   ├── ingest_data.py
│   ├── build_features.py
│   ├── build_vector_store.py
│   ├── train_models.py
│   ├── run_backtest.py
│   └── generate_report.py
├── app/
│   └── dashboard.py               # Streamlit dashboard
├── tests/                         # Unit test suite
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── .github/
    └── workflows/
        └── ci.yml                 # GitHub Actions CI
```

---

## Setup

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone and Install

```bash
git clone https://github.com/AndrewFSee/energy-trading-ai.git
cd energy-trading-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your API keys
```

| API Key Variable | Required | Free Registration |
|-----------------|----------|-------------------|
| `EIA_API_KEY` | Yes | https://www.eia.gov/opendata/register.php |
| `FRED_API_KEY` | Yes | https://fred.stlouisfed.org/docs/api/api_key.html |
| `NEWS_API_KEY` | Recommended | https://newsapi.org/register |
| `NOAA_API_KEY` | Optional | https://www.ncdc.noaa.gov/cdo-web/token |
| `OPENAI_API_KEY` | Optional | https://platform.openai.com/api-keys |

> **Note**: yfinance price data requires no API key. RSS news feeds also require no key.

### 3. Using Docker (Recommended for Production)

```bash
cd docker
docker compose up -d
# Dashboard: http://localhost:8501
# ChromaDB: http://localhost:8000
```

---

## Quick Start

```bash
# 1. Download all data
python scripts/ingest_data.py --start 2015-01-01

# 2. Build feature matrix
python scripts/build_features.py --instrument wti

# 3. Build RAG vector store (add PDFs to data/documents/ first)
python scripts/build_vector_store.py

# 4. Train models
python scripts/train_models.py --instrument wti

# 5. Run backtest
python scripts/run_backtest.py --instrument wti --start 2020-01-01

# 6. Generate morning note
python scripts/generate_report.py --print-only

# 7. Launch dashboard
streamlit run app/dashboard.py
```

---

## Data Sources

| Source | Data Type | URL |
|--------|-----------|-----|
| yfinance | Price OHLCV | Automatic via library |
| EIA Open Data | Storage, production, trade | https://www.eia.gov/opendata/ |
| FRED | Macro indicators | https://fred.stlouisfed.org/ |
| NOAA CDO | HDD/CDD weather | https://www.ncdc.noaa.gov/cdo-web/webservices/v2 |
| NewsAPI | Energy news | https://newsapi.org/ |
| Reuters RSS | Financial news | https://feeds.reuters.com/reuters/businessNews |
| EIA RSS | Energy headlines | https://www.eia.gov/rss/todayinenergy.xml |

---

## RAG/LLM Setup

**Option A: Local embeddings + OpenAI LLM**
```bash
# Generate embeddings locally (free)
python scripts/build_vector_store.py --embedding-provider sentence-transformers
# Requires OPENAI_API_KEY in .env for LLM generation
```

**Option B: Fully local (Ollama)**
```bash
# Install Ollama: https://ollama.ai/
ollama pull llama3
# Set OLLAMA_BASE_URL in .env
python scripts/generate_report.py --llm-provider ollama
```

**Option C: Full OpenAI**
```bash
python scripts/build_vector_store.py --embedding-provider openai
python scripts/generate_report.py --llm-provider openai
```

---

## Recommended Books and Reports

Place PDFs in `data/documents/` to include them in the RAG knowledge base.

### Energy Markets and Trading (Core Domain)
1. *Energy Trading & Investing* — Davis W. Edwards
2. *The Handbook of Energy Trading* — Stefano Fiorenzani
3. *Oil 101* — Morgan Downey
4. *The Quest: Energy, Security, and the Remaking of the Modern World* — Daniel Yergin
5. *The Prize: The Epic Quest for Oil, Money & Power* — Daniel Yergin
6. *Natural Gas Trading in North America* — Fletcher J. Sturm
7. *Fundamentals of Oil & Gas Accounting* — Charlotte Wright

### Quantitative Finance and Trading
8. *Advances in Financial Machine Learning* — Marcos López de Prado
9. *Machine Learning for Asset Managers* — Marcos López de Prado
10. *Trading and Exchanges: Market Microstructure for Practitioners* — Larry Harris
11. *Quantitative Trading* — Ernest Chan
12. *Algorithmic Trading* — Ernest Chan
13. *Options, Futures, and Other Derivatives* — John C. Hull
14. *Dynamic Hedging* — Nassim Nicholas Taleb

### Machine Learning and Deep Learning
15. *Deep Learning* — Ian Goodfellow et al.
16. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* — Aurélien Géron
17. *Time Series Forecasting in Python* — Marco Peixeiro
18. *Forecasting: Principles and Practice* — Hyndman & Athanasopoulos

### Free Reports and PDFs
| Report | Frequency | URL |
|--------|-----------|-----|
| EIA Short-Term Energy Outlook | Monthly | https://www.eia.gov/outlooks/steo/ |
| EIA Annual Energy Outlook | Annual | https://www.eia.gov/outlooks/aeo/ |
| OPEC Monthly Oil Market Report | Monthly | https://www.opec.org/opec_web/en/publications/338.htm |
| OPEC World Oil Outlook | Annual | https://www.opec.org/opec_web/en/publications/340.htm |
| IEA Oil Market Report | Monthly | https://www.iea.org/topics/oil-market-report |
| BP Statistical Review of World Energy | Annual | https://www.bp.com/en/global/corporate/energy-economics/statistical-review-of-world-energy.html |

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass: `pytest tests/`
5. Lint your code: `ruff check .`
6. Submit a pull request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> **Disclaimer**: This software is for educational and research purposes only. It is not financial advice. Energy markets are highly volatile and trading involves substantial risk of loss. Past backtest performance does not guarantee future results.
