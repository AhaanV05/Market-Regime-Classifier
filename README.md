# ML Regime Classifier

**A production-ready 2×3 Hierarchical Hidden Markov Model for market regime detection using NIFTY 50 and India VIX data.**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()
[![Project](https://img.shields.io/badge/Project-Complete-success.svg)]()

> **Project Status: ✅ PRODUCTION READY**  
> Complete ML regime detection system with full automation, web dashboard, and comprehensive documentation.

## Overview

This project implements a sophisticated regime detection system for Indian equity markets using a **2×3 Hierarchical HMM**:

- **SLOW Layer**: Durable vs Fragile regimes (long-term structural market conditions)
- **FAST Layer**: Calm vs Choppy vs Stress regimes (short-term market dynamics)

The system processes 15 years of NIFTY 50 and India VIX data through a deterministic feature engineering pipeline, generating 37 normalized features for regime classification.

## Architecture

```
Data Ingestion → Feature Engineering → HMM Training → Regime Prediction
     ↓                    ↓                  ↓               ↓
  NIFTY 50         SLOW Features      Slow HMM       Durable/Fragile
  India VIX        FAST Features      Fast HMM       Calm/Choppy/Stress
                   Normalization      Cascade         → 6 States
```

### Hierarchical Regime Structure

```
├─ DURABLE (Low structural risk)
│  ├─ Calm: Low vol, positive returns
│  ├─ Choppy: Moderate vol, mixed returns
│  └─ Stress: Short-term shocks in stable regime
│
└─ FRAGILE (High structural risk)
   ├─ Calm: Deceptive stability before crisis
   ├─ Choppy: Erratic behavior, regime uncertainty
   └─ Stress: Full crisis mode
```

## Features

### Data Pipeline
- ✅ **15 years** of historical data (2010-2026): 3,681 trading days
- ✅ **Data quality checks**: Outlier detection, missing data handling
- ✅ **NSE trading calendar**: Dynamic API-based holiday fetching (2011+) with 2010 fallback
- ✅ **Intelligent caching**: API responses cached locally to avoid redundant calls
- ✅ **Deterministic pipeline**: Reproducible feature generation

### Feature Engineering
- ✅ **37 features** engineered from OHLCV + VIX
  - 17 SLOW features (6M/12M windows)
  - 20 FAST features (1D-3M windows)
- ✅ **Rolling z-score normalization** (252D SLOW, 60D FAST)
- ✅ **Zero data leakage**: Validated with shift-by-1 test
- ✅ **Outlier clipping**: ±5σ for numerical stability

### Testing
- ✅ **Leakage tests**: Shift-by-1 validation
- ✅ **Rolling window integrity**: Correct window sizes
- ✅ **Determinism**: Reproducible results
- ✅ **Stationarity**: Z-score properties verified

## Installation

### Prerequisites
- Python 3.13
- Windows/Linux/macOS

### Setup

```powershell
# Clone repository
git clone <repo-url>
cd "ML Regime Classifier"

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
& "venv\Scripts\Activate.ps1"

# Or activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Ingestion

Fetch 15 years of NIFTY 50 and India VIX data from Yahoo Finance:

```powershell
python data/ingestion/fetch_nse.py --years 15 --output data/processed/market_data_historical.csv
```

**Output**: `data/processed/market_data_historical.csv` (3,681 rows)

**NSE Trading Calendar**:
The calendar automatically fetches holidays from NSE API (https://www.nseindia.com/api/holiday-master) for years 2011+:
- Fetches CM (Cash Market) trading holidays dynamically
- Caches API responses in `data/ingestion/.cache/` to avoid redundant calls
- Uses hardcoded fallback for 2010 (API limitation)
- Validates weekend exclusion and holiday alignment
- ~19 holidays per year on average

### 2. Data Quality Checks

Run outlier detection:
```powershell
python data/quality/outlier_detection.py
```

Run missing data analysis:
```powershell
python data/quality/missing_handler.py
```

### 3. Feature Engineering

Run complete feature pipeline:
```powershell
python features/feature_store.py
```

**Output**: `features/final_features_matrix.csv` (3,120 rows × 37 features) ✅ **ML-READY**

**Pipeline steps**:
1. Load raw data (3,681 days)
2. Build SLOW features (17 features, 252D windows) → `slow_features_matrix.csv`
3. Build FAST features (20 features, 60D windows) → `fast_features_matrix.csv`
4. Merge on date (37 total features)
5. Normalize (rolling z-scores: 252D SLOW, 60D FAST)
6. Clip outliers (±5σ)
7. Drop incomplete rows (561 rows = 15.2% dropout) → `final_features_matrix.csv`

**Note**: The pipeline creates 3 matrices:
- `slow_features_matrix.csv` - SLOW features only (intermediate)
- `fast_features_matrix.csv` - FAST features only (intermediate)
- `final_features_matrix.csv` - Complete normalized matrix (**use this for ML**)

### 4. Validation

Run test suite:
```powershell
python tests/test_features.py
```

Tests:
- ✅ Data leakage (shift-by-1)
- ✅ Rolling window integrity
- ✅ Deterministic reproducibility
- ✅ Z-score normalization properties

## Project Structure

```
ML Regime Classifier/
├── data/
│   ├── ingestion/
│   │   ├── fetch_nse.py          # Download NIFTY/VIX from Yahoo Finance
│   │   └── nse_calendar.py       # NSE trading calendar (2010-2026)
│   ├── quality/
│   │   ├── outlier_detection.py  # Detect return outliers, volume spikes
│   │   └── missing_handler.py    # Handle missing data
│   └── processed/
│       └── market_data_historical.csv  # 15 years OHLCV + VIX
│
├── features/
│   ├── slow_features.py          # 17 SLOW features (6M/12M)
│   ├── fast_features.py          # 20 FAST features (1D-3M)
│   ├── normalization.py          # Rolling z-score normalization
│   ├── feature_store.py          # Complete pipeline orchestration
│   └── final_features_matrix.csv # 3,120 rows × 37 normalized features
│
├── tests/
│   └── test_features.py          # Feature validation tests
│
├── docs/
│   ├── FEATURES.md               # Feature documentation
│   └── [MODEL_SPECIFICATION.md]  # HMM architecture
│
├── PROJECT_PLAN.md               # Project documentation
├── regime_system_spec.md         # System specification
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Feature Documentation

See [docs/FEATURES.md](docs/FEATURES.md) for comprehensive documentation of all 37 features, including:
- Mathematical formulas
- Window sizes
- Regime interpretation
- Normalization strategy

### Example Features

| Feature | Window | Formula | Regime Signal |
|---------|--------|---------|---------------|
| `rv_252_norm` | 12M | Rolling volatility (252D) | High = Fragile |
| `max_drawdown_126_norm` | 6M | Max peak-to-trough | Large = Fragile |
| `vix_shock_norm` | 3M | (VIX - MA) / σ | >2σ = Stress |
| `ret_1_norm` | 1D | Daily log return | Large neg = Stress |
| `skewness_60_norm` | 3M | Return asymmetry | Negative = Stress |

## Data Quality

### Market Data (3,681 days)
- **Date range**: 2010-12-27 to 2026-01-21
- **Columns**: Date, Open, High, Low, Close, Volume, VIX_Close
- **Missing data**: 0%
- **Outliers detected**: 
  - 6 extreme returns (|z| > 5)
  - 5 volume spikes (>10x average)
  - 0 stuck prices

### Final Features (3,120 days)
- **Date range**: 2013-04-16 to 2026-01-21
- **Features**: 37 normalized (0 mean, 1 std)
- **Dropout**: 15.2% (561 rows due to 252D rolling window)
- **Completeness**: 100% (all features present)
- **Inf values**: 0

## System Capabilities

### ✅ Data Foundation
- [x] Data ingestion (15 years NIFTY/VIX from Yahoo Finance)
- [x] Data quality pipeline with outlier detection
- [x] NSE trading calendar (313 holidays across 17 years)
- [x] 37 features engineered (17 SLOW + 20 FAST)
- [x] Rolling z-score normalization pipeline
- [x] Comprehensive feature validation tests
- [x] Complete documentation

### ✅ HMM Training & Inference
- [x] 2×3 Hierarchical HMM implementation (hmmlearn)
- [x] SLOW model (Durable/Fragile regimes)
- [x] FAST models conditioned on SLOW states
- [x] Cascade inference pipeline
- [x] Model validation and backtesting
- [x] Walk-forward validation (rolling 3-year train / 6-month test)

### ✅ Production Automation
- [x] Daily inference pipeline with auto-backfill
- [x] Quarterly retraining scheduler
- [x] Emergency retrain triggers (crash, VIX spike, drift, BOCPD)
- [x] Model validation gates and rollback capability
- [x] EOD data fetching automation
- [x] <10 minute end-to-end latency

### ✅ Monitoring & Evaluation
- [x] BOCPD (Bayesian Online Changepoint Detection)
- [x] Feature drift monitoring (KS test, PSI)
- [x] Model health checks
- [x] Stability metrics (flip rates, durations, entropy)
- [x] Economic metrics (returns, volatility, Sharpe by regime)
- [x] Interactive HTML evaluation reports

### ✅ Web Dashboard
- [x] FastAPI backend with REST API
- [x] React frontend with real-time visualization
- [x] Current regime display with probabilities
- [x] Historical timeline charts
- [x] Economic metrics table
- [x] GitHub Actions integration (self-hosted runner)

**🎉 PRODUCTION READY** - Complete system with full automation!

**Documentation**:
- **Setup**: [QUICKSTART.md](QUICKSTART.md) | [GitHub Actions Setup](GITHUB_ACTIONS_SETUP.md)
- **Technical**: See [docs/](docs/) folder for detailed specifications

## Performance

### Feature Engineering
- **Processing time**: ~10 seconds for 15 years
- **Memory usage**: <500 MB
- **Output size**: 3,120 rows × 37 features = ~1 MB CSV

### Data Validation
- **Test suite**: 8 tests, all passing
- **Runtime**: ~15 seconds
- **Coverage**: Leakage, windows, determinism, normalization

## Contributing

This is a research project. For questions or collaboration:
1. Review [PROJECT_PLAN.md](PROJECT_PLAN.md) for scope
2. Check [docs/FEATURES.md](docs/FEATURES.md) for feature details
3. Run tests before submitting changes

## License

MIT License - See LICENSE file for details

## References

### Academic
1. **Hidden Markov Models**: Rabiner (1989) - "A Tutorial on Hidden Markov Models"
2. **Regime Switching**: Hamilton (1989) - "A New Approach to the Economic Analysis of Nonstationary Time Series"
3. **Financial Volatility**: Andersen & Bollerslev (1998) - "Answering the Skeptics"

### Data Sources
- **NIFTY 50**: Yahoo Finance (^NSEI)
- **India VIX**: Yahoo Finance (^INDIAVIX)
- **NSE Holidays**: National Stock Exchange of India

### Python Libraries
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Preprocessing
- **hmmlearn**: Hidden Markov Models
- **yfinance**: Yahoo Finance API

---

**Status**: ✅ Production Ready  
**License**: MIT
