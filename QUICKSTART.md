# 🚀 Quick Start Guide

## What We Built

✅ **Automatic Daily Pipeline** (`scripts/daily_automation.py`) - Runs everything  
✅ **Auto-backfill** - Detects & fills missing timeline dates  
✅ **Quarterly Retraining** - Retrains on ALL data when triggered  
✅ **FastAPI Backend** (`backend/main.py`) - API layer  
✅ **React Dashboard** (`frontend/`) - Real-time UI

## 🤖 Automation Setup

### GitHub Actions (Runs Daily at 2 AM UTC)
- ✅ Trigger from anywhere via GitHub UI
- ✅ View run history and logs in GitHub
- ✅ Email notifications on failure
- ⚠️ Your PC must be on when job runs
- **Setup:** See [GITHUB_ACTIONS_SETUP.md](GITHUB_ACTIONS_SETUP.md)

### Manual Run Anytime
- **From GitHub:** Actions tab → "Daily Regime Classifier Update" → Run workflow
- **Local command:** `python scripts/daily_automation.py --build-features`

## ✨ What Happens Automatically

Once you set up automation (Option A or B above):

| Feature | Automatic? | Notes |
|---------|------------|-------|
| Daily feature updates | ✅ Yes | Processes latest market data |
| Timeline CSV updates | ✅ Yes | `features/regime_timeline_history.csv` |
| Auto-backfill missing dates | ✅ Yes | Detects gaps, fills automatically |
| Quarterly retrain | ✅ Yes | Jan/Apr/Jul/Oct, retrains on ALL data |
| Emergency retrain | ✅ Yes | On drift/BOCPD/crash detection |
| Model validation | ✅ Yes | Rejects bad models, keeps current |

**You don't need to do anything** - the pipeline handles everything.  

## 📘 Notebooks vs Scripts - READ THIS FIRST!

**Important**: See [NOTEBOOKS_VS_PRODUCTION.md](NOTEBOOKS_VS_PRODUCTION.md) for complete explanation.

**Quick Summary:**
- ✅ **Notebook 01**: Run ONCE for initial model training
- ⚠️ **Notebooks 02-05**: OPTIONAL (research only, not needed for production)
- ❌ **Notebook 06**: DON'T USE (replaced by `daily_ingestion.py` script)
- ✅ **Scripts**: Use for automated production

After running Notebook 01 once, you never need Jupyter again!

---

## 🏃 Run Everything (5 steps)

### 0. ONE-TIME: Initial Model Training

**REQUIRED BEFORE ANYTHING ELSE:**

```bash
# Open Notebook 01
jupyter notebook notebooks/01_hmm_training.ipynb

# Run all cells (creates models/hmm_regime_models.joblib)
```

This creates:
- ✅ `models/hmm_regime_models.joblib` (HMM models)
- ✅ `features/slow_features_matrix.csv` (SLOW features)
- ✅ `features/fast_features_matrix.csv` (FAST features)
- ✅ `data/processed/market_data_historical.csv` (historical data)

**After this step, you're done with notebooks!**

### 1. Test Scripts

```bash
# Test daily inference
python scripts/daily_ingestion.py

# Test quarterly retrain
python scripts/quarterly_retrain.py
```

### 2. Start Backend

```bash
# Install backend deps (first time only)
pip install fastapi uvicorn

# Run backend
uvicorn backend.main:app --reload --port 8000
```

Backend running at: **http://localhost:8000**  
API docs: **http://localhost:8000/docs**

### 3. Start Frontend

```bash
# Navigate to frontend
cd frontend

# Install deps (first time only)
npm install

# Run dev server
npm run dev
```

Dashboard running at: **http://localhost:3000**

### 4. View Dashboard

Open browser: **http://localhost:3000**

You'll see:
- 📊 Current regime card
- 📈 Probability bars
- 📉 Timeline chart (last 1 year)
- 📋 Economic metrics table

---

## 🤖 Setup Automation

### Windows (Task Scheduler)

1. Open **Task Scheduler**
2. Create two tasks:

**Task 1: Daily Ingestion**
- Name: `Market Regime Daily`
- Trigger: Daily at 3:45 PM, Mon-Fri
- Action: `<YOUR_PROJECT_PATH>\venv\Scripts\python.exe`
- Arguments: `scripts\daily_ingestion.py`
- Start in: `<YOUR_PROJECT_PATH>`

**Task 2: Quarterly Retrain**
- Name: `Market Regime Quarterly`
- Trigger: Monthly on day 1 at 1:00 AM (Jan, Apr, Jul, Oct)
- Action: `<YOUR_PROJECT_PATH>\venv\Scripts\python.exe`
- Arguments: `scripts\quarterly_retrain.py`
- Start in: `<YOUR_PROJECT_PATH>`

### Linux/Mac (Crontab)

```bash
# Edit crontab
crontab -e

# Add these lines:
# Daily at 3:45 PM Mon-Fri
45 15 * * 1-5 cd /path/to/project && /path/to/venv/bin/python scripts/daily_ingestion.py

# Quarterly on Jan/Apr/Jul/Oct 1st at 1:00 AM
0 1 1 1,4,7,10 * cd /path/to/project && /path/to/venv/bin/python scripts/quarterly_retrain.py
```

---

## 📂 Key Files

### Outputs (Auto-generated):
- `output/regime_state_latest.json` ← Dashboard reads this
- `output/regime_timeline_history.csv` ← Historical data
- `output/ema_state.json` ← State persistence

### Models:
- `models/hmm_regime_models.joblib` ← Current production models
- `models/archive/` ← Backup versions

### Logs:
- `logs/daily_ingestion.log` ← Daily script logs
- `logs/quarterly_retrain.log` ← Retrain logs

---

## 🔍 Check Status

```bash
# API health
curl http://localhost:8000/api/health

# Latest regime
curl http://localhost:8000/api/current-regime | jq

# Check logs
tail -f logs/daily_ingestion.log
```

---

## 🎯 How It Works

```
┌──────────────────────────────────────┐
│  AUTOMATION (Cron/Task Scheduler)    │
│  • daily_ingestion.py (3:45 PM)     │ ← Runs automatically
│  • quarterly_retrain.py (Quarterly) │ ← Runs automatically
└──────────────────────────────────────┘
              ↓ writes to
┌──────────────────────────────────────┐
│  DATA FILES                          │
│  • regime_state_latest.json          │
│  • regime_timeline_history.csv       │
└──────────────────────────────────────┘
              ↑ reads from
┌──────────────────────────────────────┐
│  FASTAPI BACKEND :8000               │ ← You start manually
│  GET /api/current-regime             │
│  GET /api/timeline                   │
│  GET /api/metrics                    │
└──────────────────────────────────────┘
              ↑ fetches from
┌──────────────────────────────────────┐
│  REACT DASHBOARD :3000               │ ← You start manually
│  Real-time regime visualization      │
└──────────────────────────────────────┘
```

**After setup:**
1. Scripts run automatically (daily + quarterly)
2. You start backend + frontend manually (or setup as services)
3. Dashboard shows real-time data

---

## 🐛 Troubleshooting

### Issue: Dashboard shows error

**Check if backend is running:**
```bash
curl http://localhost:8000/api/health
```

If not running:Notebook 01 first!**

The production system needs these files from Notebook 01:
- `models/hmm_regime_models.joblib`
- `features/slow_features_matrix.csv`
- `features/fast_features_matrix.csv`
- `data/processed/market_data_historical.csv`

If missing:
```bash
jupyter notebook notebooks/01_hmm_training.ipynb
# Run all cells
```
python scripts/daily_ingestion.py
```

Check output:
```bash
cat output/regime_state_latest.json
```

### Issue: Automation not triggering

**Check scheduler setup:**
- Windows: Task Scheduler has active triggers
- Linux/Mac: `crontab -l` shows entries
- GitHub Actions: Runner service is running (`.\svc.sh status`)

**Test manually first:**
```bash
python scripts/daily_automation.py --build-features
```

---

## ✅ Checklist

- [ ] Test `daily_ingestion.py` ✓
- [ ] Test `quarterly_retrain.py` ✓
- [ ] Install backend deps (`pip install fastapi uvicorn`)
- [ ] Start backend (`uvicorn backend.main:app --reload --port 8000`)
- [ ] Install frontend deps (`cd frontend && npm install`)
- [ ] Start frontend (`npm run dev`)
- [ ] View dashboard (http://localhost:3000)
- [ ] Setup cron/Task Scheduler for automation
- [ ] Configure alerts (email/Slack/Telegram) - TODO

---

**That's it! 🎉**

Your regime classifier is now:
- ✅ Automated (daily + quarterly)
- ✅ Has an API (FastAPI)
- ✅ Has a dashboard (React)
- ✅ Production-ready
