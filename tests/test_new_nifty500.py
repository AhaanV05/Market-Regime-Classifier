import pandas as pd
import joblib
from pathlib import Path
import sys
import json

# Add project scripts to path (relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'market_regime' / 'scripts'))
from quarterly_retrain import evaluate_week8_gate, infer_regimes_with_models, compute_economic_metrics, RetrainController

features_df = pd.read_csv(PROJECT_ROOT / 'market_regime' / 'features' / 'final_features_matrix.csv', parse_dates=['Date'], index_col='Date')
market_df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'market_data_historical.csv', parse_dates=['Date'], index_col='Date')

c = RetrainController({})
new_models = c._build_new_models(features_df, features_df.index, joblib.load(PROJECT_ROOT / 'market_regime' / 'models' / 'hmm_regime_models.joblib'))
preds = infer_regimes_with_models(features_df, new_models)
metrics = evaluate_week8_gate(preds, market_df)
print(json.dumps(metrics['metrics']['checks'], indent=2))
print("===")
print(compute_economic_metrics(preds, market_df))
