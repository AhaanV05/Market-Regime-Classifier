#!/usr/bin/env python3
"""
Daily ops monitor (post-inference).

Runs after `scripts/daily_inference.py` and writes:
- drift monitor result
- BOCPD result
- emergency trigger result
- retrain controller decision log

Usage:
    python scripts/daily_ops.py [--date YYYY-MM-DD]

Exit codes:
    0  -> SUCCESS
    10 -> SOFT_FAIL (e.g., no trading data yet)
    1  -> HARD_FAIL
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def resolve_base_dir() -> Path:
    script_root = Path(__file__).resolve().parent.parent
    if (script_root / "models").exists() and (script_root / "data").exists():
        return script_root
    cwd = Path.cwd().resolve()
    for candidate in [cwd, cwd.parent]:
        if (candidate / "models").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not resolve project root with models/ and data/ folders")


BASE_DIR = resolve_base_dir()
DATA_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

for p in [OUTPUT_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


class SoftFailError(Exception):
    """Non-fatal execution issue."""


def parse_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d")).normalize()


def pick_first_existing_column(df: pd.DataFrame, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"None of columns found: {candidates}")
    return None


def longest_true_streak(flags):
    streak, best = 0, 0
    for flag in flags:
        if bool(flag):
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def atomic_write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    sanitized = sanitize_for_json(payload)
    tmp.write_text(json.dumps(sanitized, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_for_json(payload)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sanitized, allow_nan=False) + "\n")


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def acquire_lock(lock_file: Path, stale_seconds: int):
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    if lock_file.exists():
        age = time.time() - lock_file.stat().st_mtime
        if age > stale_seconds:
            lock_file.unlink(missing_ok=True)
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "started_at": datetime.now().isoformat()}))
    except FileExistsError as exc:
        raise SoftFailError(f"Another daily_ops run appears active (lock: {lock_file})") from exc


def release_lock(lock_file: Path):
    try:
        lock_file.unlink(missing_ok=True)
    except Exception:
        pass


class DriftMonitor:
    """Feature drift + model-health checks."""

    def __init__(
        self,
        baseline_window=252,
        z_threshold=2.5,
        consecutive_days=5,
        flip_rate_threshold=60.0,
        occupancy_threshold=0.02,
        confidence_threshold=0.30,
    ):
        self.baseline_window = int(baseline_window)
        self.z_threshold = float(z_threshold)
        self.consecutive_days = int(consecutive_days)
        self.flip_rate_threshold = float(flip_rate_threshold)
        self.occupancy_threshold = float(occupancy_threshold)
        self.confidence_threshold = float(confidence_threshold)

    def check_feature_drift(self, feature_values, recent_window=60):
        vals = np.asarray(feature_values, dtype=float)
        vals = vals[np.isfinite(vals)]
        needed = self.baseline_window + int(recent_window)
        if len(vals) < needed:
            return {
                "drift_alert": False,
                "latest_z": np.nan,
                "max_abs_z": np.nan,
                "consecutive_breach_days": 0,
                "recent_window": int(recent_window),
                "baseline_window": int(self.baseline_window),
                "insufficient_data": True,
            }

        baseline = vals[-(self.baseline_window + recent_window) : -recent_window]
        baseline_mean = float(np.mean(baseline))
        baseline_std = float(np.std(baseline))
        baseline_std = max(baseline_std, 1e-8)

        series = pd.Series(vals)
        rolling_recent_mean = series.rolling(int(recent_window), min_periods=int(recent_window)).mean().iloc[-recent_window:]
        z_series = ((rolling_recent_mean - baseline_mean) / baseline_std).fillna(0.0).values

        breach_flags = np.abs(z_series) > self.z_threshold
        max_streak = longest_true_streak(breach_flags)
        drift_alert = bool(max_streak >= self.consecutive_days)

        return {
            "drift_alert": drift_alert,
            "latest_z": float(z_series[-1]),
            "max_abs_z": float(np.max(np.abs(z_series))),
            "consecutive_breach_days": int(max_streak),
            "recent_window": int(recent_window),
            "baseline_window": int(self.baseline_window),
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "insufficient_data": False,
        }

    def check_model_health(self, predictions: pd.DataFrame, window=60):
        if len(predictions) < 3:
            return {
                "flip_rate": np.nan,
                "min_occupancy": np.nan,
                "avg_confidence": np.nan,
                "flip_rate_alert": False,
                "occupancy_alert": False,
                "confidence_alert": False,
                "combined_col": None,
                "confidence_col": None,
                "insufficient_data": True,
            }

        recent = predictions.iloc[-int(window):].copy()
        combined_col = pick_first_existing_column(
            recent,
            ["combined_state", "combined_state_mindur", "combined_state_hysteresis"],
        )
        confidence_col = pick_first_existing_column(
            recent,
            ["combined_confidence", "confidence"],
            required=False,
        )

        if confidence_col is None:
            macro_col = pick_first_existing_column(recent, ["p_fragile", "p_fragile_smooth"], required=False)
            calm_col = pick_first_existing_column(recent, ["p_calm", "p_calm_smooth"], required=False)
            choppy_col = pick_first_existing_column(recent, ["p_choppy", "p_choppy_smooth"], required=False)
            stress_col = pick_first_existing_column(recent, ["p_stress", "p_stress_smooth"], required=False)
            if macro_col and calm_col and choppy_col and stress_col:
                p_frag = pd.to_numeric(recent[macro_col], errors="coerce").fillna(0.5).clip(0.0, 1.0)
                macro_conf = np.maximum(p_frag.values, 1.0 - p_frag.values)
                fast_mat = np.vstack(
                    [
                        pd.to_numeric(recent[calm_col], errors="coerce").fillna(1.0 / 3.0).values,
                        pd.to_numeric(recent[choppy_col], errors="coerce").fillna(1.0 / 3.0).values,
                        pd.to_numeric(recent[stress_col], errors="coerce").fillna(1.0 / 3.0).values,
                    ]
                ).T
                fast_conf = np.max(fast_mat, axis=1)
                avg_confidence = float(np.mean(np.minimum(macro_conf, fast_conf)))
            else:
                avg_confidence = np.nan
        else:
            avg_confidence = float(pd.to_numeric(recent[confidence_col], errors="coerce").mean())

        combined = recent[combined_col].astype(str).values
        flips = int((combined[1:] != combined[:-1]).sum()) if len(combined) > 1 else 0
        annualized_flip_rate = (flips / max(len(combined) - 1, 1)) * 252.0

        occupancy = recent[combined_col].value_counts(normalize=True)
        min_occupancy = float(occupancy.min()) if len(occupancy) > 0 else 0.0

        return {
            "flip_rate": float(annualized_flip_rate),
            "min_occupancy": min_occupancy,
            "avg_confidence": avg_confidence,
            "flip_rate_alert": bool(annualized_flip_rate > self.flip_rate_threshold),
            "occupancy_alert": bool(min_occupancy < self.occupancy_threshold),
            "confidence_alert": bool((not np.isnan(avg_confidence)) and (avg_confidence < self.confidence_threshold)),
            "flip_rate_threshold": float(self.flip_rate_threshold),
            "occupancy_threshold": float(self.occupancy_threshold),
            "confidence_threshold": float(self.confidence_threshold),
            "combined_col": combined_col,
            "confidence_col": confidence_col,
            "insufficient_data": False,
        }


def check_emergency_triggers(market_data: pd.DataFrame, lookback=30, crash_threshold=0.05, vix_multiplier=1.5):
    """Schema-tolerant emergency triggers: crash day and VIX spike."""
    price_col = pick_first_existing_column(market_data, ["Close", "NIFTY", "Adj Close"])
    vix_col = pick_first_existing_column(market_data, ["VIX_Close", "VIX", "India VIX"])

    frame = market_data[[price_col, vix_col]].dropna().copy()
    if len(frame) < max(int(lookback), 2):
        return {
            "crash_detected": False,
            "daily_return": np.nan,
            "vix_explosion": False,
            "current_vix": np.nan,
            "vix_baseline": np.nan,
            "emergency_retrain": False,
            "price_col": price_col,
            "vix_col": vix_col,
            "insufficient_data": True,
        }

    daily_return = float(frame[price_col].pct_change().iloc[-1])
    crash_detected = bool(abs(daily_return) >= float(crash_threshold))

    current_vix = float(frame[vix_col].iloc[-1])
    vix_baseline = float(frame[vix_col].iloc[-int(lookback):].mean())
    vix_explosion = bool(current_vix >= float(vix_multiplier) * vix_baseline)

    return {
        "crash_detected": crash_detected,
        "daily_return": daily_return * 100.0,
        "vix_explosion": vix_explosion,
        "current_vix": current_vix,
        "vix_baseline": vix_baseline,
        "emergency_retrain": bool(crash_detected or vix_explosion),
        "price_col": price_col,
        "vix_col": vix_col,
        "insufficient_data": False,
    }


def bocpd_constant_hazard(
    data,
    hazard_rate=1 / 250,
    init_mean=0.0,
    init_var=1.0,
    ewm_alpha=0.05,
    max_run_length=756,
):
    """
    Numerically stable BOCPD with constant hazard and capped run-length support.
    """
    x = np.asarray(data, dtype=float)
    t_len = len(x)
    if t_len == 0:
        return np.array([])

    max_run_length = max(int(max_run_length), 1)
    runlen_prob = np.zeros(max_run_length + 1, dtype=float)
    runlen_prob[0] = 1.0

    cp_probs = np.zeros(t_len, dtype=float)
    mean_est = float(init_mean)
    var_est = max(float(init_var), 1e-6)

    for t in range(t_len):
        obs = x[t]
        sigma = max(np.sqrt(var_est), 1e-4)

        pred_prob = np.exp(-0.5 * ((obs - mean_est) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        pred_prob = float(np.clip(pred_prob, 1e-300, 1e300))

        new_prob = np.zeros_like(runlen_prob)
        new_prob[0] = float(np.sum(runlen_prob * float(hazard_rate) * pred_prob))

        growth = runlen_prob[:-1] * (1.0 - float(hazard_rate)) * pred_prob
        new_prob[1:] += growth
        new_prob[-1] += runlen_prob[-1] * (1.0 - float(hazard_rate)) * pred_prob

        total = float(new_prob.sum())
        if total <= 1e-300:
            new_prob[:] = 0.0
            new_prob[0] = 1.0
            cp_probs[t] = 1.0
        else:
            new_prob /= total
            cp_probs[t] = float(new_prob[0])

        runlen_prob = new_prob
        mean_est = (1.0 - float(ewm_alpha)) * mean_est + float(ewm_alpha) * obs
        centered = obs - mean_est
        var_est = (1.0 - float(ewm_alpha)) * var_est + float(ewm_alpha) * centered * centered
        var_est = max(var_est, 1e-6)

    return cp_probs


def run_bocpd_monitor(
    slow_features: pd.DataFrame,
    as_of_date: pd.Timestamp,
    feature_name=None,
    hazard_rate=1 / 250,
    max_run_length=756,
    threshold_quantile=0.995,
    recent_window=10,
    min_recent_breaches=3,
):
    df = slow_features.loc[slow_features.index <= pd.Timestamp(as_of_date)].copy()
    if len(df) == 0:
        return {
            "enabled": False,
            "alert": False,
            "reason": "no_data",
            "feature": feature_name,
        }

    if feature_name is None:
        feature_name = "vix_percentile_252" if "vix_percentile_252" in df.columns else str(df.columns[0])
    if feature_name not in df.columns:
        return {
            "enabled": False,
            "alert": False,
            "reason": "feature_not_found",
            "feature": feature_name,
        }

    series = df[feature_name].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) == 0:
        return {
            "enabled": False,
            "alert": False,
            "reason": "no_finite_values",
            "feature": feature_name,
        }

    cp = bocpd_constant_hazard(
        series.values,
        hazard_rate=float(hazard_rate),
        max_run_length=int(max_run_length),
    )
    if len(cp) == 0:
        return {
            "enabled": False,
            "alert": False,
            "reason": "empty_cp_probs",
            "feature": feature_name,
        }

    cp_min = float(np.min(cp))
    cp_max = float(np.max(cp))
    cp_mean = float(np.mean(cp))
    cp_std = float(np.std(cp))
    enabled = bool(cp_std > 1e-6)

    if enabled:
        threshold = float(np.quantile(cp, float(threshold_quantile)))
        recent = np.asarray(cp[-int(recent_window):], dtype=float)
        breach_count = int((recent > threshold).sum())
        alert = bool(breach_count >= int(min_recent_breaches))
        latest_cp_prob = float(recent[-1]) if len(recent) > 0 else float(cp[-1])
    else:
        threshold = float("inf")
        breach_count = 0
        alert = False
        latest_cp_prob = float(cp[-1])

    if not enabled:
        level = "DISABLED"
    elif breach_count >= int(min_recent_breaches):
        level = "RETRAIN"
    elif breach_count >= 2:
        level = "WARN"
    elif breach_count >= 1:
        level = "WATCH"
    else:
        level = "OK"

    return {
        "enabled": enabled,
        "alert": alert,
        "level": level,
        "feature": feature_name,
        "cp_stats": {
            "min": cp_min,
            "max": cp_max,
            "mean": cp_mean,
            "std": cp_std,
        },
        "threshold": threshold,
        "threshold_quantile": float(threshold_quantile),
        "recent_window": int(recent_window),
        "min_recent_breaches": int(min_recent_breaches),
        "recent_breach_count": int(breach_count),
        "retrain_recommended": bool(level == "RETRAIN"),
        "latest_cp_prob": float(latest_cp_prob),
        "observations_used": int(len(series)),
        "last_feature_date": pd.Timestamp(series.index[-1]).strftime("%Y-%m-%d"),
    }


class RetrainDecisionEngine:
    """Build auditable retrain/no-retrain decision from monitor outputs."""

    def __init__(self, quarterly_months=None, bocpd_threshold=0.8, bocpd_enabled=True):
        self.quarterly_months = set(quarterly_months or [1, 4, 7, 10])
        self.bocpd_threshold = float(bocpd_threshold)
        self.bocpd_enabled = bool(bocpd_enabled)

    def is_first_trading_day_of_quarter(self, date_like, trading_index):
        ts = pd.Timestamp(date_like)
        if ts.month not in self.quarterly_months:
            return False
        month_days = trading_index[(trading_index.year == ts.year) & (trading_index.month == ts.month)]
        if len(month_days) == 0:
            return False
        return ts.normalize() == pd.Timestamp(month_days[0]).normalize()

    def build_decision(
        self,
        as_of_date,
        market_data: pd.DataFrame,
        predictions: pd.DataFrame,
        drift_results,
        health_status,
        emergency_status,
        bocpd_status=None,
        schema_drift_status=None,
    ):
        trading_index = market_data.index.sort_values().unique()
        trading_dates = trading_index[trading_index <= pd.Timestamp(as_of_date)]
        if len(trading_dates) == 0:
            raise SoftFailError("No trading date available on or before as_of_date")

        decision_date = pd.Timestamp(trading_dates[-1])
        quarterly = self.is_first_trading_day_of_quarter(decision_date, trading_index)

        emergency = bool(emergency_status.get("emergency_retrain", False))
        drift_features = [name for name, payload in drift_results.items() if payload.get("drift_alert", False)]
        drift_alert = len(drift_features) > 0
        schema_alert = bool((schema_drift_status or {}).get("drift_alert", False))
        health_alert = bool(
            health_status.get("flip_rate_alert", False)
            or health_status.get("occupancy_alert", False)
            or health_status.get("confidence_alert", False)
        )

        bocpd_alert = False
        bocpd_level = "DISABLED"
        latest_cp_prob = np.nan
        if bocpd_status is not None:
            latest_cp_prob = float(bocpd_status.get("latest_cp_prob", np.nan))
            bocpd_level = str(bocpd_status.get("level", "DISABLED"))
            if bool(bocpd_status.get("enabled", False)) and self.bocpd_enabled:
                if "retrain_recommended" in bocpd_status:
                    bocpd_alert = bool(bocpd_status.get("retrain_recommended", False))
                elif "alert" in bocpd_status:
                    bocpd_alert = bool(bocpd_status.get("alert", False))
                else:
                    bocpd_alert = bool(latest_cp_prob > self.bocpd_threshold)

        reasons = []
        if emergency:
            if emergency_status.get("crash_detected", False):
                reasons.append("emergency_crash")
            if emergency_status.get("vix_explosion", False):
                reasons.append("emergency_vix")
            if len(reasons) == 0:
                reasons.append("emergency")
        if bocpd_alert:
            reasons.append("bocpd_shift")
        if health_alert:
            reasons.append("health_degraded")
        if drift_alert:
            reasons.append("drift_features")
        if schema_alert:
            reasons.append("feature_schema_drift")
        if quarterly:
            reasons.append("quarterly")

        retrain = len(reasons) > 0
        primary_reason = reasons[0] if retrain else "no_trigger"

        diagnostics = {
            "decision_date": decision_date.strftime("%Y-%m-%d"),
            "retrain": retrain,
            "reason": primary_reason,
            "all_reasons": reasons,
            "quarterly": quarterly,
            "emergency": emergency_status,
            "drift_alert_features": drift_features,
            "schema_drift_alert": schema_alert,
            "health_status": health_status,
            "bocpd_latest_prob": latest_cp_prob,
            "bocpd_level": bocpd_level,
            "bocpd_alert": bocpd_alert,
            "prediction_rows": int(len(predictions)),
        }
        return retrain, primary_reason, diagnostics


def load_csv_indexed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()


def load_latest_drift_baseline(logs_dir: Path):
    candidates = sorted(logs_dir.glob("drift_baseline_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(candidates) == 0:
        return None
    try:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        payload["_source_file"] = str(candidates[0])
        return payload
    except Exception:
        return None


def derive_dynamic_health_thresholds(args, baseline_payload):
    thresholds = {
        "flip_rate_threshold": float(args.health_flip_threshold),
        "occupancy_threshold": float(args.health_occupancy_threshold),
        "confidence_threshold": float(args.health_confidence_threshold),
        "baseline_loaded": False,
        "baseline_source": None,
    }
    if baseline_payload is None:
        return thresholds

    thresholds["baseline_loaded"] = True
    thresholds["baseline_source"] = baseline_payload.get("_source_file")

    baseline_flip = baseline_payload.get("combined_flip_rate")
    if baseline_flip is not None and np.isfinite(float(baseline_flip)):
        thresholds["flip_rate_threshold"] = max(
            float(args.health_flip_threshold),
            float(baseline_flip) * 2.0,
        )

    baseline_occ = baseline_payload.get("min_occupancy")
    if baseline_occ is not None and np.isfinite(float(baseline_occ)):
        thresholds["occupancy_threshold"] = min(
            float(args.health_occupancy_threshold),
            max(0.005, float(baseline_occ) * 0.5),
        )

    baseline_conf = baseline_payload.get("avg_confidence")
    if baseline_conf is not None and np.isfinite(float(baseline_conf)):
        thresholds["confidence_threshold"] = max(
            0.15,
            min(0.60, float(baseline_conf) * 0.8),
        )

    return thresholds


def check_feature_schema_drift(
    current_feature_columns,
    models_dir: Path,
    model_bundle=None,
):
    schema_path = models_dir / "feature_schema.json"
    expected = []

    if schema_path.exists():
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            expected = list(schema.get("slow_feature_names", [])) + list(schema.get("fast_feature_names", []))
        except Exception:
            expected = []

    if len(expected) == 0 and model_bundle is not None:
        expected = list(model_bundle.get("slow_feature_names", [])) + list(model_bundle.get("fast_feature_names", []))

    expected = [str(c) for c in expected]
    current = [str(c) for c in current_feature_columns]

    if len(expected) == 0:
        return {
            "schema_available": False,
            "schema_file": str(schema_path),
            "missing_columns": [],
            "unexpected_columns": [],
            "order_drift": None,
            "drift_alert": False,
        }

    expected_set = set(expected)
    current_set = set(current)
    missing = sorted(list(expected_set - current_set))
    unexpected = sorted(list(current_set - expected_set))
    expected_present = [c for c in expected if c in current_set]
    current_expected_order = [c for c in current if c in expected_set]
    order_drift = expected_present != current_expected_order

    return {
        "schema_available": True,
        "schema_file": str(schema_path),
        "expected_count": int(len(expected)),
        "current_count": int(len(current)),
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "order_drift": bool(order_drift),
        "drift_alert": bool((len(missing) > 0) or order_drift),
    }


def audit_timeline_integrity(predictions: pd.DataFrame, market_data: pd.DataFrame, as_of_date: pd.Timestamp, lookback_days=30):
    pred_idx = pd.DatetimeIndex(predictions.index).normalize()
    duplicate_dates = pred_idx[pred_idx.duplicated()].unique().sort_values()

    trading_idx = pd.DatetimeIndex(market_data.index[market_data.index <= pd.Timestamp(as_of_date)]).normalize()
    recent_trading = trading_idx[-int(lookback_days):] if len(trading_idx) > 0 else pd.DatetimeIndex([])
    pred_recent_set = set(pred_idx[pred_idx.isin(recent_trading)])
    missing_recent = [d.strftime("%Y-%m-%d") for d in recent_trading if d not in pred_recent_set]

    return {
        "lookback_trading_days": int(len(recent_trading)),
        "missing_trading_days_last_30": int(len(missing_recent)),
        "missing_trading_days_sample": missing_recent[:10],
        "duplicate_dates_detected": bool(len(duplicate_dates) > 0),
        "duplicate_dates_count": int(len(duplicate_dates)),
        "duplicate_dates_sample": [d.strftime("%Y-%m-%d") for d in duplicate_dates[:10]],
    }


def load_latest_model_version(models_dir: Path):
    versions_path = models_dir / "model_versions.json"
    if not versions_path.exists():
        return {"available": False}
    try:
        versions_doc = json.loads(versions_path.read_text(encoding="utf-8"))
        current = next((v for v in versions_doc.get("versions", []) if v.get("current", False)), None)
        if current is None:
            return {"available": False}
        ts = pd.to_datetime(current.get("timestamp"), errors="coerce")
        days_since = None
        if pd.notna(ts):
            days_since = int((pd.Timestamp(datetime.now()) - ts).days)
        return {
            "available": True,
            "version_tag": current.get("version_tag"),
            "timestamp": current.get("timestamp"),
            "days_since_retrain": days_since,
        }
    except Exception:
        return {"available": False}


def load_latest_inference_status(logs_dir: Path):
    runlog = logs_dir / "daily_inference_runs.jsonl"
    if not runlog.exists():
        return {"available": False}
    try:
        last_line = None
        with open(runlog, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if last_line is None:
            return {"available": False}
        payload = json.loads(last_line)
        return {
            "available": True,
            "status": payload.get("status"),
            "status_code": payload.get("status_code"),
            "run_date": payload.get("run_date"),
            "asof_date": payload.get("asof_date"),
            "combined_state": payload.get("combined_state"),
        }
    except Exception:
        return {"available": False}


def write_system_health_snapshot(as_of_date: pd.Timestamp, daily_ops_payload: dict):
    model_version = load_latest_model_version(MODELS_DIR)
    inference_status = load_latest_inference_status(LOGS_DIR)
    health_payload = {
        "timestamp": datetime.now().isoformat(),
        "as_of_date": pd.Timestamp(as_of_date).strftime("%Y-%m-%d"),
        "inference": inference_status,
        "drift": {
            "alert_features": daily_ops_payload.get("drift", {}).get("alert_features", []),
            "schema_drift_alert": daily_ops_payload.get("schema_drift", {}).get("drift_alert", False),
            "timeline_missing_days_last_30": daily_ops_payload.get("timeline_integrity", {}).get(
                "missing_trading_days_last_30", None
            ),
            "timeline_duplicates": daily_ops_payload.get("timeline_integrity", {}).get("duplicate_dates_detected", None),
        },
        "bocpd": {
            "enabled": daily_ops_payload.get("bocpd", {}).get("enabled", False),
            "level": daily_ops_payload.get("bocpd", {}).get("level"),
            "alert": daily_ops_payload.get("bocpd", {}).get("alert"),
            "latest_cp_prob": daily_ops_payload.get("bocpd", {}).get("latest_cp_prob"),
        },
        "decision": daily_ops_payload.get("decision", {}),
        "model": model_version,
    }
    health_dir = OUTPUT_DIR / "health_checks"
    health_dir.mkdir(parents=True, exist_ok=True)
    dated = health_dir / f"health_check_{pd.Timestamp(as_of_date).strftime('%Y-%m-%d')}.json"
    latest = health_dir / "health_check_latest.json"
    atomic_write_json(dated, health_payload)
    atomic_write_json(latest, health_payload)
    return health_payload, dated, latest


def load_slow_features_for_ops(slow_path: Path, as_of_date: pd.Timestamp):
    """
    Preferred source: slow_features_matrix.csv.
    Fallback source: final_features_matrix.csv (restricted to model slow features if possible).
    """
    if slow_path.exists():
        slow_features = load_csv_indexed(slow_path)
        slow_features = slow_features.loc[slow_features.index <= as_of_date].copy()
        if len(slow_features) > 0:
            return slow_features, str(slow_path), "slow_features_matrix"

    final_path = FEATURES_DIR / "final_features_matrix.csv"
    if not final_path.exists():
        raise SoftFailError(f"Neither slow features nor final features found ({slow_path}, {final_path})")

    final_features = load_csv_indexed(final_path)
    final_features = final_features.loc[final_features.index <= as_of_date].copy()
    if len(final_features) == 0:
        raise SoftFailError(f"No feature rows available on/before {as_of_date.date()}")

    try:
        bundle = joblib.load(MODELS_DIR / "hmm_regime_models.joblib")
        slow_cols = [c for c in bundle.get("slow_feature_names", []) if c in final_features.columns]
        if len(slow_cols) > 0:
            return final_features[slow_cols].copy(), str(final_path), "final_features_matrix:model_slow_cols"
    except Exception:
        pass

    return final_features.copy(), str(final_path), "final_features_matrix:all_cols"


def resolve_asof_date(run_date: pd.Timestamp, market_data: pd.DataFrame) -> pd.Timestamp:
    available = market_data.index[market_data.index <= pd.Timestamp(run_date)]
    if len(available) == 0:
        raise SoftFailError(f"No market data available on/before {pd.Timestamp(run_date).date()}")
    return pd.Timestamp(available[-1]).normalize()


def parse_args():
    parser = argparse.ArgumentParser(description="Daily ops monitor after inference")
    parser.add_argument("--date", type=str, help="Run date YYYY-MM-DD (defaults to today)")
    parser.add_argument(
        "--predictions-file",
        type=str,
        default=str(FEATURES_DIR / "regime_timeline_history.csv"),
        help="Predictions timeline CSV path",
    )
    parser.add_argument(
        "--slow-features-file",
        type=str,
        default=str(FEATURES_DIR / "slow_features_matrix.csv"),
        help="Slow features CSV path",
    )
    parser.add_argument(
        "--market-file",
        type=str,
        default=str(DATA_DIR / "market_data_historical.csv"),
        help="Market data CSV path",
    )
    parser.add_argument(
        "--drift-feature-count",
        type=int,
        default=5,
        help="Number of slow features to monitor for drift (default: 5)",
    )
    parser.add_argument("--drift-recent-window", type=int, default=60)
    parser.add_argument("--drift-baseline-window", type=int, default=252)
    parser.add_argument("--drift-z-threshold", type=float, default=2.5)
    parser.add_argument("--drift-consecutive-days", type=int, default=5)
    parser.add_argument("--health-window", type=int, default=60)
    parser.add_argument("--health-flip-threshold", type=float, default=60.0)
    parser.add_argument("--health-occupancy-threshold", type=float, default=0.02)
    parser.add_argument("--health-confidence-threshold", type=float, default=0.30)
    parser.add_argument("--emergency-lookback", type=int, default=30)
    parser.add_argument("--crash-threshold", type=float, default=0.05)
    parser.add_argument("--vix-multiplier", type=float, default=1.5)
    parser.add_argument("--bocpd-feature", type=str, default=None)
    parser.add_argument("--bocpd-hazard-rate", type=float, default=1 / 250)
    parser.add_argument("--bocpd-max-run-length", type=int, default=756)
    parser.add_argument("--bocpd-threshold-quantile", type=float, default=0.995)
    parser.add_argument("--bocpd-recent-window", type=int, default=10)
    parser.add_argument("--bocpd-min-recent-breaches", type=int, default=3)
    parser.add_argument("--no-bocpd", action="store_true", help="Disable BOCPD in decision engine")
    parser.add_argument("--lock-stale-seconds", type=int, default=6 * 3600)
    return parser.parse_args()


def choose_drift_features(slow_features: pd.DataFrame, max_count: int):
    """
    Prefer structural risk features; fallback to broad heuristics, then first columns.
    """
    cols = list(slow_features.columns)
    priority = [
        "vix_percentile_252",
        "max_drawdown_252",
        "max_drawdown_252_norm",
        "max_drawdown_126_norm",
        "time_under_water_252_norm",
        "downside_vol_126_norm",
        "downside_vol_252_norm",
        "vol_of_vol_126_norm",
        "vix_relative_252",
        "rv_252",
    ]
    selected = []
    for c in priority:
        if c in cols and c not in selected:
            selected.append(c)
        if len(selected) >= max_count:
            return selected

    if len(selected) < max_count:
        tokens = ("vix", "drawdown", "downside", "time_under_water", "vol_of_vol", "rv_")
        for c in cols:
            lc = c.lower()
            if any(t in lc for t in tokens) and c not in selected:
                selected.append(c)
            if len(selected) >= max_count:
                return selected

    for c in cols:
        if c not in selected:
            selected.append(c)
        if len(selected) >= max_count:
            break
    return selected


def main():
    args = parse_args()
    run_date = parse_date(args.date) if args.date else pd.Timestamp(datetime.now()).normalize()
    lock_file = LOGS_DIR / "daily_ops.lock"
    lock_acquired = False
    start = time.time()

    try:
        acquire_lock(lock_file, stale_seconds=int(args.lock_stale_seconds))
        lock_acquired = True

        market_df = load_csv_indexed(Path(args.market_file))
        as_of_date = resolve_asof_date(run_date, market_df)

        pred_path = Path(args.predictions_file)
        if not pred_path.exists():
            fallback = OUTPUT_DIR / "walkforward_predictions.csv"
            if fallback.exists():
                pred_path = fallback
            else:
                fallback = MODELS_DIR / "regime_predictions_enhanced.csv"
                if fallback.exists():
                    pred_path = fallback
        predictions = load_csv_indexed(pred_path)
        predictions = predictions.loc[predictions.index <= as_of_date].copy()
        if len(predictions) == 0:
            raise SoftFailError(f"No prediction rows available on/before {as_of_date.date()}")

        slow_features, slow_source_path, slow_source_mode = load_slow_features_for_ops(
            slow_path=Path(args.slow_features_file),
            as_of_date=as_of_date,
        )

        final_features_path = FEATURES_DIR / "final_features_matrix.csv"
        final_features = load_csv_indexed(final_features_path).loc[lambda d: d.index <= as_of_date].copy()
        if len(final_features) == 0:
            raise SoftFailError(f"No final feature rows available on/before {as_of_date.date()}")

        try:
            model_bundle = joblib.load(MODELS_DIR / "hmm_regime_models.joblib")
        except Exception:
            model_bundle = None

        baseline_payload = load_latest_drift_baseline(LOGS_DIR)
        dynamic_thresholds = derive_dynamic_health_thresholds(args, baseline_payload)

        monitor = DriftMonitor(
            baseline_window=args.drift_baseline_window,
            z_threshold=args.drift_z_threshold,
            consecutive_days=args.drift_consecutive_days,
            flip_rate_threshold=dynamic_thresholds["flip_rate_threshold"],
            occupancy_threshold=dynamic_thresholds["occupancy_threshold"],
            confidence_threshold=dynamic_thresholds["confidence_threshold"],
        )

        drift_features = choose_drift_features(slow_features, max_count=max(1, int(args.drift_feature_count)))
        drift_results = {
            col: monitor.check_feature_drift(slow_features[col].values, recent_window=int(args.drift_recent_window))
            for col in drift_features
        }
        health_status = monitor.check_model_health(predictions, window=int(args.health_window))
        health_status["threshold_source"] = {
            "dynamic": bool(dynamic_thresholds.get("baseline_loaded", False)),
            "baseline_file": dynamic_thresholds.get("baseline_source"),
        }
        emergency_status = check_emergency_triggers(
            market_df.loc[market_df.index <= as_of_date],
            lookback=int(args.emergency_lookback),
            crash_threshold=float(args.crash_threshold),
            vix_multiplier=float(args.vix_multiplier),
        )

        schema_drift = check_feature_schema_drift(
            current_feature_columns=final_features.columns,
            models_dir=MODELS_DIR,
            model_bundle=model_bundle,
        )
        timeline_integrity = audit_timeline_integrity(
            predictions=predictions,
            market_data=market_df,
            as_of_date=as_of_date,
            lookback_days=30,
        )

        if args.no_bocpd:
            bocpd_status = {
                "enabled": False,
                "alert": False,
                "level": "DISABLED",
                "retrain_recommended": False,
                "reason": "disabled_by_flag",
            }
        else:
            bocpd_status = run_bocpd_monitor(
                slow_features=slow_features,
                as_of_date=as_of_date,
                feature_name=args.bocpd_feature,
                hazard_rate=float(args.bocpd_hazard_rate),
                max_run_length=int(args.bocpd_max_run_length),
                threshold_quantile=float(args.bocpd_threshold_quantile),
                recent_window=int(args.bocpd_recent_window),
                min_recent_breaches=int(args.bocpd_min_recent_breaches),
            )

        controller = RetrainDecisionEngine(
            quarterly_months=[1, 4, 7, 10],
            bocpd_threshold=0.8,
            bocpd_enabled=not args.no_bocpd,
        )
        should_retrain, reason, decision_diag = controller.build_decision(
            as_of_date=as_of_date,
            market_data=market_df,
            predictions=predictions,
            drift_results=drift_results,
            health_status=health_status,
            emergency_status=emergency_status,
            bocpd_status=bocpd_status,
            schema_drift_status=schema_drift,
        )

        elapsed = float(time.time() - start)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "run_date": run_date.strftime("%Y-%m-%d"),
            "as_of_date": as_of_date.strftime("%Y-%m-%d"),
            "status": "SUCCESS",
            "status_code": 0,
            "elapsed_seconds": elapsed,
            "sources": {
                "predictions_file": str(pred_path),
                "slow_features_file": slow_source_path,
                "slow_features_source_mode": slow_source_mode,
                "final_features_file": str(final_features_path),
                "market_file": str(Path(args.market_file)),
                "prediction_rows_used": int(len(predictions)),
                "slow_rows_used": int(len(slow_features)),
                "final_feature_rows_used": int(len(final_features)),
            },
            "baseline": baseline_payload if baseline_payload is not None else {"available": False},
            "dynamic_thresholds": dynamic_thresholds,
            "drift": {
                "feature_count": int(len(drift_results)),
                "alert_features": [k for k, v in drift_results.items() if v.get("drift_alert", False)],
                "results": drift_results,
            },
            "health": health_status,
            "schema_drift": schema_drift,
            "timeline_integrity": timeline_integrity,
            "emergency": emergency_status,
            "bocpd": bocpd_status,
            "decision": {
                "should_retrain": bool(should_retrain),
                "primary_reason": reason,
                "diagnostics": decision_diag,
            },
        }

        ops_dir = OUTPUT_DIR / "daily_operations"
        ops_dir.mkdir(parents=True, exist_ok=True)
        dated_out = ops_dir / f"daily_ops_{as_of_date.strftime('%Y-%m-%d')}.json"
        latest_out = ops_dir / "daily_ops_latest.json"
        atomic_write_json(dated_out, payload)
        atomic_write_json(latest_out, payload)
        health_payload, health_dated, health_latest = write_system_health_snapshot(as_of_date, payload)
        append_jsonl(LOGS_DIR / "daily_ops_runs.jsonl", payload)
        append_jsonl(LOGS_DIR / "retrain_decisions.jsonl", decision_diag)

        print(json.dumps({
            "status": payload["status"],
            "run_date": payload["run_date"],
            "as_of_date": payload["as_of_date"],
            "should_retrain": payload["decision"]["should_retrain"],
            "reason": payload["decision"]["primary_reason"],
            "drift_alert_features": payload["drift"]["alert_features"],
            "schema_drift_alert": payload["schema_drift"].get("drift_alert", False),
            "missing_trading_days_last_30": payload["timeline_integrity"].get("missing_trading_days_last_30"),
            "bocpd_alert": payload["bocpd"].get("alert", False),
            "bocpd_level": payload["bocpd"].get("level"),
            "emergency_retrain": payload["emergency"].get("emergency_retrain", False),
            "system_health_latest": str(health_latest),
        }, indent=2))
        return 0

    except SoftFailError as exc:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "run_date": run_date.strftime("%Y-%m-%d"),
            "status": "SOFT_FAIL",
            "status_code": 10,
            "error": str(exc),
            "elapsed_seconds": float(time.time() - start),
        }
        append_jsonl(LOGS_DIR / "daily_ops_runs.jsonl", payload)
        print(json.dumps(payload, indent=2))
        return 10
    except Exception as exc:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "run_date": run_date.strftime("%Y-%m-%d"),
            "status": "HARD_FAIL",
            "status_code": 1,
            "error": str(exc),
            "elapsed_seconds": float(time.time() - start),
        }
        append_jsonl(LOGS_DIR / "daily_ops_runs.jsonl", payload)
        print(json.dumps(payload, indent=2))
        return 1
    finally:
        if lock_acquired:
            release_lock(lock_file)


if __name__ == "__main__":
    raise SystemExit(main())
