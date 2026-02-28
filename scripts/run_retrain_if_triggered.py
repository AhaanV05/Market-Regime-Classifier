#!/usr/bin/env python3
"""
Trigger quarterly retraining from daily ops decision output.

Reads: output/daily_operations/daily_ops_latest.json
If decision.should_retrain=true, runs:
    python scripts/quarterly_retrain.py --emergency --reason <reason> --as-of-date <date>
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"


def append_log(payload):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "retrain_trigger_runner.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run retrain script if daily ops requested it")
    parser.add_argument("--ops-file", type=str, default=str(OUTPUT_DIR / "daily_operations" / "daily_ops_latest.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print command but do not execute")
    return parser.parse_args()


def main():
    args = parse_args()
    ops_path = Path(args.ops_file)
    ts = datetime.now().isoformat()

    if not ops_path.exists():
        payload = {"timestamp": ts, "status": "SOFT_FAIL", "error": f"Ops file not found: {ops_path}"}
        append_log(payload)
        print(json.dumps(payload, indent=2))
        return 10

    data = json.loads(ops_path.read_text(encoding="utf-8"))
    decision = data.get("decision", {})
    should_retrain = bool(decision.get("should_retrain", False))
    reason = str(decision.get("primary_reason", "trigger"))
    as_of_date = str(data.get("as_of_date") or data.get("run_date") or datetime.now().date())

    if not should_retrain:
        payload = {
            "timestamp": ts,
            "status": "SUCCESS",
            "result": "NO_TRIGGER",
            "as_of_date": as_of_date,
            "reason": reason,
        }
        append_log(payload)
        print(json.dumps(payload, indent=2))
        return 0

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "quarterly_retrain.py"),
        "--emergency",
        "--reason",
        reason,
        "--as-of-date",
        as_of_date,
    ]

    if args.dry_run:
        payload = {
            "timestamp": ts,
            "status": "SUCCESS",
            "result": "DRY_RUN",
            "command": cmd,
        }
        append_log(payload)
        print(json.dumps(payload, indent=2))
        return 0

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    payload = {
        "timestamp": ts,
        "status": "SUCCESS" if proc.returncode == 0 else "FAILED",
        "result": "EXECUTED",
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    append_log(payload)
    print(json.dumps(payload, indent=2))
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
