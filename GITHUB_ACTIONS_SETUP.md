# GitHub Actions Self-Hosted Runner Setup

This setup runs GitHub Actions **on your local machine** via cron, without deploying to the cloud.

## How It Works

1. **GitHub Actions cron** triggers at 2 AM daily (or manual trigger)
2. **Self-hosted runner** on your Windows PC receives the job
3. **Pipeline runs locally** - updates features, runs inference, checks for retraining
4. **Results stay on your machine** - no data leaves your computer

## One-Time Setup (15 minutes)

### Step 1: Push Code to GitHub

```powershell
cd "<YOUR_PROJECT_PATH>"

# Initialize git if not already done
git init
git add .
git commit -m "Initial commit - ML Regime Classifier"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/ml-regime-classifier.git
git branch -M main
git push -u origin main
```

### Step 2: Set Up Self-Hosted Runner

1. **Go to your GitHub repo** → Settings → Actions → Runners → New self-hosted runner

2. **Download the runner** (GitHub will show these commands):
   ```powershell
   # Create a folder for the runner
   mkdir C:\actions-runner
   cd C:\actions-runner
   
   # Download (GitHub provides the exact download URL)
   Invoke-WebRequest -Uri https://github.com/actions/runner/releases/download/v2.xxx/actions-runner-win-x64-2.xxx.zip -OutFile actions-runner-win-x64-2.xxx.zip
   
   # Extract
   Add-Type -AssemblyName System.IO.Compression.FileSystem
   [System.IO.Compression.ZipFile]::ExtractToDirectory("$PWD\actions-runner-win-x64-2.xxx.zip", "$PWD")
   ```

3. **Configure the runner** (use the token from GitHub):
   ```powershell
   .\config.cmd --url https://github.com/YOUR_USERNAME/ml-regime-classifier --token YOUR_TOKEN_FROM_GITHUB
   ```
   
   When prompted:
   - Runner group: `[Enter]` (default)
   - Runner name: `windows-local` or keep default
   - Work folder: `[Enter]` (default)
   - Run as service: `Y` (yes - runs in background)

4. **Install and start the service**:
   ```powershell
   .\svc.sh install
   .\svc.sh start
   ```

### Step 3: Verify It Works

1. **Go to GitHub** → Your repo → Actions
2. Click **"Daily Regime Classifier Update"** workflow
3. Click **"Run workflow"** → Run workflow (manual trigger)
4. Watch it execute on your local machine

## What Runs Automatically

**Every day at 2 AM UTC**, GitHub Actions triggers your PC to:

1. ✅ Update features from latest market data
2. ✅ Run inference with auto-backfill for missing dates
3. ✅ Check drift, BOCPD, model health
4. ✅ Trigger quarterly or emergency retrain if needed
5. ✅ Retrain on ALL historical data (not just rolling window)

**Your computer must be on** for the job to run. If off, GitHub will retry when runner comes online.

## View Results

After each run:

```powershell
# Check latest run log
Get-Content "logs\daily_automation_runs.jsonl" -Tail 1 | ConvertFrom-Json | Format-List

# View latest regime
Get-Content "output\regime_state_latest.json" | ConvertFrom-Json | Format-List

# Check timeline updated
Get-Content "features\regime_timeline_history.csv" -Tail 5
```

## Manual Trigger Anytime

- **From GitHub**: Actions tab → Daily Regime Classifier Update → Run workflow
- **From Local**: Double-click `RUN_PIPELINE.bat`

## Runner Management

**Check runner status:**
```powershell
cd C:\actions-runner
.\svc.sh status
```

**Stop runner:**
```powershell
.\svc.sh stop
```

**Start runner:**
```powershell
.\svc.sh start
```

**Uninstall runner:**
```powershell
.\svc.sh uninstall
.\config.cmd remove --token YOUR_TOKEN
```

## Advantages vs Local Task Scheduler

| Feature | GitHub Actions | Task Scheduler |
|---------|---------------|----------------|
| Trigger from anywhere | ✅ Yes (GitHub UI) | ❌ No |
| Run history/logs | ✅ Yes (GitHub) | ⚠️ Limited |
| Notifications on fail | ✅ Yes (email/Slack) | ❌ No |
| Cross-platform | ✅ Yes | ❌ Windows only |
| Must be on PC | ✅ Yes | ✅ Yes |

## Troubleshooting

**Runner not connecting:**
- Check Windows Firewall allows GitHub Actions
- Verify token hasn't expired (regenerate if needed)

**Job not running at scheduled time:**
- Computer was off → Will run when powered on
- Check runner status: `.\svc.sh status` in `C:\actions-runner`

**Pipeline fails:**
- Check GitHub Actions logs in repo
- Run manually: `.\RUN_PIPELINE.bat` to see full output
- Check `logs/daily_automation_runs.jsonl` for details

## Security Notes

- Runner has access to your entire machine - only use trusted workflows
- Don't expose sensitive paths or credentials in workflow files
- The runner service runs under your user account
- All data stays local - nothing uploads to GitHub unless you explicitly push

## Alternative: Task Scheduler (Simpler)

If GitHub Actions seems overkill, use the local Task Scheduler setup instead:
```powershell
powershell -ExecutionPolicy Bypass -File setup_scheduler.ps1
```

Both achieve the same result - pick whichever you prefer managing.
