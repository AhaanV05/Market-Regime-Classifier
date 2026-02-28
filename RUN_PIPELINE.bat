@echo off
echo ========================================
echo   Market Regime Classifier Pipeline
echo   Manual Run (Local Test)
echo ========================================
echo.
echo TIP: For scheduled runs, use GitHub Actions
echo      See GITHUB_ACTIONS_SETUP.md for setup
echo.
echo ========================================
echo.

REM Change to the directory where this script is located
cd /d "%~dp0"
call "venv\Scripts\activate.bat"

echo [%time%] Starting pipeline...
echo.

python scripts\daily_automation.py --build-features

echo.
if %ERRORLEVEL% EQU 0 (
    echo [%time%] ✓ Pipeline completed successfully!
    echo.
    echo Updated files:
    echo   - features\final_features_matrix.csv
    echo   - features\regime_timeline_history.csv
    echo   - output\regime_state_*.json
    echo   - output\daily_operations\daily_ops_latest.json
) else (
    echo [%time%] ✗ Pipeline failed with error code %ERRORLEVEL%
    echo Check logs in logs\ directory
)

echo.
echo Press any key to exit...
pause >nul
