"""
SLOW Features for ML Regime Classifier
Designed for 2×3 HMM: Slow Layer (Durable/Fragile)
Window sizes: 126D (6 months), 252D (12 months)
"""

import numpy as np
import pandas as pd
from typing import Optional


def rolling_realized_volatility(
    returns: pd.Series,
    window: int = 252,
    annualize: bool = True
) -> pd.Series:
    """
    Rolling realized volatility.
    
    Args:
        returns: Log returns
        window: Rolling window size (default 252D = 1 year)
        annualize: Whether to annualize (multiply by sqrt(252))
        
    Returns:
        Rolling volatility series
    """
    rv = returns.rolling(window).std()
    if annualize:
        rv = rv * np.sqrt(252)
    return rv


def rolling_vol_of_vol(
    returns: pd.Series,
    inner_window: int = 20,
    outer_window: int = 126
) -> pd.Series:
    """
    Volatility of volatility: std of rolling volatility.
    Captures regime uncertainty.
    
    Args:
        returns: Log returns
        inner_window: Window for volatility calculation (20D = 1 month)
        outer_window: Window for vol-of-vol (126D = 6 months)
        
    Returns:
        Vol-of-vol series
    """
    # First compute rolling vol
    rv = returns.rolling(inner_window).std() * np.sqrt(252)
    
    # Then compute std of that vol
    vol_of_vol = rv.rolling(outer_window).std()
    
    return vol_of_vol


def rolling_downside_volatility(
    returns: pd.Series,
    window: int = 252,
    threshold: float = 0.0,
    annualize: bool = True
) -> pd.Series:
    """
    Downside volatility (semi-deviation below threshold).
    
    Args:
        returns: Log returns
        window: Rolling window size
        threshold: Threshold return (default 0)
        annualize: Whether to annualize
        
    Returns:
        Downside volatility series
    """
    def downside_std(x):
        downside = x[x < threshold]
        if len(downside) < 2:
            return np.nan
        return downside.std()
    
    dv = returns.rolling(window).apply(downside_std, raw=False)
    if annualize:
        dv = dv * np.sqrt(252)
    return dv


def rolling_max_drawdown(
    prices: pd.Series,
    window: int = 252
) -> pd.Series:
    """
    Rolling maximum drawdown.
    
    Args:
        prices: Price series
        window: Rolling window size
        
    Returns:
        Max drawdown series (negative values)
    """
    def max_dd(x):
        if len(x) < 2:
            return np.nan
        cum_max = np.maximum.accumulate(x)
        drawdown = (x - cum_max) / cum_max
        return drawdown.min()
    
    return prices.rolling(window).apply(max_dd, raw=False)


def time_under_water(
    prices: pd.Series,
    window: int = 252
) -> pd.Series:
    """
    Fraction of time spent below previous peak (in drawdown).
    
    Args:
        prices: Price series
        window: Rolling window size
        
    Returns:
        Time under water (0 to 1)
    """
    def calc_tuw(x):
        if len(x) < 2:
            return np.nan
        cum_max = np.maximum.accumulate(x)
        in_drawdown = (x < cum_max).astype(int)
        return in_drawdown.mean()
    
    return prices.rolling(window).apply(calc_tuw, raw=False)


def rolling_skewness(
    returns: pd.Series,
    window: int = 252
) -> pd.Series:
    """
    Rolling skewness of returns.
    Negative skew = left tail risk (Fragile regime indicator).
    
    Args:
        returns: Log returns
        window: Rolling window size
        
    Returns:
        Skewness series
    """
    return returns.rolling(window).skew()


def rolling_kurtosis(
    returns: pd.Series,
    window: int = 252
) -> pd.Series:
    """
    Rolling excess kurtosis of returns.
    High kurtosis = fat tails (Fragile regime indicator).
    
    Args:
        returns: Log returns
        window: Rolling window size
        
    Returns:
        Excess kurtosis series
    """
    return returns.rolling(window).kurt()


def build_slow_features(
    df: pd.DataFrame,
    price_col: str = 'Close',
    vix_col: str = 'VIX_Close'
) -> pd.DataFrame:
    """
    Build all SLOW features for regime detection.
    
    Args:
        df: DataFrame with OHLCV and VIX data
        price_col: Column name for closing price
        vix_col: Column name for VIX
        
    Returns:
        DataFrame with SLOW features
    """
    result = df.copy()
    
    # Compute log returns
    result['log_ret'] = np.log(result[price_col] / result[price_col].shift(1))
    
    print("Computing SLOW features...")
    
    # 6-month features (126D)
    result['rv_126'] = rolling_realized_volatility(result['log_ret'], window=126)
    result['vol_of_vol_126'] = rolling_vol_of_vol(result['log_ret'], inner_window=20, outer_window=126)
    result['downside_vol_126'] = rolling_downside_volatility(result['log_ret'], window=126)
    result['max_drawdown_126'] = rolling_max_drawdown(result[price_col], window=126)
    result['time_under_water_126'] = time_under_water(result[price_col], window=126)
    result['skewness_126'] = rolling_skewness(result['log_ret'], window=126)
    result['kurtosis_126'] = rolling_kurtosis(result['log_ret'], window=126)
    
    # 12-month features (252D)
    result['rv_252'] = rolling_realized_volatility(result['log_ret'], window=252)
    result['vol_of_vol_252'] = rolling_vol_of_vol(result['log_ret'], inner_window=20, outer_window=252)
    result['downside_vol_252'] = rolling_downside_volatility(result['log_ret'], window=252)
    result['max_drawdown_252'] = rolling_max_drawdown(result[price_col], window=252)
    result['time_under_water_252'] = time_under_water(result[price_col], window=252)
    result['skewness_252'] = rolling_skewness(result['log_ret'], window=252)
    result['kurtosis_252'] = rolling_kurtosis(result['log_ret'], window=252)
    
    # VIX features (long-term)
    result['vix_percentile_252'] = result[vix_col].rolling(252).apply(
        lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else np.nan
    )
    result['vix_ma_252'] = result[vix_col].rolling(252).mean()
    result['vix_relative_252'] = result[vix_col] / result['vix_ma_252']
    
    print(f"✓ Created {len([c for c in result.columns if c not in df.columns])} SLOW features")
    print(f"  126D features: 7 (6M window)")
    print(f"  252D features: 7 (12M window)")
    print(f"  VIX features: 3 (long-term)")
    
    return result


def get_slow_feature_names() -> list:
    """Return list of SLOW feature column names."""
    return [
        'rv_126', 'vol_of_vol_126', 'downside_vol_126', 'max_drawdown_126',
        'time_under_water_126', 'skewness_126', 'kurtosis_126',
        'rv_252', 'vol_of_vol_252', 'downside_vol_252', 'max_drawdown_252',
        'time_under_water_252', 'skewness_252', 'kurtosis_252',
        'vix_percentile_252', 'vix_ma_252', 'vix_relative_252'
    ]


if __name__ == "__main__":
    # Test on historical data
    print("="*60)
    print("SLOW Features Module - Test")
    print("="*60)
    
    df = pd.read_csv('data/processed/market_data_historical.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Input: {len(df)} rows")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Build SLOW features
    result = build_slow_features(df)
    
    # Check NaN counts
    slow_cols = get_slow_feature_names()
    print("\nNaN counts:")
    for col in slow_cols:
        nan_count = result[col].isna().sum()
        nan_pct = 100 * nan_count / len(result)
        print(f"  {col}: {nan_count} ({nan_pct:.1f}%)")
    
    # Show sample
    print("\nSample features (last 5 rows):")
    print(result[['Date'] + slow_cols].tail())
    
    # Save output
    output_path = 'features/slow_features_matrix.csv'
    result.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
