import pandas as pd
import numpy as np
import logging
from typing import Dict, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def basic_stock_analysis(data: Union[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Performs enhanced basic analysis on stock data from a CSV file or DataFrame.

    Args:
        data (str or pd.DataFrame): Path to the stock CSV file or a DataFrame.

    Returns:
        Dict[str, float]: Dictionary with extended statistics of stock data.
    """
    try:
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        required_columns = ['Close', 'Volume', 'Date']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=required_columns, inplace=True)

        # Basic stats
        min_close = df['Close'].min()
        max_close = df['Close'].max()
        mean_close = df['Close'].mean()
        median_close = df['Close'].median()
        std_close = df['Close'].std()
        price_range = max_close - min_close

        # Percent changes
        pct_change_first_last = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
        pct_change_max_min = ((max_close - min_close) / min_close) * 100 if min_close != 0 else np.nan

        # Volume stats
        total_volume = df['Volume'].sum()
        avg_volume = df['Volume'].mean()
        days_traded = df[df['Volume'] > 0].shape[0]

        # Volume spikes: days with volume > 2*avg_volume
        volume_spikes = df[df['Volume'] > 2 * avg_volume].shape[0]

        # Trend: simple linear regression slope of Close price over time (days)
        df_sorted = df.sort_values('Date')
        x = np.arange(len(df_sorted))
        y = df_sorted['Close'].values
        slope, intercept = np.polyfit(x, y, 1)
        trend = "Upward" if slope > 0 else "Downward" if slope < 0 else "Flat"

        # Count of positive vs negative daily changes in Close price
        daily_diff = df_sorted['Close'].diff().dropna()
        positive_days = (daily_diff > 0).sum()
        negative_days = (daily_diff < 0).sum()
        flat_days = (daily_diff == 0).sum()

        # Date range
        start_date = df['Date'].min()
        end_date = df['Date'].max()

        logging.info("Enhanced basic analysis completed successfully.")

        return {
            'Start Date': str(start_date.date()),
            'End Date': str(end_date.date()),
            'Min Close': min_close,
            'Max Close': max_close,
            'Mean Close': mean_close,
            'Median Close': median_close,
            'Std Dev Close': std_close,
            'Price Range': price_range,
            'Pct Change (First to Last)': pct_change_first_last,
            'Pct Change (Max to Min)': pct_change_max_min,
            'Total Volume': total_volume,
            'Avg Volume': avg_volume,
            'Days Traded': days_traded,
            'Volume Spikes (>2x Avg)': volume_spikes,
            'Trend Direction': trend,
            'Positive Change Days': int(positive_days),
            'Negative Change Days': int(negative_days),
            'Flat Change Days': int(flat_days)
        }

    except Exception as e:
        logging.error(f"Failed to perform basic analysis: {e}")
        return {}
    
def pro_level_basic_analysis(data: Union[str, pd.DataFrame]) -> Dict[str, Union[str, float]]:
    try:
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        required_columns = ['Close', 'Volume', 'Date']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=required_columns, inplace=True)
        df.sort_values('Date', inplace=True)

        # Daily returns
        df['Daily Return'] = df['Close'].pct_change()

        # Rolling volatility
        df['Rolling Volatility (30D)'] = df['Daily Return'].rolling(window=30).std()

        # Max drawdown
        cumulative_max = df['Close'].cummax()
        drawdown = (df['Close'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100

        # CAGR
        days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
        cagr = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (365.0 / days) - 1) * 100

        # Sharpe Ratio (assumes 0% risk-free rate)
        sharpe_ratio = df['Daily Return'].mean() / df['Daily Return'].std() * np.sqrt(252)

        # Monthly return summary
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_returns = df.groupby('Month')['Daily Return'].mean() * 100
        best_month = monthly_returns.idxmax()
        worst_month = monthly_returns.idxmin()

        # Day with highest volume
        highest_volume_day = df.loc[df['Volume'].idxmax(), 'Date']

        # Best & worst single-day returns
        best_daily_return = df['Daily Return'].max() * 100
        worst_daily_return = df['Daily Return'].min() * 100

        # Use your earlier basic stats
        from copy import deepcopy
        base_stats = basic_stock_analysis(df)
        advanced_stats = {
            'CAGR (%)': cagr,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Rolling Volatility (30D avg)': df['Rolling Volatility (30D)'].mean(),
            'Best Daily Return (%)': best_daily_return,
            'Worst Daily Return (%)': worst_daily_return,
            'Best Month (Return Avg)': str(best_month),
            'Worst Month (Return Avg)': str(worst_month),
            'Highest Volume Day': str(highest_volume_day.date()),
        }

        logging.info("✅ Pro-level basic analysis completed successfully.")
        return {**base_stats, **advanced_stats}

    except Exception as e:
        logging.error(f"❌ Failed to perform pro-level basic analysis: {e}")
        return {}
