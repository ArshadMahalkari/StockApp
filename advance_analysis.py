# advanced_analysis.py
# Author: Arshad Mahalkari
# Description: Performs professional-level stock analysis with indicators.

import pandas as pd
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def advanced_stock_analysis(
    csv_path: str,
    ma_short: int = 20,
    ma_long: int = 50,
    rsi_period: int = 14
) -> Optional[pd.DataFrame]:
    """
    Perform advanced stock analysis using multiple technical indicators.

    Args:
        csv_path (str): Path to stock data CSV.
        ma_short (int): Short MA window.
        ma_long (int): Long MA window.
        rsi_period (int): RSI calculation window.

    Returns:
        Optional[pd.DataFrame]: DataFrame with technical indicators.
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Detect and rename date column
        if 'Date' not in df.columns:
            for col in df.columns:
                if 'date' in col.lower() or 'unnamed' in col.lower():
                    df.rename(columns={col: 'Date'}, inplace=True)
                    break

        if 'Date' not in df.columns:
            raise ValueError("CSV must have a 'Date' column.")

        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df.dropna(subset=['Close', 'Volume'], inplace=True)

        # ➤ Moving Averages
        df['MA_Short'] = df['Close'].rolling(window=ma_short).mean()
        df['MA_Long'] = df['Close'].rolling(window=ma_long).mean()

        # ➤ RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # ➤ MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # ➤ Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']

        # ➤ Buy/Sell Signals (MA crossover logic)
        df['Signal'] = 0
        df.loc[df['MA_Short'] > df['MA_Long'], 'Signal'] = 1
        df.loc[df['MA_Short'] < df['MA_Long'], 'Signal'] = -1

        # ➤ Golden Cross / Death Cross Events
        df['Event'] = None
        cross = df['Signal'].diff()
        df.loc[cross == 2, 'Event'] = 'Golden Cross'
        df.loc[cross == -2, 'Event'] = 'Death Cross'

        # Optional: Add Buy/Sell Price column
        df['Trade_Price'] = df['Close'].where(df['Signal'].diff() != 0)

        df.reset_index(inplace=True)
        logging.info("✅ Advanced stock analysis completed successfully.")
        return df

    except Exception as e:
        logging.error(f"❌ Failed to perform advanced analysis: {e}")
        return None
