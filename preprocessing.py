import pandas as pd
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
@st.cache_data
def process_stock_data(df: pd.DataFrame, days_limit: int = None) -> pd.Series:
    """
    Clean, validate, and flatten stock data into a single-row Series with clearly ordered columns.
    Format: date_0, open_0, high_0, ..., volume_0, date_1, open_1, ...
    """
    try:
        required_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
        df_cols = set(df.columns)

        # Auto-create 'Adj Close' if missing
        if 'Adj Close' not in df_cols and 'Close' in df_cols:
            df['Adj Close'] = df['Close']
            logging.warning("'Adj Close' column missing. Auto-created from 'Close'.")

        required_cols.add('Adj Close')
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Drop missing rows and clean dates
        df_clean = df.dropna(subset=required_cols).copy()
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean.dropna(subset=['Date'], inplace=True)
        if df_clean.empty:
            raise ValueError("No valid rows after cleaning.")

        df_clean.sort_values('Date', inplace=True)
        if days_limit and days_limit > 0:
            df_clean = df_clean.tail(days_limit)

        # Define desired order
        desired_order = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        df_clean = df_clean[desired_order]

        # Standardize column names
        col_map = {
            'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
        }
        df_clean.rename(columns=col_map, inplace=True)

        # Flatten row-by-row in proper order
        flattened_data = {}
        for i, row in enumerate(df_clean.itertuples(index=False)):
            for col in col_map.values():  # order maintained
                flattened_data[f"{col}_{i}"] = getattr(row, col)

        return pd.Series(flattened_data)

    except Exception as e:
        logging.error(f"Error in process_stock_data: {e}")
        raise

import pandas as pd

def unflatten_series_to_dataframe(flattened_series: pd.Series) -> pd.DataFrame:
    """
    Converts a flattened single-row Series (like open_0, close_0, ..., open_1) into a structured multi-row DataFrame.
    """
    daily_records = {}

    for key, value in flattened_series.items():
        if '_' in key:
            feature, index = key.rsplit('_', 1)
            if index.isdigit():
                index = int(index) + 1  # Start Day Index from 1
                if index not in daily_records:
                    daily_records[index] = {}
                daily_records[index][feature] = value

    df = pd.DataFrame.from_dict(daily_records, orient='index')

    df.index.name = 'Day Index'
    df.reset_index(inplace=True)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(by='date')

    return df
