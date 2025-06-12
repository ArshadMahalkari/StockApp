import yfinance as yf
import pandas as pd
import requests
import streamlit as st
import logging
from io import StringIO

logger = logging.getLogger(__name__)
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, start_date, end_date, alpha_vantage_api_key=None):
    """
    Fetch stock data using yfinance first; fallback to Alpha Vantage if needed.

    Returns:
        pd.DataFrame, str: Data and source used ('yfinance', 'Alpha Vantage', or 'None')
    """
    # Ensure dates are datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Step 1: Try yfinance
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if not data.empty:
            data.reset_index(inplace=True)
            logger.info(f"✅ Data fetched using yfinance for {ticker}")
            return data, "yfinance"
        else:
            raise ValueError("yfinance returned empty data")
    except Exception as e:
        logger.warning(f"⚠️ yfinance failed: {e}")

    # Step 2: Fallback to Alpha Vantage
    if not alpha_vantage_api_key:
        logger.error("❌ Alpha Vantage fallback failed: API key not provided.")
        return pd.DataFrame(), "None"

    try:
        url = (
            f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
            f"&symbol={ticker}&outputsize=full&apikey={alpha_vantage_api_key}&datatype=csv"
        )
        response = requests.get(url)
        
        if response.status_code != 200:
            raise ConnectionError(f"API request failed with status code {response.status_code}")
        
        if "Error Message" in response.text or "Note" in response.text:
            raise ValueError("Alpha Vantage returned an error or was rate-limited.")

        df = pd.read_csv(StringIO(response.text))
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter by date range
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

        df.rename(columns={
            'timestamp': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adjusted_close': 'Adj Close',
            'volume': 'Volume'
        }, inplace=True)

        # Select available columns safely
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        df = df[[col for col in expected_cols if col in df.columns]]
        df.reset_index(drop=True, inplace=True)

        if df.empty:
            raise ValueError("Alpha Vantage returned empty data after filtering.")

        logger.info(f"✅ Data fetched using Alpha Vantage for {ticker}")
        return df, "Alpha Vantage"

    except Exception as e:
        logger.error(f"❌ Alpha Vantage fetch failed: {e}")
        logger.error(f"❌ Both data sources failed for {ticker} from {start_date.date()} to {end_date.date()}")
        return pd.DataFrame(), "None"
