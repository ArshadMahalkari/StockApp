# visualize.py
# Author: Arshad Mahalkari
# Created: 2025-04-04
# Description: Visualization of stock data with flexible options.

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go



def load_latest_csv(folder_path: str) -> pd.DataFrame:
    """Loads the most recent CSV file from a given folder."""
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No CSV files found in folder.")
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    df = pd.read_csv(os.path.join(folder_path, latest_file))
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")


import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def calculate_technical_indicators(df):
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["BB_upper"] = df["MA20"] + 2 * df["Close"].rolling(window=20).std()
    df["BB_lower"] = df["MA20"] - 2 * df["Close"].rolling(window=20).std()

    # MA crossover signals
    df["MA_Short"] = df["Close"].rolling(window=20).mean()
    df["MA_Long"] = df["Close"].rolling(window=50).mean()
    df["Signal"] = 0
    df.loc[df["MA_Short"] > df["MA_Long"], "Signal"] = 1
    df.loc[df["MA_Short"] < df["MA_Long"], "Signal"] = -1

    return df

def visualize_stock_data(df: pd.DataFrame, columns: list = ["Close"], view: str = "daily"):
    if "Date" not in df.columns:
        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])

# 🧹 Ensure all values are numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["Close"], inplace=True)

    
    if view == "monthly":
        df = df.set_index("Date").resample("M").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).reset_index()

    df = calculate_technical_indicators(df)

    fig = go.Figure()

    # Selected columns
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=col))

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_upper"], name="BB Upper", line=dict(color="lightgray", dash="dot")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_lower"], name="BB Lower", line=dict(color="lightgray", dash="dot")))

    # Add Buy/Sell markers
    buy_signals = df[df["Signal"] == 1]
    sell_signals = df[df["Signal"] == -1]
    fig.add_trace(go.Scatter(x=buy_signals["Date"], y=buy_signals["Close"], mode='markers',
                             marker=dict(color='green', size=4), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals["Date"], y=sell_signals["Close"], mode='markers',
                             marker=dict(color='red', size=4), name='Sell Signal'))

    fig.update_layout(
        title="📈 Pro-Level Stock Visualization",
        xaxis_title="Date",
        yaxis_title="Stock Price / Indicators",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # RSI Plot
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], mode="lines", name="RSI", line=dict(color="purple")))
    fig_rsi.update_layout(title="📊 RSI Indicator", xaxis_title="Date", yaxis_title="RSI", template="plotly_white")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD Plot
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line=dict(color="blue")))
    fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["Signal_Line"], name="Signal Line", line=dict(color="orange")))
    fig_macd.update_layout(title="📉 MACD Indicator", xaxis_title="Date", yaxis_title="MACD", template="plotly_white")
    st.plotly_chart(fig_macd, use_container_width=True)



def visualize_candlestick_with_volume(df: pd.DataFrame):
    """Plots a candlestick chart with volume below."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=('Candlestick Chart', 'Volume'))

    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker_color='lightblue'), row=2, col=1)

    fig.update_layout(
        title='Stock Price Candlestick Chart with Volume',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    folder = "scripts/csv_files"
    df = load_latest_csv(folder)

    # Example usage:
    visualize_stock_data(df, columns=["Close", "Open"], view="daily")
    # visualize_stock_data(df, columns=["Close"], view="monthly")
    # visualize_candlestick_with_volume(df)
