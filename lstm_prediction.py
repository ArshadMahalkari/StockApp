# lstm_prediction.py
# Updated for Streamlit integration

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import logging

# Configuration
DATA_FOLDER = "scripts/csv_files"
MODEL_PATH = "models/lstm_model.h5"
PREDICT_DAYS = 30
SEQUENCE_LENGTH = 60

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_latest_csv(folder_path: str) -> pd.DataFrame:
    """Loads the most recent CSV file from a folder."""
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No CSV files found in folder.")
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    logging.info(f"Using file: {latest_file}")
    return pd.read_csv(os.path.join(folder_path, latest_file))


def preprocess_data(df: pd.DataFrame, column: str="Close") -> tuple:
    """Prepares data for LSTM model training."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    # Ensure the column is numeric and drop NaN values
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df.dropna(subset=[column], inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[column]])

    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i - SEQUENCE_LENGTH:i])
        y.append(scaled_data[i])

    return np.array(X), np.array(y), scaler


def build_lstm_model(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info("LSTM model built and compiled.")
    return model


def predict_future(
    df: pd.DataFrame, model: Sequential, scaler: MinMaxScaler,
    column: str = "Close", days: int = PREDICT_DAYS
) -> pd.DataFrame:
    """Predicts future stock prices."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    last_sequence = df[[column]].values[-SEQUENCE_LENGTH:]

    # ✅ Fix: wrap into DataFrame with column name
    last_sequence_df = pd.DataFrame(last_sequence, columns=[column])
    scaled_last_sequence = scaler.transform(last_sequence_df)

    future_predictions = []

    for _ in range(days):
        X_input = np.array([scaled_last_sequence])
        pred = model.predict(X_input, verbose=0)
        future_predictions.append(pred[0, 0])
        scaled_last_sequence = np.append(scaled_last_sequence[1:], [[pred[0, 0]]], axis=0)

    # Inverse scale predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Prepare future dates
    last_date = pd.to_datetime(df.index[-1]) if df.index.dtype == 'datetime64[ns]' else pd.Timestamp.today()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)

    future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=[f'Predicted_{column}'])
    logging.info(f"Predicted future {days} days of {column} prices.")
    return future_df

def predict_on_training_data(
    df: pd.DataFrame, model: Sequential, scaler: MinMaxScaler,
    column: str = "Close"
) -> pd.DataFrame:
    """Predicts on historical (training) data for comparison."""
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df.dropna(subset=[column], inplace=True)

    scaled_data = scaler.transform(df[[column]])
    X = []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i - SEQUENCE_LENGTH:i])
    X = np.array(X)

    predictions = model.predict(X, verbose=0)
    predictions = scaler.inverse_transform(predictions)

    prediction_dates = df.index[SEQUENCE_LENGTH:]
    predicted_df = pd.DataFrame(predictions, index=prediction_dates, columns=[f'Predicted_{column}'])
    return predicted_df

def plot_predictions(df: pd.DataFrame, future_df: pd.DataFrame, predicted_df: pd.DataFrame, column: str = "Close") -> None:
    """Modern and stylish Plotly chart for LSTM predictions."""
    import streamlit as st

    fig = go.Figure()

    # Actual prices
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[column],
        mode='lines',
        name='Actual Price',
        line=dict(color='#00BFFF', width=2),
        hovertemplate='📅 %{x}<br>💰 Price: %{y:.2f}<extra></extra>'
    ))

    # Predicted (past)
    fig.add_trace(go.Scatter(
        x=predicted_df.index,
        y=predicted_df[f'Predicted_{column}'],
        mode='lines',
        name='Predicted (Past)',
        line=dict(color='orange', width=2, dash='dot'),
        hovertemplate='📅 %{x}<br>📉 Predicted: %{y:.2f}<extra></extra>'
    ))

    # Predicted (future)
    fig.add_trace(go.Scatter(
        x=future_df.index,
        y=future_df[f'Predicted_{column}'],
        mode='lines+markers',
        name='Predicted (Future)',
        line=dict(color='limegreen', width=2, dash='dash'),
        marker=dict(size=4, symbol='circle'),
        hovertemplate='📅 %{x}<br>🔮 Future: %{y:.2f}<extra></extra>'
    ))

    # Modern layout
    fig.update_layout(
        title=dict(
            text=f"<b>📈 Stock Price Forecast — {column}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=24, family='Arial Black')
        ),
        plot_bgcolor='#0f1117',
        paper_bgcolor='#0f1117',
        font=dict(color='white', family='Segoe UI'),
        hovermode='x unified',
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor='#333',
            showline=True,
            linecolor='white',
            tickfont=dict(color='white'),
        ),
        yaxis=dict(
            title='Price',
            showgrid=True,
            gridcolor='#333',
            showline=True,
            linecolor='white',
            tickfont=dict(color='white'),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(t=70, b=40, l=50, r=50),
        height=650
    )

    st.subheader("✨LSTM Prediction Chart")
    st.plotly_chart(fig, use_container_width=True)


def main(csv_path, predict_column = "Close", predict_days = 30):
    """Main function to load data, train model, and plot predictions."""
    logging.info("Starting the program...")
    logging.info("Loading data...")
    if csv_path:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded CSV from: {csv_path}")
    else:
        df = load_latest_csv(DATA_FOLDER)

    # Use only the selected column for prediction
    if predict_column not in df.columns:
        raise ValueError(f"Column '{predict_column}' not found in data.")


    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        #if no date column, use the index
        df.index = pd.date_range(start=pd.Timestamp.today(), periods=len(df), freq='D')  
        df['Date'] = df.index

    #pass the column name to preprocess_data
    X, y, scaler = preprocess_data(df, column=predict_column)

    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        logging.info("Loaded existing model.")
    else:
        model = build_lstm_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=10, batch_size=32)
        model.save(MODEL_PATH)
        logging.info("Model trained and saved.")

    #Pass the selected column and days to prediction functions
    future_df = predict_future(df, model, scaler, column=predict_column, days=predict_days)
    predicted_df = predict_on_training_data(df, model, scaler, column=predict_column)

    # ✅ Fixed this line:
    plot_predictions(df, future_df, predicted_df, column=predict_column)
    logging.info("Predictions completed.")
    print("\n" + "="*40)
    print("✅ Stock Prediction Summary")
    print(f"📅 Days Predicted     : {predict_days}")
    print(f"📊 Column Predicted   : {predict_column}")
    print(f"📁 CSV Used           : {csv_path if csv_path else 'Latest CSV from folder'}")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()