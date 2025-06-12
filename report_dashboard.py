import streamlit as st
import os
import pandas as pd

def show_live_dashboard():
    st.markdown("<h2 style='text-align:center;'>📊 StockPredicta – Live Dashboard</h2>", unsafe_allow_html=True)

    # Session values
    ticker = st.session_state.get("ticker", "N/A")
    start_date = st.session_state.get("start_date")
    end_date = st.session_state.get("end_date")
    interval = st.session_state.get("interval", "N/A")
    df = st.session_state.get("data")
    csv_path = st.session_state.get("csv_path")

    st.info(f"🎯 Ticker: `{ticker}` | Interval: `{interval}`")
    if start_date and end_date:
        st.caption(f"📅 Date Range: {start_date.strftime('%b %d, %Y')} → {end_date.strftime('%b %d, %Y')}")

    # Preview data
    if df is not None:
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)

        st.subheader("📋 Processed Data Preview")
        st.dataframe(df.tail(), use_container_width=True)

        st.markdown("### 📈 Price Summary")
        col1, col2, col3 = st.columns(3)
        start_price = df["Close"].iloc[0]
        end_price = df["Close"].iloc[-1]
        change_pct = ((end_price - start_price) / start_price) * 100
        col1.metric("Start", f"${start_price:.2f}")
        col2.metric("End", f"${end_price:.2f}")
        col3.metric("Change", f"{change_pct:.2f}%", delta=f"{change_pct:.2f}%")
    else:
        st.warning("⚠️ No processed data available. Please fetch data first.")

    # ✅ Use dynamic charts (stored in session_state)
    st.markdown("### 📊 Charts from Analysis & Prediction")

    if "fig_vis" in st.session_state:
        st.plotly_chart(st.session_state["fig_vis"], use_container_width=True)

    if "fig_candle" in st.session_state:
        st.plotly_chart(st.session_state["fig_candle"], use_container_width=True)

    if "fig_pred" in st.session_state:
        st.plotly_chart(st.session_state["fig_pred"], use_container_width=True)

    # 📁 Show CSV path if prediction used it
    if csv_path:
        st.success("✅ Prediction used this CSV file:")
        st.code(csv_path)
    else:
        st.info("ℹ️ LSTM model not yet run.")

    st.caption("✅ This dashboard reflects your latest data and charts.")
