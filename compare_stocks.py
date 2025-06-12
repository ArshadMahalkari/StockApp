import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from data_fetching import fetch_stock_data


def compare_two_stocks():
    st.title("📊 Pro-Level Stock Comparison")

    # Input section
    col1, col2 = st.columns(2)
    with col1:
        ticker1 = st.text_input("🎯 First Stock Ticker", value="AAPL")
    with col2:
        ticker2 = st.text_input("🎯 Second Stock Ticker", value="GOOGL")

    d1, d2 = st.columns(2)
    with d1:
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
    with d2:
        end_date = st.date_input("End Date", value=datetime.today())

    show_raw = st.checkbox("📈 Also show raw price comparison")

    if st.button("🔍 Compare Now"):
        with st.spinner("Fetching data..."):
            df1, source1 = fetch_stock_data(ticker1, start_date, end_date)
            df2, source2 = fetch_stock_data(ticker2, start_date, end_date)

        if df1.empty or df2.empty:
            st.error("❌ One or both stocks returned no data.")
            return

        # Flatten MultiIndex if needed
        if isinstance(df1.columns, pd.MultiIndex):
            df1.columns = [col[0] for col in df1.columns]
        if isinstance(df2.columns, pd.MultiIndex):
            df2.columns = [col[0] for col in df2.columns]

        df1["Date"] = pd.to_datetime(df1["Date"])
        df2["Date"] = pd.to_datetime(df2["Date"])

        # Keep only needed columns
        df1 = df1[["Date", "Close", "Volume"]].rename(columns={"Close": f"{ticker1}_Close", "Volume": f"{ticker1}_Volume"})
        df2 = df2[["Date", "Close", "Volume"]].rename(columns={"Close": f"{ticker2}_Close", "Volume": f"{ticker2}_Volume"})

        merged = pd.merge(df1, df2, on="Date", how="inner")

        # 📊 Add normalized columns
        merged[f"{ticker1}_Norm"] = merged[f"{ticker1}_Close"] / merged[f"{ticker1}_Close"].iloc[0] * 100
        merged[f"{ticker2}_Norm"] = merged[f"{ticker2}_Close"] / merged[f"{ticker2}_Close"].iloc[0] * 100

        # 📈 Normalized price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=merged["Date"], y=merged[f"{ticker1}_Norm"],
            mode="lines", name=ticker1, line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=merged["Date"], y=merged[f"{ticker2}_Norm"],
            mode="lines", name=ticker2, line=dict(color="green")
        ))
        fig.update_layout(
            title="📈 Normalized Price Comparison (Base 100)",
            xaxis_title="Date",
            yaxis_title="Price Index",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # 📉 Raw price chart (optional)
        if show_raw:
            raw = go.Figure()
            raw.add_trace(go.Scatter(x=merged["Date"], y=merged[f"{ticker1}_Close"], name=ticker1, line=dict(color="blue")))
            raw.add_trace(go.Scatter(x=merged["Date"], y=merged[f"{ticker2}_Close"], name=ticker2, line=dict(color="green")))
            raw.update_layout(
                title="💵 Actual Price Comparison",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(raw, use_container_width=True)

        # 📈 Correlation
        correlation = merged[f"{ticker1}_Close"].corr(merged[f"{ticker2}_Close"])
        st.metric("📊 Price Correlation", f"{correlation:.2f}")

        # 📘 Summary Function
        def summarize(ticker):
            col = f"{ticker}_Close"
            volume = f"{ticker}_Volume"
            returns = merged[col].pct_change().dropna()
            trend = "Uptrend" if merged[col].iloc[-1] > merged[col].iloc[0] else "Downtrend" if merged[col].iloc[-1] < merged[col].iloc[0] else "Sideways"
            return {
                "Start Price": f"${merged[col].iloc[0]:.2f}",
                "End Price": f"${merged[col].iloc[-1]:.2f}",
                "Change %": f"{((merged[col].iloc[-1] - merged[col].iloc[0]) / merged[col].iloc[0] * 100):.2f}%",
                "Highest": f"${merged[col].max():.2f}",
                "Lowest": f"${merged[col].min():.2f}",
                "Avg Close": f"${merged[col].mean():.2f}",
                "Avg Volume": f"{merged[volume].mean():,.0f}",
                "Volatility %": f"{returns.std() * 100:.2f}%",
                "Trend": trend
            }

        # 📊 Display side-by-side summaries
        st.markdown("### 📊 Stock Performance Summary")

        colA, colB = st.columns(2)
        with colA:
            st.subheader(f"📘 {ticker1}")
            stats1 = summarize(ticker1)
            for k, v in stats1.items():
                st.markdown(f"**{k}:** {v}")
            # Trend Badge
            if stats1["Trend"] == "Uptrend":
                st.success("📈 Uptrend")
            elif stats1["Trend"] == "Downtrend":
                st.error("📉 Downtrend")
            else:
                st.warning("➖ Sideways")

        with colB:
            st.subheader(f"📗 {ticker2}")
            stats2 = summarize(ticker2)
            for k, v in stats2.items():
                st.markdown(f"**{k}:** {v}")
            if stats2["Trend"] == "Uptrend":
                st.success("📈 Uptrend")
            elif stats2["Trend"] == "Downtrend":
                st.error("📉 Downtrend")
            else:
                st.warning("➖ Sideways")

        st.markdown("---")
        st.caption("📌 Note: All metrics are based on selected date range and closing price.")
