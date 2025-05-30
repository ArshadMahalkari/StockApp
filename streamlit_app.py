# streamlit_app.py

import streamlit as st
import os
import io
import pandas as pd
from datetime import timedelta
from data_fetching import fetch_stock_data
from preprocessing import unflatten_series_to_dataframe
from basic_analysis import basic_stock_analysis
from basic_analysis import pro_level_basic_analysis
from advance_analysis import advanced_stock_analysis
from visualize import visualize_stock_data
from lstm_prediction import main as run_lstm_prediction
from preprocessing import process_stock_data
from visualize import visualize_stock_data, visualize_candlestick_with_volume
from auth import login_ui, is_authenticated, logout
from feedback_handling import save_feedback
import streamlit as st
import pandas as pd
import os
from lstm_prediction import main  # Import the function from your script
import streamlit as st
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# 📁 Where to save and load CSVs
DATA_DIR = "scripts/csv_files"
os.makedirs(DATA_DIR, exist_ok=True)

# 🖥️ Configure app layout and title
st.set_page_config(page_title="📊 Stock Analyzer App", layout="wide")

# 🔐 Login Check
if not login_ui():
    st.stop()

# 🧠 Setup session state to store CSV path and data across pages
if "csv_path" not in st.session_state:
    st.session_state.csv_path = None
if "data" not in st.session_state:
    st.session_state.data = None

# 🚪 Sidebar UI - Cleanly Designed
# Optional: Custom CSS for spacing and smoothness
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
            padding: 2rem 1rem;
            border-right: 1px solid #e6e6e6;
        }
        .stButton>button {
            width: 100%;
        }
        .logout-btn {
            margin-top: 2rem;
            background-color: #ff4b4b;
            color: white;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 **Stock Analyzer**")
    st.markdown("Welcome, **Analyst!** 👋")
    st.markdown("---")

    # 🚀 Sleek Menu using Option Menu
    page = option_menu(
        menu_title=None,  # Hide title
        options=[
            "🏠 Home",
            "📥 Fetch / Upload Data",
            "🧹 Data Processing",
            "📊 Basic Analysis",
            "📈 Advanced Analysis",
            "📊 Visualize Data",
            "🤖 LSTM Prediction",
            "📝 Feedback",
            "🚀 About"
        ],
        icons=[
            "house", "cloud-upload", "tools",
            "bar-chart", "graph-up", "pie-chart",
            "robot", "chat-dots", "info-circle"
        ],
        default_index=0,
        menu_icon="cast",
        orientation="vertical"
    )

    st.markdown("---")
    st.markdown("⚙️ **Settings**")

    # 🔒 Logout Button
    if st.button("🔒 Logout", key="logout", help="Click to logout", use_container_width=True):
        logout()
        st.rerun()

    st.markdown("---")
    st.caption("Made with ❤️ by Arshad Mahalkari")


if page == "🏠 Home":
    st.markdown("## 📈 Welcome to the Stock Analyzer App")
    st.markdown("##### 👤 Logged in as: " + f"`{st.session_state.get('username', 'Guest')}`")

    # Highlight box
    st.success("🎯 Your all-in-one platform to analyze, visualize, and predict stock market trends using AI.")

    # Layout in columns for sleek look
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("logo2.png", width=350)
    with col2:
        st.markdown("""
        ### 🚀 What You Can Do:
        - 📥 **Fetch real-time stock data** from Yahoo Finance  
        - 🧹 **Clean and process** raw financial data  
        - 📊 **Explore insights** through basic statistical analysis  
        - 📈 **Apply Moving Averages** for trend following  
        - 🤖 **Predict future prices** using LSTM Neural Networks  

        👉 Use the navigation panel on the left to get started!
        """)

    st.markdown("---")
    st.markdown("### 🆘 Need Help?")
    st.info("""
    - 🧾 Read the full [📘 User Guide](https://stockpredicta.com/user-guide-link.com)
    - 📧 Contact [Support](mailto:support@stockpredicta.com)
    - 💡 Tip: Check out the 'About' section for tech stack & credits!
    """)

    st.markdown("---")
    st.caption("Crafted with ❤️ by Arshad Mahalkari | DYPCET'28")

# 📥 DATA FETCHING
elif page == "📥 Fetch / Upload Data":
    st.markdown("## 📥 Fetch Stock Data or Upload CSV")
    st.markdown("Easily fetch real-time stock market data or upload your own CSV for custom analysis.")

    with st.container():
        st.markdown("### 🔎 Enter Stock Details")

        # Input fields with columns
        col1, col2 = st.columns([1.5, 1])
        with col1:
            ticker = st.text_input("🎯 Stock Symbol", value="AAPL", help="e.g. AAPL for Apple Inc.")
        with col2:
            interval = st.selectbox("🕒 Interval", ["1d", "1wk", "1mo"], help="Select data frequency")

        # Date selection
        st.markdown("#### 📆 Select Date Range")
        date1, date2 = st.columns(2)
        from datetime import datetime, timedelta
        start_date = date1.date_input("Start Date", value=datetime.today() - timedelta(days=7))
        end_date = date2.date_input("End Date", value=datetime.today())

        st.markdown("---")

        # 🗂️ Upload CSV section
        st.markdown("### 📂 Or Upload Your Own CSV")
        uploaded_file = st.file_uploader("Drag and drop or browse a CSV file", type=["csv"])

        # ✨ Input Validation
        if not ticker.isalpha() or len(ticker) < 1:
            st.warning("⚠️ Please enter a valid stock ticker symbol.")
            st.stop()

        if start_date >= end_date:
            st.warning("⚠️ Start date must be before end date.")
            st.stop()

        # 🟩 Handle CSV Upload
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.markdown("### 🔍 Preview of Uploaded Data")
            st.dataframe(df.head(), use_container_width=True)

            saved_path = os.path.join(DATA_DIR, "user_uploaded.csv")
            df.to_csv(saved_path, index=False)
            st.session_state.csv_path = saved_path
            st.session_state.data = df

        # 📡 Fetch data button
        elif st.button("🔎 Fetch Data from Yahoo Finance"):
            if start_date == end_date:
                end_date += timedelta(days=1)
            try:
                with st.spinner("📡 Fetching data... please wait"):
                    df, source = fetch_stock_data(
                        ticker, start_date, end_date, 
                        alpha_vantage_api_key=st.secrets["ALPHA_VANTAGE_KEY"]
                    )

                if df.empty:
                    st.warning("⚠️ No data fetched. Try a different ticker or date range.")
                else:
                    file_name = f"{ticker}_{start_date}_{end_date}_{interval}.csv"
                    saved_path = os.path.join(DATA_DIR, file_name)
                    df.to_csv(saved_path, index=False)
                    st.session_state.csv_path = saved_path
                    st.session_state.data = df
                    st.success(f"✅ Data fetched successfully and saved as `{file_name}`")

                    st.markdown("### 🔍 Latest Fetched Records")
                    st.dataframe(df.tail(), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Failed to fetch data: {e}")

    st.markdown("---")
    st.caption("📌 Tip: Use short ticker names like `TSLA`, `GOOGL`, `MSFT` to get best results.")



# 🧹 DATA PROCESSING
elif page == "🧹 Data Processing":
    st.markdown("## 🧹 Process & Explore Stock Data")
    st.markdown("Refine, filter, and sort your fetched/uploaded stock data for better insights.")

    if st.session_state.csv_path:
        try:
            df_raw = pd.read_csv(st.session_state.csv_path)

            with st.expander("🔍 Raw Data Preview", expanded=False):
                st.dataframe(df_raw.head(), use_container_width=True)

            st.markdown("### 📅 Filter Options")
            max_days = len(df_raw)
            days_to_show = st.slider(
                "📆 Select number of recent days to display",
                min_value=5,
                max_value=max_days,
                value=30,
                help="Choose how many recent days to include in the analysis"
            )

            # Process and structure
            flattened = process_stock_data(df_raw, days_limit=days_to_show)
            structured_df = unflatten_series_to_dataframe(flattened)

            # Sorting section
            with st.container():
                st.markdown("### 🔃 Sort Your Data")
                sort_col = st.selectbox(
                    "📊 Sort by column",
                    options=structured_df.columns,
                    index=structured_df.columns.get_loc("date") if "date" in structured_df.columns else 0,
                )
                sort_order = st.radio("🔽 Select Sort Order", ["Ascending", "Descending"], horizontal=True)
                structured_df = structured_df.sort_values(
                    by=sort_col, ascending=(sort_order == "Ascending")
                )

            # Display processed data
            st.markdown("### 📋 Processed Data Table")
            st.dataframe(structured_df, use_container_width=True)

            # Download button
            csv = structured_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv,
                file_name="processed_stock_data.csv",
                mime="text/csv",
                help="Download your filtered & sorted data",
            )

        except Exception as e:
            st.error(f"❌ Failed to process and display data:\n`{e}`")
    else:
        st.warning("⚠️ No data available. Please upload or fetch stock data first.")



# 📊 BASIC ANALYSIS

elif page == "📊 Basic Analysis":
    st.title("📊 Stock Analysis Summary")

    if st.session_state.csv_path:
        view = st.radio("Choose Analysis Level:", ["Basic", "Pro-Level"], horizontal=True)

        # Helper: Prepare stats text summary
        def generate_summary(stats):
            if not stats:
                return "No summary available."
            summary = f"""
**Stock Data Summary:**

- Period: {stats.get('Start Date', 'N/A')} to {stats.get('End Date', 'N/A')}
- Price ranged from ${stats.get('Min Close', 0):.2f} to ${stats.get('Max Close', 0):.2f}, averaging ${stats.get('Mean Close', 0):.2f}.
- The trend direction was {stats.get('Trend Direction', 'N/A').lower()}.
- The stock experienced {stats.get('Positive Change Days', 0)} positive days and {stats.get('Negative Change Days', 0)} negative days.
- Total traded volume was {int(stats.get('Total Volume', 0)):,} shares.
"""
            if 'CAGR (%)' in stats:
                summary += f"- Compound Annual Growth Rate (CAGR) was {stats.get('CAGR (%)', 0):.2f}%.\n"
            if 'Sharpe Ratio' in stats:
                summary += f"- Sharpe Ratio stood at {stats.get('Sharpe Ratio', 0):.2f}, indicating risk-adjusted return.\n"
            return summary

        # Helper: Prepare DataFrame for download
        def prepare_download(stats):
            if not stats:
                return None
            df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
            return df

        # ---------------- BASIC ------------------
        if view == "Basic":
            try:
                stats = basic_stock_analysis(st.session_state.csv_path)
                if stats:
                    st.markdown("### 📌 Basic Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Start Date", stats.get('Start Date', 'N/A'))
                        st.metric("Min Close", f"${stats.get('Min Close', 0):.2f}")
                        st.metric("Avg Volume", f"{int(stats.get('Avg Volume', 0)):,}")
                    with col2:
                        st.metric("End Date", stats.get('End Date', 'N/A'))
                        st.metric("Max Close", f"${stats.get('Max Close', 0):.2f}")
                        st.metric("Positive Days", stats.get('Positive Change Days', 0))
                    with col3:
                        st.metric("Avg Close", f"${stats.get('Mean Close', 0):.2f}")
                        st.metric("Median Close", f"${stats.get('Median Close', 0):.2f}")
                        st.metric("Total Volume", f"{int(stats.get('Total Volume', 0)):,}")

                    st.markdown("### 📝 Summary")
                    st.markdown(generate_summary(stats))

                    df_download = prepare_download(stats)
                    if df_download is not None:
                        st.download_button(
                            label="⬇️ Download Basic Stats",
                            data=df_download.to_csv(index=False),
                            file_name="basic_stock_stats.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("⚠ No data found.")
            except Exception as e:
                st.error(f"❌ Failed to display basic stats: {e}")

        # ---------------- PRO-LEVEL ------------------
        elif view == "Pro-Level":
            try:
                stats = pro_level_basic_analysis(st.session_state.csv_path)
                if stats:
                    st.markdown("### 💼 Pro-Level Statistics")

                    st.markdown("#### 📈 Price Stats")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Price Range", f"${stats.get('Price Range', 0):.2f}")
                        st.metric("Trend", stats.get("Trend Direction", "N/A"))
                    with col2:
                        st.metric("Std Dev (Close)", f"${stats.get('Std Dev Close', 0):.2f}")
                        st.metric("Best Daily Return", f"{stats.get('Best Daily Return (%)', 0):.2f}%")
                    with col3:
                        st.metric("Worst Daily Return", f"{stats.get('Worst Daily Return (%)', 0):.2f}%")

                    st.markdown("#### 📉 Volume & Trend Stats")
                    col4, col5, col6 = st.columns(3)
                    with col4:
                        st.metric("Volume Spikes", stats.get("Volume Spikes (>2x Avg)", 0))
                        st.metric("Flat Days", stats.get("Flat Change Days", 0))
                    with col5:
                        st.metric("Highest Volume Day", stats.get("Highest Volume Day", "N/A"))
                        st.metric("Positive Days", stats.get("Positive Change Days", 0))
                    with col6:
                        st.metric("Negative Days", stats.get("Negative Change Days", 0))

                    st.markdown("#### 📊 Performance & Risk")
                    col7, col8, col9 = st.columns(3)
                    with col7:
                        st.metric("CAGR (%)", f"{stats.get('CAGR (%)', 0):.2f}%")
                        st.metric("Sharpe Ratio", f"{stats.get('Sharpe Ratio', 0):.2f}")
                    with col8:
                        st.metric("Max Drawdown (%)", f"{stats.get('Max Drawdown (%)', 0):.2f}%")
                        st.metric("Volatility (30D)", f"{stats.get('Rolling Volatility (30D avg)', 0):.4f}")
                    with col9:
                        st.metric("Best Month", stats.get("Best Month (Return Avg)", "N/A"))
                        st.metric("Worst Month", stats.get("Worst Month (Return Avg)", "N/A"))

                    st.markdown("### 📝 Summary")
                    st.markdown(generate_summary(stats))

                    df_download = prepare_download(stats)
                    if df_download is not None:
                        st.download_button(
                            label="⬇️ Download Pro-Level Stats",
                            data=df_download.to_csv(index=False),
                            file_name="pro_level_stock_stats.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("⚠ No stats generated.")
            except Exception as e:
                st.error(f"❌ Failed to display pro stats: {e}")
    else:
        st.warning("⚠ Please upload or fetch stock data first.")


# 📈 ADVANCED ANALYSIS
elif page == "📈 Advanced Analysis":
    st.title("📈 Advanced Analysis (Moving Averages + RSI + MACD)")

    if st.session_state.csv_path:
        try:
            # Inputs for moving averages
            ma_short = st.number_input("Short-term Moving Average Window", min_value=1, value=20)
            ma_long = st.number_input("Long-term Moving Average Window", min_value=1, value=50)

            adv_df = advanced_stock_analysis(
                st.session_state.csv_path,
                ma_short=ma_short,
                ma_long=ma_long
            )

            if adv_df is not None and not adv_df.empty:
                st.markdown("### 📊 Key Metrics Summary")
                col1, col2, col3, col4 = st.columns(4)

                # Example key metrics (adjust based on what advanced_stock_analysis returns)
                col1.metric("Latest Close Price", f"${adv_df['Close'].iloc[-1]:.2f}")
                col2.metric(f"MA {ma_short}", f"${adv_df['MA_Short'].iloc[-1]:.2f}")
                col3.metric(f"MA {ma_long}", f"${adv_df['MA_Long'].iloc[-1]:.2f}")
                col4.metric("Latest RSI", f"{adv_df['RSI'].iloc[-1]:.2f}")

                st.markdown("### 📝 Summary")
                summary_text = f"""
                - The latest closing price was **${adv_df['Close'].iloc[-1]:.2f}**.
                - Short-term moving average (MA{ma_short}) is at **${adv_df['MA_Short'].iloc[-1]:.2f}**.
                - Long-term moving average (MA{ma_long}) is at **${adv_df['MA_Long'].iloc[-1]:.2f}**.
                - The latest RSI value is **{adv_df['RSI'].iloc[-1]:.2f}**, indicating {'overbought' if adv_df['RSI'].iloc[-1]>70 else ('oversold' if adv_df['RSI'].iloc[-1]<30 else 'neutral')} conditions.
                """

                st.markdown(summary_text)

                st.download_button(
                    label="⬇️ Download Analysis Data (CSV)",
                    data=adv_df.to_csv(index=False),
                    file_name="advanced_stock_analysis.csv",
                    mime="text/csv"
                )

                st.markdown("---")

                # --- Price + MA + Signals Plot ---
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(
                    x=adv_df['Date'], y=adv_df['Close'], mode='lines', name='Close',
                    line=dict(color='blue')
                ))
                fig_price.add_trace(go.Scatter(
                    x=adv_df['Date'], y=adv_df['MA_Short'], mode='lines', name=f'MA {ma_short}',
                    line=dict(color='orange', dash='dash')
                ))
                fig_price.add_trace(go.Scatter(
                    x=adv_df['Date'], y=adv_df['MA_Long'], mode='lines', name=f'MA {ma_long}',
                    line=dict(color='green', dash='dot')
                ))

                buys = adv_df[adv_df['Event'] == 'Golden Cross']
                fig_price.add_trace(go.Scatter(
                    x=buys['Date'], y=buys['Trade_Price'], mode='markers',
                    marker=dict(symbol='triangle-up', color='green', size=14),
                    name='Buy Signal (Golden Cross)'
                ))

                sells = adv_df[adv_df['Event'] == 'Death Cross']
                fig_price.add_trace(go.Scatter(
                    x=sells['Date'], y=sells['Trade_Price'], mode='markers',
                    marker=dict(symbol='triangle-down', color='red', size=14),
                    name='Sell Signal (Death Cross)'
                ))

                fig_price.update_layout(
                    title='📈 Stock Price with Moving Averages and Buy/Sell Signals',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    height=600,
                    legend=dict(bordercolor="Black", borderwidth=1),
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(fig_price, use_container_width=True)

                # --- RSI Plot ---
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=adv_df['Date'], y=adv_df['RSI'], mode='lines', name='RSI',
                    line=dict(color='purple')
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="top left")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="bottom left")

                fig_rsi.update_layout(
                    title="⚖️ RSI (Relative Strength Index)",
                    yaxis=dict(range=[0, 100]),
                    xaxis_title="Date",
                    yaxis_title="RSI Value",
                    height=300,
                    template="plotly_white",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_rsi, use_container_width=True)

                # --- MACD Plot ---
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=adv_df['Date'], y=adv_df['MACD'], mode='lines', name='MACD',
                    line=dict(color='blue')
                ))
                fig_macd.add_trace(go.Scatter(
                    x=adv_df['Date'], y=adv_df['MACD_Signal'], mode='lines', name='Signal Line',
                    line=dict(color='orange', dash='dash')
                ))

                fig_macd.update_layout(
                    title="🔄 MACD (Moving Average Convergence Divergence)",
                    xaxis_title="Date",
                    yaxis_title="MACD Value",
                    height=300,
                    template="plotly_white",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_macd, use_container_width=True)

            else:
                st.warning("⚠ No advanced data to display.")

        except Exception as e:
            st.error(f"⚠ Failed to run advanced analysis: {e}")
    else:
        st.warning("⚠ Please upload or fetch data first.")


# 📊 VISUALIZE DATA
elif page == "📊 Visualize Data":
       st.title("📊 Visualize Stock Data")
       if st.session_state.csv_path:
           try:
               df = pd.read_csv(st.session_state.csv_path)
               st.write("📈 Data Preview:")
               st.dataframe(df.tail())

               # Select columns to visualize
               columns = st.multiselect("Select columns to visualize", options=df.columns.tolist(), default=["Close"])
               view = st.selectbox("Select view", ["daily", "monthly"])

               if st.button("Visualize"):
                   visualize_stock_data(df, columns=columns, view=view)
                   visualize_candlestick_with_volume(df)
           except Exception as e:
               st.error(f"❌ Failed to visualize data: {e}")
       else:
           st.warning("⚠ Please upload or fetch data first.")
           
# 🤖 LSTM PREDICTION
elif page == "🤖 LSTM Prediction":
    st.title("🤖 LSTM-Based Stock Price Prediction")

    if st.session_state.csv_path:
        try:
            df = pd.read_csv(st.session_state.csv_path)

            # 🧹 Clean each column by removing $ and , and converting to float
            df_cleaned = df.copy()
            for col in df.columns:
                df_cleaned[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[\$,]", "", regex=True)  # Remove $ and ,
                    .str.strip()
                )
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")

            df_cleaned.dropna(axis=0, inplace=True)

            numeric_cols = df_cleaned.select_dtypes(include=["float64", "int64"]).columns.tolist()
            if 'Date' in numeric_cols:
                numeric_cols.remove('Date')
            if not numeric_cols:
                st.error("❌ No valid numeric columns found for prediction.")
                st.stop()

            selected_column = st.selectbox("🎯 Select column to predict:", numeric_cols)
            days = st.slider("📅 How many days to predict into the future?", min_value=1, max_value=180, value=30)

            if st.button("🚀 Run Prediction"):
                valid_data = df_cleaned[selected_column].dropna()
                # if valid_data.shape[0] <= 60:
                #     st.error("❌ Not enough valid numeric data for LSTM prediction (needs > 60 rows).")
                #     st.stop()

                # Save cleaned CSV temporarily
                cleaned_csv = os.path.join(DATA_DIR, "cleaned_temp.csv")
                df_cleaned.to_csv(cleaned_csv, index=False)

                 # Store days in session state
                st.session_state.predict_days = days

                # Store cleaned CSV path in session state
                with st.spinner("🔄 Running LSTM prediction..."):
                    if st.session_state.csv_path:
                        try:
                            run_lstm_prediction(st.session_state.csv_path, predict_column=selected_column, predict_days=days)
                            st.success("✅ Prediction complete!")
                            st.markdown("✅ Stock prediction summary")
                            st.markdown(f"Predicted {selected_column} prices for the next {days} days.")
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {e}")
            else:
                st.warning("⚠ Please upload or fetch data first.")
        
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
    else:
        st.warning("⚠ Please upload or fetch data first.")



# 🚀 ABOUT PAGE
elif page == "🚀 About":
    st.title("🚀 About This App")
    
    # Intro Section
    st.markdown("""
        <div style="background-color:#1c1e26; padding:25px; border-radius:10px;">
            <h2 style="color:#00FFAA;">📊 LSTM-Based Stock Price Predictor</h2>
            <p style="color:#CCCCCC; font-size:16px;">
                This interactive tool leverages <b>deep learning (LSTM)</b> to analyze stock market data and forecast future prices.
                Designed with a focus on clarity, usability, and insight — it’s perfect for both beginners and data enthusiasts.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # 🎯 Features & Purpose
    with st.expander("🔍 What this app does", expanded=True):
        st.markdown("""
        - 📈 **Analyze** historical stock price trends  
        - 🎨 **Visualize** them with modern, interactive charts  
        - 🔮 **Predict** future prices using LSTM deep learning  
        - ☁️ **Upload your own CSVs** or use default stock data  
        - ⚙️ **Fully automated** preprocessing and training steps
        """)

    # 🛠 Technologies Used
    with st.expander("🧠 Technologies Behind the App"):
        st.markdown("""
        | Tool         | Purpose                         |
        |--------------|---------------------------------|
        | `Python`     | Core programming language       |
        | `Streamlit`  | Web interface and interaction   |
        | `TensorFlow` | Deep learning and LSTM modeling |
        | `Pandas`     | Data manipulation and cleaning  |
        | `Plotly`     | Interactive data visualization  |
        | `Matplotlib` | Classic visual plots            |
        """)

    # 👤 Developer Info
    with st.expander("👤 About the Developer"):
        st.info("""
        **Arshad Mahalkari**  
        DY Patil College of Engineering, Akurdi  
        B.Tech Computer Engineering (Batch 2028)  

        Passionate about building intelligent and user-friendly tech experiences!
        """)

    # 🚀 Project Highlights
    with st.expander("🚀 Key Features & Highlights"):
        st.success("""
        - ✅ Accepts CSV uploads or uses recent stock data
        - 🔄 Automatic data cleaning and preparation
        - 📆 Predicts future prices up to **180 days**
        - 🖼️ Interactive, clean, beginner-friendly UI
        """)

    # ⚠️ Disclaimer
    with st.expander("⚠️ Disclaimer"):
        st.warning("""
        This tool is for educational and experimental use only.  
        Always perform your own research before making financial decisions.
        """)

    # 📬 Feedback Section
    st.markdown("---")
    st.markdown("""
        <div style="text-align:center;">
            <h4>💬 Have questions, suggestions, or feedback?</h4>
            <p>Feel free to reach out — let's connect and improve together!</p>
        </div>
    """, unsafe_allow_html=True)



# 📝 FEEDBACK

# Custom CSS for modern styling
st.markdown("""
    <style>
        .feedback-container {
            background: #f9f9fc;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            max-width: 600px;
            margin: auto;
            margin-top: 30px;
            animation: fadeIn 1s ease-in-out;
        }
        .stTextInput > div > input, .stTextArea textarea {
            border-radius: 12px;
            padding: 10px 14px;
            border: 1px solid #d1d1e0;
        }
        .stTextInput > label, .stTextArea > label {
            font-weight: 600;
            color: #333;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            transition: 0.3s ease;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

if page == "📝 Feedback":
    # st.markdown('<div class="feedback-container">', unsafe_allow_html=True)

    st.markdown("## 💬 We value your feedback")
    st.markdown("Help us improve by sharing your experience. Everything you write is read with care. 🙏")

    with st.form("feedback_form"):
        name = st.text_input("👤 Your Name", placeholder="e.g., Arshad Mahalkari")
        email = st.text_input("📧 Your Email", placeholder="e.g., arshad@email.com")
        feedback = st.text_area("📝 Your Feedback", placeholder="Write your thoughts here...", height=150)

        submitted = st.form_submit_button("🚀 Submit Feedback")

        if submitted:
            if name.strip() and email.strip() and feedback.strip():
                save_feedback(name.strip(), email.strip(), feedback.strip())
                st.success("✅ Thank you! Your feedback means a lot 🙏")
                st.balloons()
            else:
                st.warning("⚠️ Please complete all fields before submitting.")

    st.markdown("</div>", unsafe_allow_html=True)
