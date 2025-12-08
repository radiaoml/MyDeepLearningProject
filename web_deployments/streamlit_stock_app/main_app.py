# ================================================================
# Streamlit App: Stock Price Prediction (LSTM + Plotly Interactive)
# Modern Professional Interface with Enhanced Layout
# ================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
import requests
import os
from datetime import datetime

# ---------------------------------------------------------------
# Streamlit Page Settings
# ---------------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .info-box {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------
st.markdown('<h1 class="main-header">🤖 AI-Powered Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced LSTM Neural Network for S&P 500 Stock Forecasting</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# Fetch S&P 500 Tickers from Wikipedia
# ---------------------------------------------------------------
@st.cache_data
def fetch_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error("❌ Failed to fetch S&P 500 tickers.")
        st.stop()
    df = pd.read_html(response.text)[0]
    tickers = df["Symbol"].tolist()
    return sorted(tickers)

tickers = fetch_sp500_symbols()

# ---------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/stocks.png", width=80)
    st.title("⚙️ Configuration Panel")
    st.markdown("---")
    
    st.subheader("📊 Stock Selection")
    symbol = st.selectbox(
        "Choose a stock symbol",
        tickers,
        index=tickers.index("AAPL"),
        help="Select from S&P 500 companies"
    )
    
    st.markdown("---")
    st.subheader("📅 Historical Data Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            pd.to_datetime("2015-01-01"),
            help="Beginning of training data"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            pd.to_datetime("2023-12-31"),
            help="End of training data"
        )
    
    st.markdown("---")
    st.subheader("🔮 Prediction Horizon")
    
    prediction_type = st.radio(
        "Select prediction timeframe:",
        ["Quick (Days)", "Medium (Weeks)", "Long (Months)", "Extended (Years)"],
        help="Choose your prediction timeframe"
    )
    
    if prediction_type == "Quick (Days)":
        days = st.slider("Number of Days", 1, 90, 30)
        weeks, months, years = 0, 0, 0
    elif prediction_type == "Medium (Weeks)":
        weeks = st.slider("Number of Weeks", 1, 52, 4)
        days, months, years = 0, 0, 0
    elif prediction_type == "Long (Months)":
        months = st.slider("Number of Months", 1, 24, 3)
        days, weeks, years = 0, 0, 0
    else:
        years = st.slider("Number of Years", 1, 5, 1)
        days, weeks, months = 0, 0, 0
    
    st.markdown("---")
    predict_button = st.button("🚀 GENERATE PREDICTION", type="primary")
    
    st.markdown("---")
    st.caption("💡 **Tip:** First prediction trains the model (5-10 min). Subsequent predictions are instant!")

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------
def calculate_prediction_days(days, weeks, months, years):
    return days + (weeks * 7) + (months * 30) + (years * 365)

def fetch_stock_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, interval="1d",
                     auto_adjust=False, actions=False, progress=False)
    
    if df.empty:
        st.warning(f"⚠️ Primary download empty for {symbol}. Trying fallback method...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval="1d", auto_adjust=False)
    
    if df.empty:
        st.error(f"❌ No data found for symbol: {symbol}.")
        st.stop()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip().replace(" ", "_") for col in df.columns]
    
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df.rename(columns={"Adj Close": "Close"}, inplace=True)
        elif any("Close" in col for col in df.columns):
            close_col = [c for c in df.columns if "Close" in c][0]
            df.rename(columns={close_col: "Close"}, inplace=True)
        else:
            st.error(f"❌ No 'Close' column found for {symbol}.")
            st.stop()
    
    for col in ["Open", "High", "Low", "Volume"]:
        if col not in df.columns:
            df[col] = df["Close"]
    
    df.dropna(inplace=True)
    return df

def load_or_create_lstm_model(symbol, X_train, y_train):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(script_dir, f"{symbol}.h5")
    
    if os.path.exists(model_filename):
        st.success(f"✅ Loading existing model: {symbol}.h5")
        return load_model(model_filename)
    
    st.warning(f"⚙️ Training new model for {symbol}... This may take a few minutes.")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(50):
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        progress_bar.progress((epoch + 1) / 50)
        status_text.text(f"Training Progress: {epoch + 1}/50 epochs")
    
    model.save(model_filename)
    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ Model trained and saved as {symbol}.h5")
    return model

# ---------------------------------------------------------------
# Main Prediction Logic
# ---------------------------------------------------------------
if predict_button:
    with st.spinner(f"🔄 Fetching data for {symbol}..."):
        data = fetch_stock_data(symbol, start_date, end_date)
        close_column = [c for c in data.columns if "Close" in c][0]
    
    if data.empty:
        st.error("⚠️ No data available for selected stock/date range.")
        st.stop()
    
    # Display current stock info
    current_price = float(data[close_column].iloc[-1])
    price_change = float(data[close_column].iloc[-1] - data[close_column].iloc[-2])
    price_change_pct = (price_change / data[close_column].iloc[-2]) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Stock Symbol", symbol)
    with col2:
        st.metric("💰 Current Price", f"${current_price:.2f}", f"{price_change_pct:+.2f}%")
    with col3:
        st.metric("📈 Data Points", len(data))
    with col4:
        st.metric("📅 Last Updated", data.index[-1].strftime("%Y-%m-%d"))
    
    st.markdown("---")
    
    # Prepare data
    dataset = data[[close_column]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    training_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:training_size]
    
    X_train, y_train = [], []
    for i in range(100, len(train_data)):
        X_train.append(train_data[i-100:i, 0])
        y_train.append(train_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = load_or_create_lstm_model(symbol, X_train, y_train)
    
    # Historical Prediction
    test_data = scaled_data[training_size-100:]
    X_test = [test_data[i-100:i, 0] for i in range(100, len(test_data))]
    X_test = np.array(X_test).reshape(-1, 100, 1)
    predictions = scaler.inverse_transform(model.predict(X_test))
    
    # Future Prediction
    future_days = calculate_prediction_days(days, weeks, months, years)
    last_100 = scaled_data[-100:].tolist()
    future_values = []
    
    with st.spinner(f"🔮 Generating {future_days}-day forecast..."):
        for _ in range(future_days):
            x_input = np.array(last_100[-100:]).reshape(1, 100, 1)
            pred_scaled = model.predict(x_input, verbose=0)[0][0]
            pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
            future_values.append(pred_real)
            pred_rescaled = scaler.transform([[pred_real]])[0][0]
            last_100.append([pred_rescaled])
    
    future_values_rescaled = np.array(future_values).reshape(-1, 1)
    future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1),
                                 periods=future_days, freq="D")
    
    # Display prediction results
    st.success("✅ Prediction Complete!")
    
    # Key Metrics
    predicted_price = future_values_rescaled[-1][0]
    price_diff = predicted_price - current_price
    price_diff_pct = (price_diff / current_price) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "🎯 Predicted Price",
            f"${predicted_price:.2f}",
            f"{price_diff_pct:+.2f}%",
            delta_color="normal"
        )
    with col2:
        st.metric("📊 Price Change", f"${abs(price_diff):.2f}", 
                 "Increase" if price_diff > 0 else "Decrease")
    with col3:
        st.metric("📅 Forecast Period", f"{future_days} days")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Interactive Chart", "📊 Data Analysis", "📋 Detailed Forecast"])
    
    with tab1:
        st.subheader("📈 Stock Price Prediction Visualization")
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[close_column],
            mode='lines',
            name='Historical Price',
            line=dict(color='#3b82f6', width=2),
            hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Future predictions
        fig.add_trace(go.Scatter(
            x=future_index,
            y=future_values_rescaled.flatten(),
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='#10b981', width=3, dash='dash'),
            marker=dict(size=6, color='#10b981'),
            hovertemplate='<b>Date</b>: %{x}<br><b>Predicted</b>: $%{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{symbol} Stock Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            height=600,
            template='plotly_white',
            font=dict(size=12),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Statistical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Historical Data Summary")
            summary_stats = data[close_column].describe()
            st.dataframe(summary_stats, use_container_width=True)
        
        with col2:
            st.markdown("### Prediction Summary")
            pred_df = pd.DataFrame({
                'Metric': ['Min Price', 'Max Price', 'Mean Price', 'Std Dev'],
                'Value': [
                    f"${future_values_rescaled.min():.2f}",
                    f"${future_values_rescaled.max():.2f}",
                    f"${future_values_rescaled.mean():.2f}",
                    f"${future_values_rescaled.std():.2f}"
                ]
            })
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("📋 Detailed Daily Forecast")
        
        forecast_df = pd.DataFrame({
            "Date": future_index,
            "Predicted Price": future_values_rescaled.flatten(),
            "Day": range(1, future_days + 1)
        })
        forecast_df["Predicted Price"] = forecast_df["Predicted Price"].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "Predicted Price": st.column_config.TextColumn("Predicted Price"),
                "Day": st.column_config.NumberColumn("Day #", format="%d")
            }
        )
        
        # Download button
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv,
            file_name=f"{symbol}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

else:
    # Welcome screen when no prediction is running
    st.info("👈 **Get Started:** Configure your prediction parameters in the sidebar and click 'GENERATE PREDICTION' to begin!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 How It Works
        1. Select a stock from S&P 500
        2. Choose historical data range
        3. Set prediction timeframe
        4. Click to generate forecast
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 AI Technology
        - LSTM Neural Networks
        - 100-day lookback window
        - Recursive prediction
        - Auto-scaling normalization
        """)
    
    with col3:
        st.markdown("""
        ### ⚡ Features
        - Real-time S&P 500 data
        - Interactive visualizations
        - Model caching
        - CSV export capability
        """)
