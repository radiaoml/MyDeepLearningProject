# ================================================================
# Streamlit App: Stock Price Prediction (LSTM + Plotly Interactive)
# Fixed version with proper scaling and recursive prediction
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
from io import BytesIO

# ---------------------------------------------------------------
# Streamlit Page Settings
# ---------------------------------------------------------------
st.set_page_config(page_title="Stock Prediction LSTM", layout="wide")
st.title("📈 Stock Price Prediction (LSTM) - S&P 500 Stocks")

# ---------------------------------------------------------------
# Fetch S&P 500 Tickers from Wikipedia
# ---------------------------------------------------------------
@st.cache_data
def load_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error("❌ Failed to fetch S&P 500 tickers.")
        st.stop()
    df = pd.read_html(response.text)[0]
    tickers = df["Symbol"].tolist()
    return sorted(tickers)

tickers = load_sp500_tickers()

# ---------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------
st.sidebar.header("Configuration")

symbol = st.sidebar.selectbox("Select Stock", tickers, index=tickers.index("AAPL"))
start_date = st.sidebar.date_input("Historical Data Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("Historical Data End Date", pd.to_datetime("2023-12-31"))

st.sidebar.subheader("Prediction Horizon")
days = st.sidebar.number_input("Days", min_value=0, value=0)
weeks = st.sidebar.number_input("Weeks", min_value=0, value=0)
months = st.sidebar.number_input("Months", min_value=0, value=0)
years = st.sidebar.number_input("Years", min_value=0, value=0)

predict_button = st.sidebar.button("RUN PREDICTION 🚀")

# ---------------------------------------------------------------
# Convert horizon values to total days
# ---------------------------------------------------------------
def total_days(days, weeks, months, years):
    return days + (weeks * 7) + (months * 30) + (years * 365)

# ---------------------------------------------------------------
# ✅ Robust YFinance Downloader
# ---------------------------------------------------------------
def download_fixed(symbol, start, end):
    # Try the normal API first
    df = yf.download(symbol, start=start, end=end, interval="1d",
                     auto_adjust=False, actions=False, progress=False)

    # If empty, try the fallback API
    if df.empty:
        st.warning(f"⚠️ Primary download empty for {symbol}. Trying fallback method...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval="1d", auto_adjust=False)

    # If still empty, stop gracefully
    if df.empty:
        st.error(f"❌ No data found for symbol: {symbol}. Please check the ticker or internet connection.")
        st.stop()

    # Flatten columns if multi-level (can happen in yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip().replace(" ", "_") for col in df.columns]

    # ✅ Debug info (shows columns once)
    print(f"Downloaded columns for {symbol}: {list(df.columns)}")

    # ✅ Ensure we have a usable Close column
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df.rename(columns={"Adj Close": "Close"}, inplace=True)
        elif any("Close" in col for col in df.columns):
            # Sometimes comes as "Close_XX" in multiindex flattening
            close_col = [c for c in df.columns if "Close" in c][0]
            df.rename(columns={close_col: "Close"}, inplace=True)
        else:
            st.error(f"❌ No 'Close' or 'Adj Close' column found for {symbol}. Columns were: {list(df.columns)}")
            st.stop()

    # ✅ Fill missing OHLCV columns gracefully
    for col in ["Open", "High", "Low", "Volume"]:
        if col not in df.columns:
            df[col] = df["Close"]

    # Clean NaNs
    df.dropna(inplace=True)
    return df

# ---------------------------------------------------------------
# Build or Load LSTM model
# ---------------------------------------------------------------
def get_or_train_model(symbol, X_train, y_train):
    # Get the directory where this script is located (A3 folder)
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
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    model.save(model_filename)
    st.success(f"✅ Model trained and saved as {symbol}.h5")
    return model

# ---------------------------------------------------------------
# Prediction Execution
# ---------------------------------------------------------------
if predict_button:
    data = download_fixed(symbol, start_date, end_date)
    close_column = [c for c in data.columns if "Close" in c][0]

    if data.empty:
        st.error("⚠️ No data available for selected stock/date range.")
        st.stop()

    dataset = data[[close_column]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    training_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:training_size]

    X_train, y_train = [], []
    for i in range(100, len(train_data)):  # longer lookback
        X_train.append(train_data[i-100:i, 0])
        y_train.append(train_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = get_or_train_model(symbol, X_train, y_train)

    # ----------------- Historical Prediction -----------------
    test_data = scaled_data[training_size-100:]
    X_test = [test_data[i-100:i, 0] for i in range(100, len(test_data))]
    X_test = np.array(X_test).reshape(-1, 100, 1)
    predictions = scaler.inverse_transform(model.predict(X_test))

    # ----------------- Future Prediction -----------------
    future_days = total_days(days, weeks, months, years)
    last_100 = scaled_data[-100:].tolist()
    future_values = []

    for _ in range(future_days):
        x_input = np.array(last_100[-100:]).reshape(1, 100, 1)
        pred_scaled = model.predict(x_input, verbose=0)[0][0]

        # convert scaled -> actual -> scaled again to maintain realistic feedback loop
        pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
        future_values.append(pred_real)
        pred_rescaled = scaler.transform([[pred_real]])[0][0]
        last_100.append([pred_rescaled])

    future_values_rescaled = np.array(future_values).reshape(-1, 1)
    future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1),
                                 periods=future_days, freq="D")

    # ---------------------------------------------------------------
    # UI LAYOUT (TABS)
    # ---------------------------------------------------------------
    tab_hist, tab_pred = st.tabs(["📉 Historical Chart", "🔮 Prediction Chart"])

    # TAB 1: Historical Chart
    with tab_hist:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=data.index, y=data[close_column],
                                      mode="lines", name="Historical Price"))
        fig_hist.update_layout(title=f"Historical Stock Price - {symbol}",
                               xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.dataframe(data.tail())

    # TAB 2: Prediction Chart
    with tab_pred:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=data.index, y=data[close_column],
                                      mode="lines", name="Historical"))
        fig_pred.add_trace(go.Scatter(x=future_index,
                                      y=future_values_rescaled.flatten(),
                                      mode="lines+markers", name="Predicted", line=dict(color="green")))
        fig_pred.update_layout(title=f"🔮 {symbol} Price Prediction ({future_days} days ahead)",
                               xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig_pred, use_container_width=True)

        st.metric(label=f"Predicted price after {future_days} days",
                  value=f"${future_values_rescaled[-1][0]:.2f}")

        future_df = pd.DataFrame({"Date": future_index, "Predicted Price": future_values_rescaled.flatten()})
        st.dataframe(future_df)
