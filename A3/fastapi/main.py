# ================================================================
# FastAPI: Stock Price Prediction API
# ================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Dropout
import os
from typing import List, Optional
from datetime import datetime, timedelta

app = FastAPI(
    title="📈 Stock Price Prediction API",
    description="LSTM-based stock price prediction API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# ---------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------

class PredictionRequest(BaseModel):
    symbol: str
    start_date: str = "2015-01-01"
    end_date: str = "2023-12-31"
    prediction_days: int = 30

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_prices: List[float]
    prediction_dates: List[str]
    message: str

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------

def download_stock_data(symbol: str, start: str, end: str):
    """Download stock data from Yahoo Finance"""
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d",
                        auto_adjust=False, actions=False, progress=False)
        
        if df.empty:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval="1d", auto_adjust=False)
        
        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip().replace(" ", "_") for col in df.columns]
        
        # Ensure Close column exists
        if "Close" not in df.columns:
            if "Adj Close" in df.columns:
                df.rename(columns={"Adj Close": "Close"}, inplace=True)
            elif any("Close" in col for col in df.columns):
                close_col = [c for c in df.columns if "Close" in c][0]
                df.rename(columns={close_col: "Close"}, inplace=True)
            else:
                raise ValueError(f"No 'Close' column found for {symbol}")
        
        df.dropna(inplace=True)
        return df
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading data: {str(e)}")

def get_or_train_model(symbol: str, X_train, y_train):
    """Load existing model or train a new one"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(script_dir, f"{symbol}.h5")
    
    if os.path.exists(model_filename):
        return load_model(model_filename)
    
    # Train new model
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
    return model

# ---------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------

@app.get("/")
def root():
    """Serve the web UI"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

@app.get("/api")
def api_info():
    """API information"""
    return {
        "message": "📈 Stock Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/": "GET - Web UI",
            "/api": "GET - API information",
            "/predict": "POST - Get stock price predictions",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
def predict_stock_price(request: PredictionRequest):
    """
    Predict future stock prices using LSTM model
    
    - **symbol**: Stock ticker symbol (e.g., AAPL, MSFT)
    - **start_date**: Historical data start date (YYYY-MM-DD)
    - **end_date**: Historical data end date (YYYY-MM-DD)
    - **prediction_days**: Number of days to predict into the future
    """
    try:
        # Download data
        data = download_stock_data(request.symbol, request.start_date, request.end_date)
        close_column = [c for c in data.columns if "Close" in c][0]
        dataset = data[[close_column]].values
        
        # Prepare data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        training_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:training_size]
        
        # Create training sequences
        X_train, y_train = [], []
        for i in range(100, len(train_data)):
            X_train.append(train_data[i-100:i, 0])
            y_train.append(train_data[i, 0])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Get or train model
        model = get_or_train_model(request.symbol, X_train, y_train)
        
        # Make future predictions
        last_100 = scaled_data[-100:].tolist()
        future_predictions = []
        
        for _ in range(request.prediction_days):
            x_input = np.array(last_100[-100:]).reshape(1, 100, 1)
            pred_scaled = model.predict(x_input, verbose=0)[0][0]
            
            pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
            future_predictions.append(float(pred_real))
            
            pred_rescaled = scaler.transform([[pred_real]])[0][0]
            last_100.append([pred_rescaled])
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=request.prediction_days,
            freq="D"
        )
        
        current_price = float(dataset[-1][0])
        
        return PredictionResponse(
            symbol=request.symbol,
            current_price=round(current_price, 2),
            predicted_prices=[round(p, 2) for p in future_predictions],
            prediction_dates=[d.strftime("%Y-%m-%d") for d in future_dates],
            message=f"Successfully predicted {request.prediction_days} days for {request.symbol}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------
# Run with: uvicorn main:app --reload
# ---------------------------------------------------------------
