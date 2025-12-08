# ================================================================
# Flask: Stock Price Prediction App
# ================================================================

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Dropout
import os
from datetime import timedelta

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------

@app.route('/')
def render_home_page():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_stock_prediction():
    """Handle prediction requests with full LSTM model"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        start_date = data.get('start_date', '2015-01-01')
        end_date = data.get('end_date', '2023-12-31')
        prediction_days = int(data.get('prediction_days', 30))
        
        # Download stock data
        df = yf.download(symbol, start=start_date, end=end_date, 
                       interval="1d", auto_adjust=False, actions=False, progress=False)
        
        if df.empty:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)
        
        if df.empty:
            return jsonify({
                'status': 'error',
                'message': f'No data found for {symbol}'
            }), 400
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip().replace(" ", "_") for col in df.columns]
        
        # Get Close column
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns:
                df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            elif any('Close' in col for col in df.columns):
                close_col = [c for c in df.columns if 'Close' in c][0]
                df.rename(columns={close_col: 'Close'}, inplace=True)
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'No Close column found for {symbol}'
                }), 400
        
        dataset = df[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        training_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:training_size]
        
        # Prepare training data
        X_train, y_train = [], []
        for i in range(100, len(train_data)):
            X_train.append(train_data[i-100:i, 0])
            y_train.append(train_data[i, 0])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Get or train model
        model_filename = f"{symbol}.h5"
        
        if os.path.exists(model_filename):
            model = load_model(model_filename)
        else:
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
        
        # Make future predictions
        last_100 = scaled_data[-100:].tolist()
        future_predictions = []
        
        for _ in range(prediction_days):
            x_input = np.array(last_100[-100:]).reshape(1, 100, 1)
            pred_scaled = model.predict(x_input, verbose=0)[0][0]
            
            pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
            future_predictions.append(float(pred_real))
            
            pred_rescaled = scaler.transform([[pred_real]])[0][0]
            last_100.append([pred_rescaled])
        
        # Generate future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=prediction_days,
            freq="D"
        )
        
        current_price = float(dataset[-1][0])
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'predicted_prices': [round(p, 2) for p in future_predictions],
            'prediction_dates': [d.strftime("%Y-%m-%d") for d in future_dates],
            'message': f'Successfully predicted {prediction_days} days for {symbol}'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ---------------------------------------------------------------
# Run the app
# ---------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, port=5000)
