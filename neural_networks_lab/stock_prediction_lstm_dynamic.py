# ================================================================
# Lab: Stock Price Prediction using LSTM and Yahoo Finance
# Réalisé par: Your Name (EMSI 2023/2024)
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# ------------------------------
# Step 1: Download dataset
# ------------------------------
# You can change 'AAPL' to any other stock symbol (e.g., 'GOOG', 'MSFT', 'TSLA', 'META', etc.)
symbol = 'AAPL'
data = yf.download(symbol, start='2015-01-01', end='2023-12-31')
print(data.head())

# Use only the 'Close' price
dataset = data[['Close']].values

# ------------------------------
# Step 2: Data preprocessing
# ------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

training_data_len = int(len(scaled_data) * 0.8)

train_data = scaled_data[:training_data_len]
test_data = scaled_data[training_data_len - 60:]

# Prepare training data
X_train = []
y_train = []
for i in range(60, len(train_data)):
    X_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# ------------------------------
# Step 3: Build LSTM Model
# ------------------------------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Output layer (next closing price)

model.compile(optimizer='adam', loss='mean_squared_error')

# ------------------------------
# Step 4: Train the model
# ------------------------------
print("Training the LSTM model...")
model.fit(X_train, y_train, epochs=50, batch_size=32)

# ------------------------------
# Step 5: Prepare test data
# ------------------------------
X_test = []
y_test = dataset[training_data_len:]

#60 days prior to predict the 61st day(wich is the next day after the 60th day)
for i in range(60, len(test_data)):
    X_test.append(test_data[i - 60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# ------------------------------
# Step 6: Predictions
# ------------------------------
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# ------------------------------
# Step 7: Evaluation and Visualization
# ------------------------------
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(12, 6))
plt.title(f'{symbol} Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.plot(train['Close'], label='Training Data')
plt.plot(valid[['Close']], label='Actual Price', color='black')
plt.plot(valid[['Predictions']], label='Predicted Price', color='green')
plt.legend()
plt.show()

# ------------------------------
# Step 8: Save model
# ------------------------------
model.save(f'{symbol}_lstm_model.h5')
print(f"LSTM model saved as {symbol}_lstm_model.h5")
