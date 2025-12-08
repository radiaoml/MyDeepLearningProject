# Flask Stock Prediction App

## 🚀 Quick Start

### Run the Flask App

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Navigate to flask folder
cd A3\flask

# Run the Flask app
python app.py

# Open: http://localhost:5000
```

---

## 📚 Features

- ✅ **Full LSTM Predictions** - Complete stock price forecasting
- ✅ **Beautiful UI** - Modern gradient design with animations
- ✅ **Real-time Data** - Downloads stock data from Yahoo Finance
- ✅ **Model Caching** - Trains once, loads instantly after
- ✅ **CORS Enabled** - Can be called from external apps

---

## 🎯 How It Works

1. **Enter stock symbol** (e.g., AAPL, MSFT, GOOGL)
2. **Select date range** for historical data
3. **Choose prediction horizon** (days into the future)
4. **Click "Predict"** and wait for results
5. **View predictions** with current price and future forecasts

---

## 📁 Project Structure

```
A3/flask/
├── app.py                # Flask application with LSTM
├── templates/
│   └── index.html       # Beautiful web UI
└── *.h5                 # Trained models (auto-generated)
```

---

## 🔧 Configuration

### Change Port
```python
# In app.py, change the last line:
app.run(debug=True, port=5001)
```

### Production Mode
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

---

## 📊 All Deployments Comparison

| Framework | Port | UI | LSTM | Status |
|-----------|------|----|----- |--------|
| **Streamlit** | 8501 | ✅ Built-in | ✅ Full | ✅ Ready |
| **FastAPI** | 8000 | ✅ Custom | ✅ Full | ✅ Ready |
| **Django** | 8000 | ✅ Custom | ✅ Full | ✅ Ready |
| **Flask** | 5000 | ✅ Custom | ✅ Full | ✅ Ready |

---

## 🎉 You Now Have 4 Complete Deployments!

All with full LSTM prediction functionality and beautiful UIs! 🚀
