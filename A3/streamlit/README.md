# Streamlit Stock Prediction App

## 🌐 Live Demo

**Try it now:** [https://lab1-stock-prediction-app.streamlit.app/](https://lab1-stock-prediction-app.streamlit.app/)

The app is deployed on Streamlit Cloud and ready to use!

---

## 🚀 Run Locally

### Run the Streamlit App

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Navigate to streamlit folder
cd A3\streamlit

# Run the Streamlit app
streamlit run stock_prediction_app.py

# Open: http://localhost:8501
```

---

## 📚 Features

- ✅ **S&P 500 Stock Selection** - Choose from 500+ stocks
- ✅ **Interactive Charts** - Beautiful Plotly visualizations
- ✅ **Full LSTM Predictions** - Complete stock price forecasting
- ✅ **Custom Date Ranges** - Select your own historical data period
- ✅ **Flexible Predictions** - Predict days, weeks, months, or years ahead
- ✅ **Model Caching** - Trains once, loads instantly after
- ✅ **Real-time Data** - Downloads stock data from Yahoo Finance

---

## 🎯 How It Works

1. **Select stock** from S&P 500 dropdown (default: AAPL)
2. **Choose date range** for historical data
3. **Set prediction horizon** (days/weeks/months/years)
4. **Click "RUN PREDICTION"** and wait for results
5. **View charts** in two tabs:
   - 📉 Historical Chart - Past stock prices
   - 🔮 Prediction Chart - Future forecasts

---

## 📊 What Makes Streamlit Special?

| Feature | Streamlit | Others |
|---------|-----------|--------|
| **UI Framework** | Built-in | Custom HTML/CSS |
| **Interactivity** | Native widgets | JavaScript |
| **Charts** | Plotly integration | Manual setup |
| **Development** | Fastest | More setup |
| **Best For** | Data apps & demos | Production APIs |

---

## 🔧 Configuration

### Change Port
```powershell
streamlit run stock_prediction_app.py --server.port 8502
```

### Disable Auto-reload
```powershell
streamlit run stock_prediction_app.py --server.runOnSave false
```

---

## 📁 Project Structure

```
A3/streamlit/
├── stock_prediction_app.py  # Main Streamlit application
├── AAPL.h5                   # Pre-trained model (optional)
└── README.md                 # This file
```

---

## 🚀 Deployment

### Streamlit Cloud (Live)
This app is deployed at: **[https://lab1-stock-prediction-app.streamlit.app/](https://lab1-stock-prediction-app.streamlit.app/)**

### Deploy Your Own
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `A3/streamlit/stock_prediction_app.py`
5. Deploy!

---

## 💡 Tips

- **First prediction** for each stock takes 5-10 minutes (training)
- **Subsequent predictions** are instant (uses cached model)
- **Models are saved** as `{SYMBOL}.h5` in the streamlit folder
- **Try different stocks** - All S&P 500 stocks are available!

---

## 🎉 Streamlit is the Most Feature-Complete!

This is the **recommended deployment** for:
- ✅ Quick demos and presentations
- ✅ Data exploration and analysis
- ✅ Interactive stock predictions
- ✅ Beautiful visualizations

For APIs, use FastAPI, Django, or Flask instead! 🚀
