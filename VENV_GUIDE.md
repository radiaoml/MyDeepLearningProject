# Virtual Environment Setup Guide

## ✅ Virtual Environment Created Successfully!

A Python virtual environment has been created with all necessary dependencies for your neural network models and Streamlit app.

---

## 📦 Installed Packages

The following packages have been installed:

- **tensorflow** (>=2.13.0) - Deep learning framework
- **keras** (>=2.13.0) - High-level neural networks API
- **numpy** (>=1.24.0) - Numerical computing
- **pandas** (>=2.0.0) - Data manipulation
- **scikit-learn** (>=1.3.0) - Machine learning utilities
- **matplotlib** (>=3.7.0) - Plotting library
- **plotly** (>=5.14.0) - Interactive visualizations
- **streamlit** (>=1.28.0) - Web app framework
- **yfinance** (>=0.2.28) - Yahoo Finance data
- **requests** (>=2.31.0) - HTTP library

---

## 🚀 How to Use the Virtual Environment

### Activate the Virtual Environment

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
.\venv\Scripts\activate.bat
```

After activation, your terminal prompt will show `(venv)` prefix.

### Run Your Applications

#### 1. Run Streamlit App (Primary App)
```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1

# Navigate to A3
cd A3

# Run the app
streamlit run stock_prediction_app.py
```

#### 2. Run Neural Network Training Scripts
```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1

# Navigate to A1
cd A1

# Run any of the models:
python Lab_ANN_Bousmah (1).py      # ANN - Fashion MNIST
python Lab_lstm.py                  # LSTM - TATA stock
python lab_lstm_dynamic.py          # LSTM - Dynamic stock
python lab_rnn.py                   # RNN - Stock prediction
```

### Deactivate the Virtual Environment

```powershell
deactivate
```

---

## 📝 Requirements File

All dependencies are listed in `requirements.txt`. To reinstall packages:

```powershell
.\venv\Scripts\pip install -r requirements.txt
```

---

## 🔧 Troubleshooting

### If activation fails on PowerShell:
```powershell
# Run this once to allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### To upgrade packages:
```powershell
.\venv\Scripts\pip install --upgrade -r requirements.txt
```

### To check installed packages:
```powershell
.\venv\Scripts\pip list
```

---

## 📂 Project Structure Reminder

```
lab1/
├── venv/                    # ✅ Virtual environment
├── requirements.txt         # ✅ Dependencies list
├── A1/                      # Neural network models
│   ├── Lab_ANN_Bousmah (1).py
│   ├── Lab_lstm.py
│   ├── lab_lstm_dynamic.py
│   └── lab_rnn.py
└── A3/                      # Streamlit app
    ├── AAPL.h5
    └── stock_prediction_app.py
```

---

## ✨ Quick Start

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run your Streamlit app
cd A3
streamlit run stock_prediction_app.py

# 3. Open browser at http://localhost:8501
```

Enjoy your clean, organized project! 🎉
