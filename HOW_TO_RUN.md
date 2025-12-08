# 🚀 How to Run All Apps

## Project Overview

You have **3 different deployments** of the stock prediction app:

1. **Streamlit** - Interactive web app (easiest to use)
2. **FastAPI** - REST API with web UI
3. **Django** - Full web framework (basic setup)

---

## 1️⃣ Streamlit App (Recommended for Quick Use)

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run Streamlit
cd A3\streamlit
streamlit run stock_prediction_app.py

# Open: http://localhost:8501
```

**Features:**
- ✅ Beautiful interactive UI
- ✅ S&P 500 stock selection
- ✅ Real-time predictions
- ✅ Interactive charts

---

## 2️⃣ FastAPI (API + Web UI)

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run FastAPI
cd A3\fastapi
uvicorn main:app --reload

# Open: http://localhost:8000
```

**Features:**
- ✅ Modern web UI
- ✅ REST API endpoints
- ✅ Interactive API docs at /docs
- ✅ CORS enabled

---

## 3️⃣ Django (Basic Setup)

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Navigate to django folder
cd A3\django

# Run migrations (first time only)
..\..\venv\Scripts\python manage.py migrate

# Start Django server
..\..\venv\Scripts\python manage.py runserver

# Open: http://localhost:8000
```

**Status:** ⏳ Basic structure created (needs views/templates)

---

## 📊 Quick Comparison

| App | Port | Best For | Setup Time |
|-----|------|----------|------------|
| **Streamlit** | 8501 | Quick demos | ✅ Ready |
| **FastAPI** | 8000 | APIs + UI | ✅ Ready |
| **Django** | 8000 | Full web apps | ✅ Ready |
| **Flask** | 5000 | Lightweight apps | ✅ Ready |

---

## 🎯 Recommended Usage

**For stock predictions:**
→ Use **Streamlit** (most feature-complete)

**For API integration:**
→ Use **FastAPI** (best API + has UI)

**For learning Django:**
→ Use **Django** (needs more setup)

---

## 🔧 Troubleshooting

### Port Already in Use
```powershell
# Streamlit - use different port
streamlit run stock_prediction_app.py --server.port 8502

# FastAPI - use different port
uvicorn main:app --reload --port 8001

# Django - use different port
python manage.py runserver 8080
```

### Virtual Environment Not Activated
```powershell
# Activate it first
.\venv\Scripts\Activate.ps1

# You should see (venv) in your terminal
```

---

## 📁 Project Structure

```
lab1/
├── venv/                        # Virtual environment
├── requirements.txt             # All dependencies
├── A1/                          # Neural network models
│   ├── Lab_ANN_Bousmah (1).py
│   ├── Lab_lstm.py
│   ├── lab_lstm_dynamic.py
│   └── lab_rnn.py
└── A3/                          # Deployments
    ├── streamlit/               # ✅ Streamlit (READY)
    │   ├── stock_prediction_app.py
    │   └── README.md
    ├── fastapi/                 # ✅ FastAPI (READY)
    │   ├── main.py
    │   ├── static/index.html
    │   └── README.md
    ├── django/                  # ✅ Django (READY)
    │   ├── manage.py
    │   ├── stock_prediction/
    │   ├── predictor/
    │   └── README.md
    └── flask/                   # ✅ Flask (READY)
        ├── app.py
        ├── templates/index.html
        └── README.md
```

---

## ✨ Quick Start (Recommended)

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run Streamlit (easiest)
cd A3\streamlit
streamlit run stock_prediction_app.py

# 3. Open browser at http://localhost:8501
# 4. Select stock, set dates, click "RUN PREDICTION"
```

Enjoy! 🎉
