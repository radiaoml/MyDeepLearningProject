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
cd web_deployments\streamlit_stock_app
streamlit run main_app.py

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
cd web_deployments\fastapi_stock_service
uvicorn api_server:app --reload

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
cd web_deployments\django_stock_platform

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
streamlit run main_app.py --server.port 8502

# FastAPI - use different port
uvicorn api_server:app --reload --port 8001

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
my-deep-learning-repo/
├── venv/                        # Virtual environment
├── requirements.txt             # All dependencies
├── neural_networks_lab/         # Neural network models
│   ├── fashion_mnist_classification.py
│   ├── stock_prediction_lstm.py
│   ├── stock_prediction_lstm_dynamic.py
│   └── stock_prediction_rnn.py
└── web_deployments/             # Deployments
    ├── streamlit_stock_app/     # ✅ Streamlit (READY)
    │   ├── main_app.py
    │   └── README.md
    ├── fastapi_stock_service/   # ✅ FastAPI (READY)
    │   ├── api_server.py
    │   ├── static/index.html
    │   └── README.md
    ├── django_stock_platform/   # ✅ Django (READY)
    │   ├── manage.py
    │   ├── stock_prediction/
    │   ├── predictor/
    │   └── README.md
    └── flask_stock_api/         # ✅ Flask (READY)
        ├── stock_api.py
        ├── templates/index.html
        └── README.md
```

---

## ✨ Quick Start (Recommended)

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run Streamlit (easiest)
cd web_deployments\streamlit_stock_app
streamlit run main_app.py

# 3. Open browser at http://localhost:8501
# 4. Select stock, set dates, click "RUN PREDICTION"
```

Enjoy! 🎉
