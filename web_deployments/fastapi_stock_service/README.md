# FastAPI Stock Prediction API

## 🚀 Quick Start

### 1. Install Dependencies (if not already installed)
```powershell
pip install fastapi uvicorn[standard] pydantic
```

### 2. Run the API
```powershell
# From the project root
cd web_deployments\fastapi_stock_service
uvicorn api_server:app --reload

# Or from anywhere
uvicorn web_deployments.fastapi_stock_service.api_server:app --reload
```

### 3. Access the Application

Once running, you can access:
- **🎨 Web UI**: http://localhost:8000 (Beautiful interactive interface)
- **📚 API Docs (Swagger)**: http://localhost:8000/docs
- **📖 Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **ℹ️ API Info**: http://localhost:8000/api

---

## 🎨 Web UI Features

The web interface provides:
- ✨ **Modern gradient design** with smooth animations
- 📊 **Interactive form** for stock symbol and date selection
- 🚀 **Real-time predictions** with loading indicators
- 📈 **Visual prediction cards** showing future prices
- 💰 **Current price display** for selected stock
- 🔄 **Easy reset** functionality

Simply open http://localhost:8000 in your browser and start predicting!

---

## 📚 API Documentation

Once running, visit:
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc

---

## 🔌 API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```
GET /health
```
Returns API health status and timestamp.

### 3. Stock Price Prediction
```
POST /predict
```

**Request Body:**
```json
{
  "symbol": "AAPL",
  "start_date": "2015-01-01",
  "end_date": "2023-12-31",
  "prediction_days": 30
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "current_price": 195.50,
  "predicted_prices": [196.20, 197.10, ...],
  "prediction_dates": ["2024-01-01", "2024-01-02", ...],
  "message": "Successfully predicted 30 days for AAPL"
}
```

---

## 💻 Example Usage

### Using cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "start_date": "2015-01-01",
    "end_date": "2023-12-31",
    "prediction_days": 30
  }'
```

### Using Python Requests
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "symbol": "AAPL",
    "start_date": "2015-01-01",
    "end_date": "2023-12-31",
    "prediction_days": 30
}

response = requests.post(url, json=data)
print(response.json())
```

### Using JavaScript (Fetch)
```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    symbol: 'AAPL',
    start_date: '2015-01-01',
    end_date: '2023-12-31',
    prediction_days: 30
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## 🎯 Features

- ✅ **RESTful API** with FastAPI
- ✅ **LSTM Model** for stock price prediction
- ✅ **Auto-caching** - Loads existing models or trains new ones
- ✅ **CORS enabled** - Can be called from web browsers
- ✅ **Interactive documentation** - Swagger UI included
- ✅ **Type validation** - Pydantic models for request/response
- ✅ **Error handling** - Proper HTTP status codes

---

## 📁 Project Structure

```
web_deployments/
├── fastapi_stock_service/
│   ├── api_server.py        # FastAPI application
│   ├── requirements.txt     # FastAPI dependencies
│   ├── README.md           # This file
│   ├── static/             # Web UI files
│   └── *.h5                # Trained models (auto-generated)
```

---

## 🔧 Configuration

### Change Port
```powershell
uvicorn api_server:app --reload --port 8080
```

### Production Mode (no auto-reload)
```powershell
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### With Workers (for production)
```powershell
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🐛 Troubleshooting

### Port Already in Use
```powershell
# Use a different port
uvicorn api_server:app --reload --port 8001
```

### Module Not Found
```powershell
# Make sure you're in the right directory
cd web_deployments\fastapi_stock_service
uvicorn api_server:app --reload
```

### Model Training Takes Too Long
- First prediction will train the model (5-10 minutes)
- Subsequent predictions use the cached model (instant)
- Models are saved as `{SYMBOL}.h5` in the fastapi_stock_service folder

---

## 📊 Supported Stocks

Any stock symbol available on Yahoo Finance:
- **Tech**: AAPL, MSFT, GOOGL, AMZN, META, TSLA
- **Finance**: JPM, BAC, GS, MS
- **Healthcare**: JNJ, PFE, UNH
- **And many more!**

---

## 🚀 Next Steps

1. Test the API using the Swagger docs at http://localhost:8000/docs
2. Integrate with your frontend application
3. Deploy to production (Heroku, AWS, Azure, etc.)

Enjoy your FastAPI stock prediction service! 📈
