# 🌍 RAFT: Unified Dual-Domain Dashboard for Extreme Event Forecasting

### Retrieval-Augmented Forecasting of Time Series | Live AI Forecasting Dashboard

RAFT is a **Retrieval-Augmented Forecasting system** designed to improve time-series forecasting for unusual and extreme events by combining **live observations, historical pattern retrieval, and deep-learning-based forecasting**.

Instead of relying only on the latest observations, RAFT retrieves similar historical patterns from a large historical dataset using **FAISS** and provides this historical context to the forecasting pipeline.

The system currently supports two domains:

* 📉 **Financial Domain — Bitcoin Flash-Crash Forecasting**
* 🌊 **Hydrological Domain — River-Level Forecasting**

The application integrates **live REST APIs** with trained forecasting models and provides the results through an interactive **Streamlit dashboard**.

---

## 🎯 Project Objective

Traditional time-series forecasting models primarily learn from sequential historical data. RAFT explores whether retrieving **historically similar patterns** can provide additional context when forecasting unusual or extreme behavior.

The project combines:

**Live Data + Historical Retrieval + Deep Learning + Time-Series Forecasting + Interactive Visualization**

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     LIVE DATA       │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
          Binance REST API             USGS Water API
                 │                           │
                 ▼                           ▼
          Bitcoin Market Data          River Measurements
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   PREPROCESSING     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ HISTORICAL RETRIEVAL │
                    │       FAISS         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  RETRIEVED CONTEXT  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ PYTORCH FORECASTER  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ FORECAST + STATUS   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ STREAMLIT DASHBOARD │
                    └─────────────────────┘
```

---

# 🔌 Live Data APIs

RAFT uses external APIs to obtain the latest observations before generating forecasts.

## 📉 Binance REST API — Financial Domain

The financial pipeline uses the **Binance Spot REST API** to retrieve public Bitcoin market data.

The Binance API provides public market-data endpoints including candlestick/Kline data. The Kline endpoint is:

```text
GET /api/v3/klines
```

Typical inputs include:

* `symbol` — e.g. `BTCUSDT`
* `interval` — e.g. `1m`
* `limit` — number of observations
* Optional `startTime` / `endTime`

For public market data, Binance also provides the market-data-only endpoint:

```text
https://data-api.binance.vision
```

Binance documents `/api/v3/klines` as its candlestick-data endpoint and supports intervals including `1m`, `5m`, `15m`, `1h`, etc.

### Financial Data Flow

```text
Binance REST API
       ↓
Live BTC Market Data
       ↓
Data Preprocessing
       ↓
Historical Pattern Retrieval
       ↓
FAISS Similarity Search
       ↓
Retrieved Historical Context
       ↓
PyTorch Forecasting Model
       ↓
Bitcoin Forecast
       ↓
Flash-Crash Status / Alert
```

---

## 🌊 USGS Water Data API — Hydrological Domain

The hydrological pipeline uses the **USGS Water Data APIs** to retrieve river measurements.

USGS provides machine-readable water data through REST APIs, including continuous measurements such as **streamflow and gage height**.

The modernized USGS Water Data APIs include:

* Continuous Values
* Daily Values
* Monitoring Locations
* Time Series Metadata

Continuous Values are particularly relevant for applications requiring recent sensor measurements.

### Hydrological Data Flow

```text
USGS Water Data API
       ↓
Live River Measurements
       ↓
Data Preprocessing
       ↓
Historical Pattern Retrieval
       ↓
FAISS Similarity Search
       ↓
Retrieved Historical Context
       ↓
PyTorch Forecasting Model
       ↓
River-Level Forecast
       ↓
Flood / Event Status
```

### ⚠️ API Compatibility Note

USGS has introduced modernized Water Data APIs that are replacing the legacy WaterServices APIs. The legacy WaterServices family is scheduled for decommissioning in **Q1 2027**. If the current implementation uses the legacy `waterservices.usgs.gov` endpoints, migration to the modernized APIs is recommended.

---

# 🤖 Retrieval-Augmented Forecasting

The core idea behind RAFT is to retrieve historically similar patterns before forecasting.

```text
Current Time-Series Window
            ↓
     Feature Representation
            ↓
      FAISS Similarity Search
            ↓
 Similar Historical Patterns
            ↓
   Historical Context
            ↓
    Forecasting Model
            ↓
       Prediction
```

FAISS acts as the historical retrieval layer, allowing the system to search for similar patterns within the stored historical data.

---

# 📊 Supported Domains

## 📉 Financial Domain

**Target:** Bitcoin

**Purpose:** Short-term extreme-event / flash-crash forecasting

**Live Source:** Binance REST API

**Historical Dataset:** Bitcoin historical 1-minute data

**Forecasting:** PyTorch

**Retrieval:** FAISS

---

## 🌊 Hydrological Domain

**Target:** River-level behavior

**Purpose:** Short-term extreme-event forecasting

**Live Source:** USGS Water Data API

**Historical Dataset:** USGS Mississippi River historical data

**Forecasting:** PyTorch

**Retrieval:** FAISS

---

# 💻 Tech Stack

| Technology              | Purpose                      |
| ----------------------- | ---------------------------- |
| **Python**              | Core development             |
| **PyTorch**             | Deep-learning forecasting    |
| **FAISS**               | Historical pattern retrieval |
| **Pandas**              | Data processing              |
| **NumPy**               | Numerical computation        |
| **Scikit-learn**        | Preprocessing / ML utilities |
| **Streamlit**           | Interactive dashboard        |
| **Requests**            | REST API communication       |
| **Binance REST API**    | Live Bitcoin data            |
| **USGS Water Data API** | Live hydrological data       |

---

# 📊 Datasets

## Bitcoin

[Kaggle - Bitcoin Historical Data (1-Min Intervals)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

Used as the historical dataset for the financial-domain forecasting pipeline.

## River

[Kaggle - USGS Mississippi River (Baton Rouge)](https://www.kaggle.com/datasets/protobioengineering/usgs-mississippi-river-at-baton-rouge-2004-2023)

Used as the historical dataset for the hydrological-domain forecasting pipeline.

---

# 📈 Streamlit Dashboard

The project includes an interactive Streamlit dashboard for live inference.

The dashboard provides:

* 📉 Live Bitcoin monitoring
* 🌊 Live river-level monitoring
* 🔄 Automatic live-data updates
* 🔮 RAFT forecasts
* 📊 Forecast trajectory visualization
* ⚠️ Event-status indicators
* 🌐 Separate financial and hydrological monitoring

The dashboard connects the complete pipeline:

```text
Live API Data
      ↓
Preprocessing
      ↓
RAFT Retrieval
      ↓
Forecasting
      ↓
Visualization
      ↓
Event Monitoring
```

---

# 🚀 How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/26yashu/RAFT-Unified-Dual-Domain-Dashboard-for-Extreme-Event-Forecasting.git

cd RAFT-Unified-Dual-Domain-Dashboard-for-Extreme-Event-Forecasting
```

## 2. Install Dependencies

```bash
pip install torch pandas numpy scikit-learn matplotlib faiss-cpu streamlit requests
```

## 3. Prepare and Train the Crypto Model

```bash
python crypto_experiment/crypto_data_prep.py

python crypto_experiment/run_crypto_training.py
```

## 4. Prepare and Train the River Model

```bash
python river_experiment/river_data_prep.py

python river_experiment/run_river_training.py
```

## 5. Launch the Dashboard

```bash
streamlit run app.py
```

Once the application starts, enable live updates in the dashboard to retrieve the latest data and generate forecasts.

---

# 📁 Project Structure

```text
RAFT-Unified-Dual-Domain-Dashboard-for-Extreme-Event-Forecasting/
│
├── app.py
│
├── crypto_experiment/
│   ├── crypto_data_prep.py
│   └── run_crypto_training.py
│
├── river_experiment/
│   ├── river_data_prep.py
│   └── run_river_training.py
│
├── README.md
└── requirements.txt
```

---

# 👥 Contributors

## Gandlaparthi Yaswanthi

**Financial Domain & System Architecture**

Responsibilities:

* Developed the financial-domain forecasting pipeline.
* Worked on the overall system architecture.
* Implemented/integrated the financial-domain workflow.
* Contributed to historical pattern retrieval using FAISS.
* Integrated live financial data into the forecasting pipeline.
* Contributed to Streamlit dashboard integration.

## Gooty Kummara Snigdha

**Hydrological Domain**

Responsibilities:

* Developed the hydrological-domain forecasting pipeline.
* Worked on river-data preprocessing.
* Contributed to hydrological forecasting integration.

---

# 🔬 Key Concepts Demonstrated

* Retrieval-Augmented Forecasting
* Time-Series Forecasting
* Historical Pattern Retrieval
* Similarity Search
* FAISS Vector Search
* Deep Learning with PyTorch
* REST API Integration
* Real-Time Data Processing
* Financial Data Analysis
* Hydrological Data Analysis
* Interactive Data Visualization
* Streamlit Application Development
* End-to-End AI System Architecture

---

# 🎯 Future Improvements

* Improve forecasting accuracy through additional features and model architectures.
* Evaluate retrieval quality using quantitative retrieval metrics.
* Add additional financial and environmental datasets.
* Improve anomaly and extreme-event detection.
* Add model-performance monitoring.
* Add historical prediction-versus-actual evaluation.
* Migrate and maintain compatibility with the latest USGS Water Data APIs.
* Deploy the dashboard as a production web application.

---

# 📌 Project Summary

RAFT demonstrates an end-to-end approach to **Retrieval-Augmented Time-Series Forecasting**, connecting historical pattern retrieval with live data sources and deep-learning forecasting models.

The project combines:

**FAISS + PyTorch + REST APIs + Time-Series Data + Streamlit**

to create a unified dashboard for monitoring and forecasting extreme-event behavior across financial and hydrological domains.
