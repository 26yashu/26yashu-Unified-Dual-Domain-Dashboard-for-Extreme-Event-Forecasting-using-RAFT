# 🌍 RAFT: Extreme Event Predictor

## What is this?
Most AI models are good at predicting normal, everyday trends, but they fail when a rare disaster happens. 

This project uses **Retrieval-Augmented Forecasting of Time Series (RAFT)**. Instead of just guessing, the AI looks at live data and instantly searches through 10 years of history to find a matching pattern. We built it to predict two types of disasters:
1. **📉 Bitcoin Flash Crashes** (using the live Binance API)
2. **🌊 River Flash Floods** (using the live USGS API)

## 💻 Tech Stack
* **Python** (Core language)
* **PyTorch** (The AI Brain)
* **FAISS** (The 10-year historical memory search)
* **Streamlit** (The live web dashboard)

### 📊 Dataset Links
* **Bitcoin Data:**
   [Kaggle - Bitcoin Historical Data (1-Min Intervals)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
* **River Data:**
   [Kaggle - USGS Mississippi River (Baton Rouge)](https://www.kaggle.com/datasets/protobioengineering/usgs-mississippi-river-at-baton-rouge-2004-2023)

## 🚀 How to Run It

**1. Install required packages:**

`pip install torch pandas numpy scikit-learn matplotlib faiss-cpu streamlit requests`

**2. Train the AI Models:**

First, you need to build the memory banks and train both AIs.

*Crypto Model:*

`python crypto_experiment/crypto_data_prep.py`
`python crypto_experiment/run_crypto_training.py`

*River Model:*

`python river_experiment/river_data_prep.py`
`python river_experiment/run_river_training.py`

**3. Start the Live Dashboard:**

Once the models are trained, launch the user interface:

`streamlit run app.py`

Click **"Fetch Live Global Data"** in the app to see the real-time predictions!

## 👥 Creators
* **Gandlaparthi Yaswanthi** - Financial Domain & System Architecture
* **Gooty Kummara Snigdha** - Hydrological Domain

