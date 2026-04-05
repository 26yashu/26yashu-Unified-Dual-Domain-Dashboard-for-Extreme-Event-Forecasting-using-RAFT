import sys
import os
import requests
import numpy as np
import torch
import faiss
import pickle

# Setup paths to find your Core folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import from your specific 'Core' folder
from Core.model_definition import RAFTModel

def get_live_crypto_forecast():
    """Fetches live Binance data, runs RAFT, and returns the forecast for the UI."""
    PROCESSED_DATA_PATH = os.path.join(current_dir, "data_processed")
    MODEL_SAVE_PATH = os.path.join(current_dir, "saved_models")
    
    # These MUST match what you trained with
    INPUT_LEN = 12
    TARGET_LEN = 1
    NUM_MATCHES = 3
    HIDDEN_DIM = 64
    memory_input_dim = NUM_MATCHES * TARGET_LEN
    
    # 1. Fetch Live Binance Data (Last 60 mins / 12 x 5min candles)
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=12"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None, None, "API Error: Binance connection failed"
        
        # Binance candle format: index 4 is the Close price
        live_window = np.array([float(candle[4]) for candle in response.json()])
        current_price = live_window[-1]
    except Exception as e:
        return None, None, f"API Error: {str(e)}"
    
    # 2. Load Saved Assets from your successful training run
    with open(os.path.join(PROCESSED_DATA_PATH, "crypto_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    faiss_index = faiss.read_index(os.path.join(PROCESSED_DATA_PATH, "crypto_faiss.index"))
    with open(os.path.join(PROCESSED_DATA_PATH, "crypto_targets.pkl"), "rb") as f:
        y_history = pickle.load(f)
        
    model = RAFTModel(current_input_dim=INPUT_LEN, memory_input_dim=memory_input_dim, output_dim=TARGET_LEN, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "crypto_raft_model.pth")))
    model.eval()
    
    # 3. Predict the Future
    # Scale the live dollars to 0.0-1.0
    live_scaled = scaler.transform(live_window.reshape(-1, 1)).flatten()
    X_tensor = torch.from_numpy(live_scaled).float().unsqueeze(0).unsqueeze(-1)
    
    with torch.no_grad():
        # Search memory bank
        distances, indices = faiss_index.search(X_tensor.view(1, -1).numpy(), NUM_MATCHES)
        retrieved_y_tensor = torch.from_numpy(y_history[indices[0]]).float().unsqueeze(0)
        
        # Make prediction
        prediction_scaled = model(X_tensor, retrieved_y_tensor).numpy()
        
    # Unscale the prediction back to dollars
    predicted_price = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
    
    # 4. Determine Status Warning
    diff = predicted_price - current_price
    if diff < -100: 
        status = "⚠️ CRASH WARNING"
    elif diff > 100: 
        status = "🚀 SPIKE WARNING"
    else: 
        status = "⚖️ STABLE"
        
    return current_price, predicted_price, status

# Quick test to make sure it works if you run this file directly
if __name__ == "__main__":
    print("Testing Live Crypto API integration...")
    curr, pred, stat = get_live_crypto_forecast()
    print(f"Current Price: ${curr:,.2f}")
    print(f"Predicted Price (Next 5m): ${pred:,.2f}")
    print(f"System Status: {stat}")