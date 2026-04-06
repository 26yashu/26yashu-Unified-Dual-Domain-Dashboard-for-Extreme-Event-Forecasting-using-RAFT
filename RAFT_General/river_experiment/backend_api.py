import sys
import os
import requests
import pandas as pd
import numpy as np
import torch
import faiss
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Core.model_definition import RAFTModel

def get_live_river_forecast():
    PROCESSED_DATA_PATH = os.path.join(current_dir, "data_processed")
    MODEL_SAVE_PATH = os.path.join(current_dir, "saved_models")
    
    INPUT_LEN = 24; TARGET_LEN = 6; NUM_MATCHES = 3; HIDDEN_DIM = 64
    memory_input_dim = NUM_MATCHES * TARGET_LEN
    
    # 1. Fetch Live USGS Data
    url = "https://waterservices.usgs.gov/nwis/iv/?format=json&sites=08158000&parameterCd=00065&period=P7D"
    try:
        response = requests.get(url)
        if response.status_code != 200: return None, None, "API Error"
        
        data = response.json()
        time_series = data['value']['timeSeries'][0]['values'][0]['value']
        
        df = pd.DataFrame({
            'timestamp': [entry['dateTime'] for entry in time_series],
            'value': [float(entry['value']) for entry in time_series]
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df['value'] = df['value'].interpolate(method='linear', limit=6)
        df_hourly = df.resample('1h').mean().dropna()
        
        live_window = df_hourly['value'].values[-INPUT_LEN:]
        current_level = live_window[-1]
    except Exception as e:
        return None, None, f"API Error: {str(e)}"
    
    # 2. Load Assets
    with open(os.path.join(PROCESSED_DATA_PATH, "river_scaler.pkl"), "rb") as f: scaler = pickle.load(f)
    faiss_index = faiss.read_index(os.path.join(PROCESSED_DATA_PATH, "river_faiss.index"))
    with open(os.path.join(PROCESSED_DATA_PATH, "river_targets.pkl"), "rb") as f: y_history = pickle.load(f)
        
    model = RAFTModel(INPUT_LEN, memory_input_dim, TARGET_LEN, HIDDEN_DIM)
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "river_raft_model.pth")))
    model.eval()
    
    # 3. Predict
    live_scaled = scaler.transform(live_window.reshape(-1, 1)).flatten()
    X_tensor = torch.from_numpy(live_scaled).float().unsqueeze(0).unsqueeze(-1)
    
    with torch.no_grad():
        distances, indices = faiss_index.search(X_tensor.view(1, -1).numpy(), NUM_MATCHES)
        retrieved_y_tensor = torch.from_numpy(y_history[indices[0]]).float().unsqueeze(0)
        prediction_scaled = model(X_tensor, retrieved_y_tensor).numpy()
        
    predicted_levels = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
    max_predicted_level = np.max(predicted_levels)
    
    # 4. Status
    if max_predicted_level > (current_level + 2.0): status = "⚠️ FLASH FLOOD WARNING"
    else: status = "⚖️ STABLE"
        
    return current_level, max_predicted_level, status

if __name__ == "__main__":
    print("Testing Live River API...")
    curr, pred, stat = get_live_river_forecast()
    print(f"Current Level: {curr:.2f} ft")
    print(f"Max Predicted (Next 6h): {pred:.2f} ft")
    print(f"Status: {stat}")

