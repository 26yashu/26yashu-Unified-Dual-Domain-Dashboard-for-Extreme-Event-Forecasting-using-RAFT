import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import faiss
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

def load_and_clean_river_csv(filepath):
    print(f"Loading real Kaggle River data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find {filepath}. Make sure you downloaded the Kaggle CSV!")

    # Read the Kaggle CSV
    df = pd.read_csv(filepath)
    print(f"🔍 DEBUG: Raw CSV loaded. Found {len(df)} total rows.")

    # Dynamically find the Time column and the River Flow/Height column
    time_col = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()][0]
    
    # Kaggle datasets might call it 'flow', 'discharge', 'height', or 'level'
    val_col = [col for col in df.columns if 'flow' in col.lower() or 'height' in col.lower() or 'level' in col.lower() or 'discharge' in col.lower()][0]
    
    df = df[[time_col, val_col]].copy()
    df.columns = ['timestamp', 'value']
    
    # Convert to Datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']) # Drop rows if time format was completely broken
    df = df.set_index('timestamp').sort_index()
    
    # Forward fill missing data, then drop any remaining NaNs
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['value'] = df['value'].ffill()
    df = df.dropna()
    
    # Resample to Hourly intervals to ensure standard spacing
    df_hourly = df.resample('1H').mean().dropna()
    print(f"Data cleaned. Retained {len(df_hourly)} hourly samples.")
    return df_hourly

def create_sliding_windows(data, input_len, target_len):
    values = data['value'].values
    num_samples = len(values) - input_len - target_len + 1
    
    X_hist, y_hist = [], []
    for i in range(num_samples):
        X_hist.append(values[i : i + input_len])
        y_hist.append(values[i + input_len : i + input_len + target_len])
    return np.array(X_hist), np.array(y_hist)

if __name__ == "__main__":
    # Pointing to the new CSV file
    DATA_PATH = os.path.join(current_dir, "data", "river_raw.csv")
    SAVE_PATH = os.path.join(current_dir, "data_processed")
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # River Parameters: 24 hours in, 6 hours out
    INPUT_LEN = 24
    TARGET_LEN = 6
    
    # 1. Load & Clean
    df_clean = load_and_clean_river_csv(DATA_PATH)
    
    # 2. Split Data (80/20)
    split_idx = int(len(df_clean) * 0.8)
    df_train = df_clean.iloc[:split_idx].copy()
    df_test = df_clean.iloc[split_idx:].copy()
    
    # 3. Normalize
    scaler = MinMaxScaler()
    df_train['value'] = scaler.fit_transform(df_train['value'].values.reshape(-1, 1))
    df_test['value'] = scaler.transform(df_test['value'].values.reshape(-1, 1))
    
    # 4. Create Windows
    X_train, y_train = create_sliding_windows(df_train, INPUT_LEN, TARGET_LEN)
    X_test, y_test = create_sliding_windows(df_test, INPUT_LEN, TARGET_LEN)
    
    # 5. Build FAISS Index
    print("Building FAISS index for River memory bank...")
    historical_vectors = X_train.reshape(X_train.shape[0], -1).astype('float32')
    index = faiss.IndexFlatL2(INPUT_LEN)
    index.add(historical_vectors)
    
    # 6. Save Assets
    faiss.write_index(index, os.path.join(SAVE_PATH, "river_faiss.index"))
    with open(os.path.join(SAVE_PATH, "river_targets.pkl"), "wb") as f: pickle.dump(y_train, f)
    with open(os.path.join(SAVE_PATH, "river_scaler.pkl"), "wb") as f: pickle.dump(scaler, f)
    np.save(os.path.join(SAVE_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(SAVE_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(SAVE_PATH, "y_test.npy"), y_test)
    
    print(f"✅ Success! River Memory Bank saved to {SAVE_PATH}")


