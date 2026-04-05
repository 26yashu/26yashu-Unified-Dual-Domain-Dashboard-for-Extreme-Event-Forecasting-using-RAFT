import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import faiss
import os

# Automatically find where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

def load_and_clean_crypto_data(filepath):
    print(f"Loading raw Bitcoin data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find {filepath}. Check your file name!")

    # Read the CSV
    df = pd.read_csv(filepath)
    print(f"🔍 DEBUG: Raw CSV loaded. Found {len(df)} total rows.") # Let's see how big the file is

    time_col = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()][0]
    close_col = [col for col in df.columns if 'close' in col.lower()][0]
    
    df = df[[time_col, close_col]].copy()
    df.columns = ['timestamp', 'value']
    
    # ---------------------------------------------------------
    # SMART TIMESTAMP PARSER
    # Let's figure out if your time is normal dates, seconds, or milliseconds
    # ---------------------------------------------------------
    sample_time = str(df['timestamp'].iloc[0])
    
    if sample_time.replace('.','',1).isdigit(): # If it looks like a pure number (Unix)
        if len(sample_time) > 11:
            print("🔍 DEBUG: Detected Millisecond Unix Timestamps.")
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            print("🔍 DEBUG: Detected Second Unix Timestamps.")
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        print("🔍 DEBUG: Detected Standard Text Dates.")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    df = df.set_index('timestamp').sort_index()
    
    print(f"🔍 DEBUG: Time range of your data is from {df.index.min()} to {df.index.max()}")
    
    # Forward fill missing minutes, then drop any remaining NaNs
    df['value'] = df['value'].ffill()
    df = df.dropna()
    
    # Resample to 5-Minute intervals to smooth the noise
    df_5min = df.resample('5min').mean().dropna()
    print(f"Data cleaned. Retained {len(df_5min)} 5-minute samples.")
    return df_5min

def create_sliding_windows(data, input_len, target_len):
    values = data['value'].values
    num_samples = len(values) - input_len - target_len + 1
    
    X_hist, y_hist = [], []
    for i in range(num_samples):
        X_hist.append(values[i : i + input_len])
        y_hist.append(values[i + input_len : i + input_len + target_len])
        
    return np.array(X_hist), np.array(y_hist)

if __name__ == "__main__":
    # Point to your exact file location
    DATA_PATH = os.path.join(current_dir, "data", "bitcoin.csv")
    SAVE_PATH = os.path.join(current_dir, "data_processed")
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Parameters: 60 mins in (12 x 5min), 5 mins out (1 x 5min)
    INPUT_LEN = 12 
    TARGET_LEN = 1
    
    # 1. Load & Clean
    df_clean = load_and_clean_crypto_data(DATA_PATH)
    
    # 2. Split Data (80% Train, 20% Test)
    split_idx = int(len(df_clean) * 0.8)
    df_train = df_clean.iloc[:split_idx].copy()
    df_test = df_clean.iloc[split_idx:].copy()
    
    # 3. Normalize (CRUCIAL)
    scaler = MinMaxScaler()
    df_train['value'] = scaler.fit_transform(df_train['value'].values.reshape(-1, 1))
    df_test['value'] = scaler.transform(df_test['value'].values.reshape(-1, 1))
    
    # 4. Create Windows
    X_train, y_train = create_sliding_windows(df_train, INPUT_LEN, TARGET_LEN)
    X_test, y_test = create_sliding_windows(df_test, INPUT_LEN, TARGET_LEN)
    
    # 5. Build FAISS Index
    print("Building FAISS index for Crypto memory bank...")
    historical_vectors = X_train.reshape(X_train.shape[0], -1).astype('float32')
    index = faiss.IndexFlatL2(INPUT_LEN)
    index.add(historical_vectors)
    
    # 6. Save Everything
    faiss.write_index(index, os.path.join(SAVE_PATH, "crypto_faiss.index"))
    with open(os.path.join(SAVE_PATH, "crypto_targets.pkl"), "wb") as f: pickle.dump(y_train, f)
    with open(os.path.join(SAVE_PATH, "crypto_scaler.pkl"), "wb") as f: pickle.dump(scaler, f)
    np.save(os.path.join(SAVE_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(SAVE_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(SAVE_PATH, "y_test.npy"), y_test)
    
    print(f"✅ Success! Memory Bank built and saved to {SAVE_PATH}")