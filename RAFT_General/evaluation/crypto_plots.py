import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import faiss

# --- Path Routing ---
# Tell Python to look in the main project folder to find Core and crypto_experiment
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Core.model_definition import RAFTModel

# Re-define Baseline architecture to load the weights cleanly
class StandardLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super(StandardLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.output_head(hidden[-1])

def retrieve_historical_targets(current_input_batch, faiss_index, historical_targets, k=3):
    """Helper function to search the memory bank during evaluation."""
    query_vectors = current_input_batch.view(current_input_batch.size(0), -1).detach().numpy().astype('float32')
    distances, indices = faiss_index.search(query_vectors, k)
    retrieved_y_batch = [historical_targets[indices[i]] for i in range(current_input_batch.size(0))]
    return torch.from_numpy(np.array(retrieved_y_batch)).float()

if __name__ == "__main__":
    # Setup Paths pointing to the crypto_experiment folder
    CRYPTO_DIR = os.path.join(parent_dir, "crypto_experiment")
    PROCESSED_DATA = os.path.join(CRYPTO_DIR, "data_processed")
    MODELS = os.path.join(CRYPTO_DIR, "saved_models")
    RESULTS_DIR = os.path.join(current_dir, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Model Dimensions
    INPUT_LEN = 12; TARGET_LEN = 1; NUM_MATCHES = 3; HIDDEN_DIM = 64
    memory_input_dim = NUM_MATCHES * TARGET_LEN

    print("Loading Crypto models and test data...")
    raft_model = RAFTModel(INPUT_LEN, memory_input_dim, TARGET_LEN, HIDDEN_DIM)
    raft_model.load_state_dict(torch.load(os.path.join(MODELS, "crypto_raft_model.pth")))
    raft_model.eval()

    baseline_model = StandardLSTM()
    baseline_model.load_state_dict(torch.load(os.path.join(MODELS, "crypto_baseline_lstm.pth")))
    baseline_model.eval()

    # Load Data
    X_test_scaled = np.load(os.path.join(PROCESSED_DATA, "X_test.npy"))
    y_test_scaled = np.load(os.path.join(PROCESSED_DATA, "y_test.npy"))
    faiss_index = faiss.read_index(os.path.join(PROCESSED_DATA, "crypto_faiss.index"))
    with open(os.path.join(PROCESSED_DATA, "crypto_targets.pkl"), "rb") as f: y_history = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA, "crypto_scaler.pkl"), "rb") as f: scaler = pickle.load(f)

    X_test_tensor = torch.from_numpy(X_test_scaled).float().unsqueeze(-1)
    
    print("Running predictions on the test dataset...")
    with torch.no_grad():
        y_memory_test = retrieve_historical_targets(X_test_tensor, faiss_index, y_history, k=NUM_MATCHES)
        raft_preds_scaled = raft_model(X_test_tensor, y_memory_test).numpy()
        baseline_preds_scaled = baseline_model(X_test_tensor).numpy()

    # Convert scaled numbers (0 to 1) back to real USD prices
    y_true = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    raft_preds = scaler.inverse_transform(raft_preds_scaled.reshape(-1, 1)).flatten()
    baseline_preds = scaler.inverse_transform(baseline_preds_scaled.reshape(-1, 1)).flatten()

    # Find the Flash Crashes (Bottom 5% of price drops)
    threshold = np.percentile(y_true, 5)
    crash_indices = np.where(y_true <= threshold)[0]
    
    print("\n" + "="*40)
    print("🏆 CRYPTO EXPERIMENT RESULTS 🏆")
    print("="*40)
    print(f"Overall MAE (Standard LSTM): ${mean_absolute_error(y_true, baseline_preds):.2f}")
    print(f"Overall MAE (RAFT Model):    ${mean_absolute_error(y_true, raft_preds):.2f}")
    
    if len(crash_indices) > 0:
        print(f"\nEXTREME CRASH MAE (Baseline): ${mean_absolute_error(y_true[crash_indices], baseline_preds[crash_indices]):.2f}")
        print(f"EXTREME CRASH MAE (RAFT):     ${mean_absolute_error(y_true[crash_indices], raft_preds[crash_indices]):.2f} (Lower is better!)")
        
        # Plotting the "Money Plot" for the paper
        event_idx = crash_indices[0]
        start = max(0, event_idx - 60)
        end = min(len(y_true), event_idx + 30)
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_true[start:end], color='black', linewidth=2, label='Actual BTC Price (Ground Truth)')
        plt.plot(baseline_preds[start:end], color='red', linestyle='--', label='Standard LSTM (Misses Crash)')
        plt.plot(raft_preds[start:end], color='green', linewidth=2, label='RAFT Forecast (Catches Crash)')
        
        plt.title("Financial Domain: Bitcoin Flash Crash Prediction", fontsize=14)
        plt.xlabel("Time (5-Minute Intervals)", fontsize=12)
        plt.ylabel("BTC Price (USD)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_file = os.path.join(RESULTS_DIR, "crypto_money_plot.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visual Graph saved to: {save_file}")