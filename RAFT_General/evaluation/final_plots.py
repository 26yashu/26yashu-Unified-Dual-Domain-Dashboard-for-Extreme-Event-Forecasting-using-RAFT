import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import faiss

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Core.model_definition import RAFTModel

# Re-define Baseline architecture
class StandardLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=6): # Notice output_dim is 6 for River!
        super(StandardLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.output_head(hidden[-1])

def retrieve_historical_targets(current_input_batch, faiss_index, historical_targets, k=3):
    query_vectors = current_input_batch.view(current_input_batch.size(0), -1).detach().numpy().astype('float32')
    distances, indices = faiss_index.search(query_vectors, k)
    retrieved_y_batch = [historical_targets[indices[i]] for i in range(current_input_batch.size(0))]
    return torch.from_numpy(np.array(retrieved_y_batch)).float()

if __name__ == "__main__":
    RIVER_DIR = os.path.join(parent_dir, "river_experiment")
    PROCESSED_DATA = os.path.join(RIVER_DIR, "data_processed")
    MODELS = os.path.join(RIVER_DIR, "saved_models")
    RESULTS_DIR = os.path.join(current_dir, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # River Dimensions
    INPUT_LEN = 24; TARGET_LEN = 6; NUM_MATCHES = 3; HIDDEN_DIM = 64
    memory_input_dim = NUM_MATCHES * TARGET_LEN

    print("Loading River models and test data...")
    raft_model = RAFTModel(INPUT_LEN, memory_input_dim, TARGET_LEN, HIDDEN_DIM)
    raft_model.load_state_dict(torch.load(os.path.join(MODELS, "river_raft_model.pth")))
    raft_model.eval()

    baseline_model = StandardLSTM(output_dim=TARGET_LEN)
    baseline_model.load_state_dict(torch.load(os.path.join(MODELS, "river_baseline_lstm.pth")))
    baseline_model.eval()

    X_test_scaled = np.load(os.path.join(PROCESSED_DATA, "X_test.npy"))
    y_test_scaled = np.load(os.path.join(PROCESSED_DATA, "y_test.npy"))
    faiss_index = faiss.read_index(os.path.join(PROCESSED_DATA, "river_faiss.index"))
    with open(os.path.join(PROCESSED_DATA, "river_targets.pkl"), "rb") as f: y_history = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA, "river_scaler.pkl"), "rb") as f: scaler = pickle.load(f)

    X_test_tensor = torch.from_numpy(X_test_scaled).float().unsqueeze(-1)
    
    print("Running predictions on the test dataset...")
    with torch.no_grad():
        y_memory_test = retrieve_historical_targets(X_test_tensor, faiss_index, y_history, k=NUM_MATCHES)
        raft_preds_scaled = raft_model(X_test_tensor, y_memory_test).numpy()
        baseline_preds_scaled = baseline_model(X_test_tensor).numpy()

    # Inverse transform
    y_true_flat = scaler.inverse_transform(y_test_scaled.reshape(-1, 1))
    y_true = y_true_flat.reshape(y_test_scaled.shape[0], TARGET_LEN)
    raft_preds = scaler.inverse_transform(raft_preds_scaled.reshape(-1, 1)).reshape(raft_preds_scaled.shape[0], TARGET_LEN)
    baseline_preds = scaler.inverse_transform(baseline_preds_scaled.reshape(-1, 1)).reshape(baseline_preds_scaled.shape[0], TARGET_LEN)

    # For the graph, we usually just plot the 1st hour of the 6-hour forecast
    y_true_1h = y_true[:, 0]
    raft_1h = raft_preds[:, 0]
    base_1h = baseline_preds[:, 0]

    # Find the Flash Floods (Top 5% highest water levels)
    threshold = np.percentile(y_true_1h, 95)
    flood_indices = np.where(y_true_1h >= threshold)[0]
    
    print("\n" + "="*40)
    print("🏆 HYDROLOGY EXPERIMENT RESULTS 🏆")
    print("="*40)
    print(f"Overall MAE (Standard LSTM): {mean_absolute_error(y_true_1h, base_1h):.2f} ft")
    print(f"Overall MAE (RAFT Model):    {mean_absolute_error(y_true_1h, raft_1h):.2f} ft")
    
    if len(flood_indices) > 0:
        # Plotting the "Money Plot" for the paper
        event_idx = flood_indices[0]
        start = max(0, event_idx - 72) # Look 3 days before
        end = min(len(y_true_1h), event_idx + 48) # Look 2 days after
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_true_1h[start:end], color='black', linewidth=2, label='Actual River Level (Ground Truth)')
        plt.plot(base_1h[start:end], color='red', linestyle='--', label='Standard LSTM (Underestimates Flood)')
        plt.plot(raft_1h[start:end], color='blue', linewidth=2, label='RAFT Forecast (Catches Flood Spike)')
        
        plt.title("Hydrology Domain: Flash Flood Prediction", fontsize=14)
        plt.xlabel("Time (Hourly)", fontsize=12)
        plt.ylabel("River Gage Height (Feet)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_file = os.path.join(RESULTS_DIR, "river_money_plot.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visual Graph saved to: {save_file}")

