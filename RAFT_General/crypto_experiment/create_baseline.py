import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))

class StandardLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super(StandardLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.output_head(hidden[-1])

if __name__ == "__main__":
    PROCESSED_DATA_PATH = os.path.join(current_dir, "data_processed")
    MODEL_SAVE_PATH = os.path.join(current_dir, "saved_models")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    TARGET_LEN = 1

    print("Loading scaled data for baseline...")
    X_train_scaled = np.load(os.path.join(PROCESSED_DATA_PATH, "X_train.npy"))
    with open(os.path.join(PROCESSED_DATA_PATH, "crypto_targets.pkl"), "rb") as f: y_history = pickle.load(f)
    
    y_train_scaled = y_history.reshape(y_history.shape[0], TARGET_LEN, 1)

    X_train_tensor = torch.from_numpy(X_train_scaled).float().unsqueeze(-1)
    y_train_tensor = torch.from_numpy(y_train_scaled).float().squeeze(-1)

    baseline_model = StandardLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    
    BATCH_SIZE = 64
    EPOCHS = 5 

    print("Starting Standard LSTM baseline training...")
    baseline_model.train()
    
    num_samples = X_train_tensor.size(0)
    num_batches = num_samples // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        indices = torch.randperm(num_samples)
        X_shuffled = X_train_tensor[indices]
        y_shuffled = y_train_tensor[indices]
        
        for batch_idx in range(num_batches):
            start = batch_idx * BATCH_SIZE
            end = start + BATCH_SIZE
            
            optimizer.zero_grad()
            outputs = baseline_model(X_shuffled[start:end]) 
            loss = criterion(outputs, y_shuffled[start:end])
            loss.backward()
            optimizer.step()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")
        
    torch.save(baseline_model.state_dict(), os.path.join(MODEL_SAVE_PATH, "crypto_baseline_lstm.pth"))
    print(f"✅ Baseline LSTM saved.")