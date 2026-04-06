import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import faiss
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import shared blueprint
from Core.model_definition import RAFTModel

def retrieve_historical_targets(current_input_batch, faiss_index, historical_targets, k=3):
    query_vectors = current_input_batch.view(current_input_batch.size(0), -1).detach().numpy().astype('float32')
    distances, indices = faiss_index.search(query_vectors, k)
    retrieved_y_batch = [historical_targets[indices[i]] for i in range(current_input_batch.size(0))]
    return torch.from_numpy(np.array(retrieved_y_batch)).float()

if __name__ == "__main__":
    PROCESSED_DATA_PATH = os.path.join(current_dir, "data_processed")
    MODEL_SAVE_PATH = os.path.join(current_dir, "saved_models")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    INPUT_LEN = 24; TARGET_LEN = 6; NUM_MATCHES = 3; HIDDEN_DIM = 64
    memory_input_dim = NUM_MATCHES * TARGET_LEN
    
    print("Loading data and Core blueprint...")
    model = RAFTModel(INPUT_LEN, memory_input_dim, TARGET_LEN, HIDDEN_DIM)
    
    faiss_index = faiss.read_index(os.path.join(PROCESSED_DATA_PATH, "river_faiss.index"))
    with open(os.path.join(PROCESSED_DATA_PATH, "river_targets.pkl"), "rb") as f: y_history = pickle.load(f)
    X_train_scaled = np.load(os.path.join(PROCESSED_DATA_PATH, "X_train.npy"))
    
    # Reshape for training
    y_train_scaled = y_history.reshape(y_history.shape[0], TARGET_LEN)
    X_train_tensor = torch.from_numpy(X_train_scaled).float().unsqueeze(-1)
    y_train_tensor = torch.from_numpy(y_train_scaled).float()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    BATCH_SIZE = 64
    EPOCHS = 5 # Fast test run
    
    print(f"Starting River RAFT training for {EPOCHS} epochs...")
    model.train()
    num_samples = X_train_tensor.size(0)
    num_batches = num_samples // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        indices = torch.randperm(num_samples)
        X_shuffled = X_train_tensor[indices]
        y_shuffled = y_train_tensor[indices]
        
        for batch_idx in range(num_batches):
            start = batch_idx * BATCH_SIZE
            end = start + BATCH_SIZE
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            y_memory_batch = retrieve_historical_targets(X_batch, faiss_index, y_history, k=NUM_MATCHES)
            
            optimizer.zero_grad()
            outputs = model(X_batch, y_memory_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/num_batches:.6f}")
        
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "river_raft_model.pth"))
    print(f"✅ River RAFT Model saved.")