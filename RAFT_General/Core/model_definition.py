# SAVE THIS FILE AS: RAFT_Project/core/model_definition.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RAFTModel(nn.Module):
    """
    Modular RAFT (Retrieval-Augmented Forecasting) Model.
    Designed to be generic and reusable across different time series domains.
    """
    def __init__(self, current_input_dim, memory_input_dim, output_dim, hidden_dim=64):
        """
        Args:
            current_input_dim (int): Length of recent time window (e.g., 24 for 24h)
            memory_input_dim (int): Total size of retrieved historical target data
                                    (e.g., num_matches * target_len)
            output_dim (int): Length of future forecast window (e.g., 6 for 6h)
            hidden_dim (int): Number of units in hidden layers
        """
        super(RAFTModel, self).__init__()
        
        # HEAD 1: Current Data Encoder (Processes recent window, e.g., 24h)
        # Using LSTM to encode temporal patterns
        self.lstm_current = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        
        # HEAD 2: Memory Data Encoder (Processes retrieved historical targets)
        # Using Dense (Linear) layers for simpler processing of past outcomes
        self.dense_memory = nn.Sequential(
            nn.Linear(memory_input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # FUSION & FORECASTING LAYERS
        # Merges outputs from Head 1 and Head 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            # Final output layer to produce multi-step forecast
            nn.Linear(hidden_dim * 2, output_dim) 
        )

    def forward(self, current_data_window, retrieved_historical_values):
        """
        Args:
            current_data_window (torch.Tensor): Shape [Batch, current_input_dim, 1]
            retrieved_historical_values (torch.Tensor): Shape [Batch, num_matches, output_dim]
        """
        # 1. Encode Current Data -> get final LSTM hidden state
        # X shape: [Batch, current_input_dim, 1] -> [Batch, Hidden]
        _, (hidden_current, _) = self.lstm_current(current_data_window)
        encoded_current = hidden_current[-1] # Shape [Batch, Hidden]
        
        # 2. Encode Retrieved Memory -> flatten and pass through dense layer
        # [Batch, num_matches, output_dim] -> [Batch, Flattened_Memory_Size]
        flat_memories = retrieved_historical_values.view(retrieved_historical_values.size(0), -1)
        encoded_memory = self.dense_memory(flat_memories) # Shape [Batch, Hidden]
        
        # 3. Concatenate and Forecast
        # Shape [Batch, Hidden*2]
        combined_context = torch.cat((encoded_current, encoded_memory), dim=1)
        
        final_forecast = self.fusion_layer(combined_context)
        return final_forecast