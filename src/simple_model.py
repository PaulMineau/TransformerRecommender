"""
Simplified and more stable recommendation model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

class SimpleRecommender(nn.Module):
    """Simplified recommendation model with better stability"""
    
    def __init__(self, 
                 n_items: int,
                 embedding_dim: int = 64,
                 hidden_dim: int = 128,
                 max_seq_len: int = 20,
                 dropout: float = 0.2,
                 pad_token: int = 0):
        super().__init__()
        
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.pad_token = pad_token
        self.max_seq_len = max_seq_len
        
        # Item embedding (add 1 for padding token)
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=pad_token)
        
        # Simple LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_items)
        
        # Initialize weights properly
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights with proper scaling"""
        # Initialize embeddings
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        self.item_embedding.weight.data[self.pad_token].fill_(0)
        
        # Initialize LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize linear layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input sequences of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, n_items)
        """
        batch_size, seq_len = x.shape
        
        # Create mask for padding tokens
        mask = (x != self.pad_token).float()
        
        # Embed items
        embedded = self.item_embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Apply mask to embeddings
        embedded = embedded * mask.unsqueeze(-1)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        # Take the last non-padded position for each sequence
        seq_lengths = mask.sum(dim=1).long() - 1  # -1 for 0-indexing
        seq_lengths = torch.clamp(seq_lengths, min=0, max=seq_len-1)
        
        batch_indices = torch.arange(batch_size, device=x.device)
        last_outputs = lstm_out[batch_indices, seq_lengths]
        
        # Apply dropout and fully connected layers
        out = self.dropout(last_outputs)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits
    
    def predict_top_k(self, x: torch.Tensor, k: int = 10, 
                     exclude_seen: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k items for given sequences
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            
            if exclude_seen:
                # Mask out items that appear in the input sequence
                for i, seq in enumerate(x):
                    seen_items = seq[seq != self.pad_token]
                    logits[i, seen_items] = float('-inf')
            
            # Get top-k predictions
            top_k_scores, top_k_items = torch.topk(logits, k, dim=1)
            
        return top_k_items, top_k_scores

class SimpleTrainer:
    """Simple trainer for the recommendation model"""
    
    def __init__(self, model: SimpleRecommender, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_step(self, batch_sequences: torch.Tensor, batch_labels: torch.Tensor,
                  optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Single training step with stability checks"""
        self.model.train()
        
        # Move to device
        batch_sequences = batch_sequences.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = self.model(batch_sequences)
        
        # Check for numerical issues
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN or Inf in logits, skipping batch")
            return float('nan')
        
        # Compute loss
        loss = criterion(logits, batch_labels)
        
        # Check loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN or Inf in loss, skipping batch")
            return float('nan')
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, test_loader, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        valid_batches = 0
        
        with torch.no_grad():
            for batch_sequences, batch_labels in test_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                logits = self.model(batch_sequences)
                
                # Skip batch if numerical issues
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    continue
                
                loss = criterion(logits, batch_labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                total_loss += loss.item()
                valid_batches += 1
                
                # Calculate accuracy (top-1)
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == batch_labels).sum().item()
                total_samples += batch_labels.size(0)
        
        if valid_batches == 0:
            return float('nan'), 0.0
        
        avg_loss = total_loss / valid_batches
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
