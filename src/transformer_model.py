"""
Transformer-based recommendation model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class TransformerRecommender(nn.Module):
    """Transformer-based recommendation model"""
    
    def __init__(self, 
                 n_items: int,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 d_ff: int = 512,
                 max_seq_len: int = 50,
                 dropout: float = 0.1,
                 pad_token: int = 0):
        super().__init__()
        
        self.n_items = n_items
        self.d_model = d_model
        self.pad_token = pad_token
        
        # Item embedding (add 1 for padding token)
        self.item_embedding = nn.Embedding(n_items + 1, d_model, padding_idx=pad_token)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, n_items)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        initrange = 0.02  # Smaller initialization range
        self.item_embedding.weight.data.uniform_(-initrange, initrange)
        # Set padding token embedding to zero
        self.item_embedding.weight.data[self.pad_token].fill_(0)
        
        self.output_projection.bias.data.zero_()
        nn.init.xavier_uniform_(self.output_projection.weight)  # Better initialization
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask for transformer"""
        return (x == self.pad_token)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input sequences of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, n_items)
        """
        # Create padding mask
        padding_mask = self.create_padding_mask(x)
        
        # Embed items
        embedded = self.item_embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded.transpose(0, 1)).transpose(0, 1)
        embedded = self.dropout(embedded)
        
        # Apply transformer
        transformer_out = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # Use the last non-padded position for each sequence
        batch_size, seq_len, d_model = transformer_out.shape
        
        # Find last non-padded position for each sequence
        seq_lengths = (~padding_mask).sum(dim=1)  # Number of valid tokens
        seq_lengths = torch.clamp(seq_lengths - 1, min=0, max=seq_len-1)  # Convert to indices
        
        # Ensure we have valid indices
        batch_indices = torch.arange(batch_size, device=x.device)
        
        # Gather last valid outputs
        last_outputs = transformer_out[batch_indices, seq_lengths]
        
        # Project to item space
        logits = self.output_projection(last_outputs)
        
        return logits
    
    def predict_top_k(self, x: torch.Tensor, k: int = 10, 
                     exclude_seen: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k items for given sequences
        
        Args:
            x: Input sequences of shape (batch_size, seq_len)
            k: Number of top items to return
            exclude_seen: Whether to exclude items already in the sequence
            
        Returns:
            Tuple of (top_k_items, top_k_scores)
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

class RecommenderTrainer:
    """Training utilities for the transformer recommender"""
    
    def __init__(self, model: TransformerRecommender, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_step(self, batch_sequences: torch.Tensor, batch_labels: torch.Tensor,
                  optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Single training step"""
        self.model.train()
        
        # Move to device
        batch_sequences = batch_sequences.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = self.model(batch_sequences)
        
        # Check for NaN or Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN or Inf detected in model output!")
            return float('nan')
        
        loss = criterion(logits, batch_labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, test_loader, criterion: nn.Module) -> Tuple[float, float, float]:
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_sequences, batch_labels in test_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                logits = self.model(batch_sequences)
                loss = criterion(logits, batch_labels)
                
                total_loss += loss.item()
                
                # Calculate accuracy (top-1)
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == batch_labels).sum().item()
                total_samples += batch_labels.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy, total_samples

def calculate_metrics(model: TransformerRecommender, test_loader, device: str = 'cpu', 
                     k_values: list = [1, 5, 10]) -> dict:
    """Calculate recommendation metrics (Hit Rate, NDCG, MRR)"""
    model.eval()
    
    hit_rates = {k: 0 for k in k_values}
    ndcg_scores = {k: 0 for k in k_values}
    mrr_scores = []
    total_samples = 0
    
    with torch.no_grad():
        for batch_sequences, batch_labels in test_loader:
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)
            
            # Get top-k predictions
            max_k = max(k_values)
            top_k_items, top_k_scores = model.predict_top_k(batch_sequences, k=max_k)
            
            batch_size = batch_labels.size(0)
            
            for i in range(batch_size):
                true_item = batch_labels[i].item()
                predictions = top_k_items[i].cpu().numpy()
                
                # Calculate metrics for different k values
                for k in k_values:
                    top_k_preds = predictions[:k]
                    
                    # Hit Rate
                    if true_item in top_k_preds:
                        hit_rates[k] += 1
                    
                    # NDCG
                    if true_item in top_k_preds:
                        rank = list(top_k_preds).index(true_item) + 1
                        ndcg_scores[k] += 1.0 / math.log2(rank + 1)
                
                # MRR (Mean Reciprocal Rank)
                if true_item in predictions:
                    rank = list(predictions).index(true_item) + 1
                    mrr_scores.append(1.0 / rank)
                else:
                    mrr_scores.append(0.0)
                
                total_samples += 1
    
    # Calculate final metrics
    metrics = {}
    for k in k_values:
        metrics[f'hit_rate@{k}'] = hit_rates[k] / total_samples
        metrics[f'ndcg@{k}'] = ndcg_scores[k] / total_samples
    
    metrics['mrr'] = sum(mrr_scores) / len(mrr_scores)
    
    return metrics
