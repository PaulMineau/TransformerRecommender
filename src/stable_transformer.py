"""
Stable transformer-based recommendation model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class StablePositionalEncoding(nn.Module):
    """Stable positional encoding implementation"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings"""
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class StableTransformerRecommender(nn.Module):
    """Stable transformer-based recommendation model"""
    
    def __init__(self, 
                 n_items: int,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 3,  # Fewer layers for stability
                 d_ff: int = 256,    # Smaller feedforward
                 max_seq_len: int = 50,
                 dropout: float = 0.1,
                 pad_token: int = 0):
        super().__init__()
        
        self.n_items = n_items
        self.d_model = d_model
        self.pad_token = pad_token
        
        # Item embedding with proper scaling
        self.item_embedding = nn.Embedding(n_items + 1, d_model, padding_idx=pad_token)
        self.embedding_scale = math.sqrt(d_model)
        
        # Positional encoding
        self.pos_encoding = StablePositionalEncoding(d_model, max_seq_len, dropout)
        
        # Input layer norm
        self.input_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder with proper configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',  # GELU is more stable than ReLU
            batch_first=True,
            norm_first=True     # Pre-norm is more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection with intermediate layer
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        
        # Two-layer output projection for better stability
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, n_items)
        
        # Initialize weights properly
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights with proper scaling"""
        # Embedding initialization
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.02)
        self.item_embedding.weight.data[self.pad_token].fill_(0)
        
        # Linear layer initialization
        for module in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(module.weight, gain=0.02)  # Small gain for stability
            nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask for transformer"""
        return (x == self.pad_token)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with stability checks
        
        Args:
            x: Input sequences of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, n_items)
        """
        batch_size, seq_len = x.shape
        
        # Create padding mask
        padding_mask = self.create_padding_mask(x)
        
        # Embed items with controlled scaling
        embedded = self.item_embedding(x)
        embedded = embedded * (self.embedding_scale * 0.1)  # Reduced scaling factor
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        embedded = self.input_norm(embedded)
        
        # Apply transformer
        transformer_out = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # Use mean pooling over valid positions (more stable than last position)
        mask = (~padding_mask).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
        masked_out = transformer_out * mask
        seq_lengths = mask.sum(dim=1)  # (batch_size, 1)
        
        # Avoid division by zero
        seq_lengths = torch.clamp(seq_lengths, min=1.0)
        pooled_output = masked_out.sum(dim=1) / seq_lengths  # (batch_size, d_model)
        
        # Output projection with normalization
        output = self.output_norm(pooled_output)
        output = self.output_dropout(output)
        
        # Two-layer projection
        output = F.gelu(self.fc1(output))
        output = self.output_dropout(output)
        logits = self.fc2(output)
        
        return logits
    
    def predict_top_k(self, x: torch.Tensor, k: int = 10, 
                     exclude_seen: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict top-k items for given sequences"""
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

class StableTransformerTrainer:
    """Trainer for stable transformer model"""
    
    def __init__(self, model: StableTransformerRecommender, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_step(self, batch_sequences: torch.Tensor, batch_labels: torch.Tensor,
                  optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Single training step with extensive stability checks"""
        self.model.train()
        
        # Move to device
        batch_sequences = batch_sequences.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            logits = self.model(batch_sequences)
            
            # Check for numerical issues in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Warning: NaN/Inf in logits")
                return float('nan')
            
            # Check logits magnitude
            if logits.abs().max() > 100:
                print(f"Warning: Large logits detected: {logits.abs().max():.2f}")
            
            # Compute loss
            loss = criterion(logits, batch_labels)
            
            # Check loss
            if torch.isnan(loss) or torch.isinf(loss) or loss > 50:
                print(f"Warning: Problematic loss: {loss.item()}")
                return float('nan')
            
            # Backward pass with aggressive gradient clipping
            loss.backward()
            
            # Check gradients
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            if total_norm > 10:
                print(f"Warning: Large gradient norm: {total_norm:.2f}")
            
            optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            print(f"Error in training step: {e}")
            return float('nan')
    
    def evaluate(self, test_loader, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        valid_batches = 0
        
        with torch.no_grad():
            for batch_sequences, batch_labels in test_loader:
                try:
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
                    
                    # Calculate accuracy
                    predictions = torch.argmax(logits, dim=1)
                    correct_predictions += (predictions == batch_labels).sum().item()
                    total_samples += batch_labels.size(0)
                    
                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    continue
        
        if valid_batches == 0:
            return float('nan'), 0.0
        
        avg_loss = total_loss / valid_batches
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
