"""
Enhanced transformer with user features, item categories, attention visualization, and beam search
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedPositionalEncoding(nn.Module):
    """Enhanced positional encoding with learnable components"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Standard sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
        # Learnable position embeddings for fine-tuning
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        # Combine sinusoidal and learnable encodings
        pos_encoding = self.pe[:, :seq_len, :] + self.learnable_pe[:, :seq_len, :]
        x = x + pos_encoding
        return self.dropout(x)

class MultiHeadAttentionWithVisualization(nn.Module):
    """Multi-head attention with attention weight extraction for visualization"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Store for visualization
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights.detach()  # Store for visualization
        
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.w_o(context)
        return output

class EnhancedTransformerRecommender(nn.Module):
    """Enhanced transformer with user features, item categories, and attention visualization"""
    
    def __init__(self, 
                 n_items: int,
                 n_users: int,
                 n_categories: int = 10,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 d_ff: int = 256,
                 max_seq_len: int = 50,
                 dropout: float = 0.1,
                 pad_token: int = 0):
        super().__init__()
        
        self.n_items = n_items
        self.n_users = n_users
        self.n_categories = n_categories
        self.d_model = d_model
        self.pad_token = pad_token
        
        # Enhanced embeddings
        self.item_embedding = nn.Embedding(n_items + 1, d_model, padding_idx=pad_token)
        self.user_embedding = nn.Embedding(n_users + 1, d_model, padding_idx=0)
        self.category_embedding = nn.Embedding(n_categories + 1, d_model // 4, padding_idx=0)
        
        # Position in sequence embedding (recency matters in recommendations)
        self.position_embedding = nn.Embedding(max_seq_len, d_model // 4)
        
        # Combine different features
        self.feature_projection = nn.Linear(d_model + d_model // 4 + d_model // 4, d_model)
        
        # Enhanced positional encoding
        self.pos_encoding = EnhancedPositionalEncoding(d_model, max_seq_len, dropout)
        
        # Input normalization
        self.input_norm = nn.LayerNorm(d_model)
        
        # Custom transformer layers with attention visualization
        self.attention_layers = nn.ModuleList([
            MultiHeadAttentionWithVisualization(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        
        # Enhanced output projection with category prediction
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        
        # Multi-task outputs
        self.item_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_items)
        )
        
        # Auxiliary category prediction (helps with generalization)
        self.category_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, n_categories)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Enhanced weight initialization"""
        # Embeddings
        for embedding in [self.item_embedding, self.user_embedding, self.category_embedding, self.position_embedding]:
            nn.init.normal_(embedding.weight, mean=0, std=0.02)
        
        # Set padding embeddings to zero
        self.item_embedding.weight.data[self.pad_token].fill_(0)
        self.user_embedding.weight.data[0].fill_(0)
        self.category_embedding.weight.data[0].fill_(0)
        
        # Linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask"""
        return (x == self.pad_token)
    
    def get_item_categories(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Generate pseudo-categories based on item IDs (in real scenario, use actual categories)"""
        # Simple hash-based category assignment for demo
        categories = (item_ids % self.n_categories).clamp(min=0)
        categories[item_ids == self.pad_token] = 0  # Padding category
        return categories
    
    def forward(self, x: torch.Tensor, user_ids: torch.Tensor = None, 
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with multi-task learning
        
        Args:
            x: Item sequences (batch_size, seq_len)
            user_ids: User IDs (batch_size,)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with item_logits, category_logits, and optionally attention_weights
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Create masks
        padding_mask = self.create_padding_mask(x)
        
        # Get embeddings
        item_embeds = self.item_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Add user context (broadcast user embedding to all positions)
        if user_ids is not None:
            user_embeds = self.user_embedding(user_ids).unsqueeze(1).expand(-1, seq_len, -1)
        else:
            user_embeds = torch.zeros_like(item_embeds)
        
        # Get category embeddings
        categories = self.get_item_categories(x)
        category_embeds = self.category_embedding(categories)  # (batch_size, seq_len, d_model//4)
        
        # Position embeddings (recency in sequence)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)  # (batch_size, seq_len, d_model//4)
        
        # Combine all features
        combined_features = torch.cat([
            item_embeds + user_embeds,  # Item + user context
            category_embeds,            # Category information
            position_embeds            # Position in sequence
        ], dim=-1)
        
        # Project to model dimension
        embedded = self.feature_projection(combined_features)
        embedded = embedded * (math.sqrt(self.d_model) * 0.1)  # Controlled scaling
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        embedded = self.input_norm(embedded)
        
        # Apply transformer layers
        hidden = embedded
        attention_weights_list = []
        
        for i in range(len(self.attention_layers)):
            # Self-attention with residual connection
            attn_output = self.attention_layers[i](hidden, hidden, hidden, padding_mask)
            hidden = self.layer_norms1[i](hidden + attn_output)
            
            # Store attention weights for visualization
            if return_attention:
                attention_weights_list.append(self.attention_layers[i].attention_weights)
            
            # Feed-forward with residual connection
            ff_output = self.feed_forward_layers[i](hidden)
            hidden = self.layer_norms2[i](hidden + ff_output)
        
        # Global pooling (mean of non-padded positions)
        mask = (~padding_mask).float().unsqueeze(-1)
        masked_hidden = hidden * mask
        seq_lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled_output = masked_hidden.sum(dim=1) / seq_lengths
        
        # Output projections
        output = self.output_norm(pooled_output)
        output = self.output_dropout(output)
        
        # Multi-task outputs
        item_logits = self.item_projection(output)
        category_logits = self.category_projection(output)
        
        result = {
            'item_logits': item_logits,
            'category_logits': category_logits
        }
        
        if return_attention:
            result['attention_weights'] = attention_weights_list
        
        return result
    
    def beam_search_predict(self, x: torch.Tensor, user_ids: torch.Tensor = None, 
                           k: int = 10, beam_width: int = 5, exclude_seen: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search for better top-k predictions
        
        Args:
            x: Input sequences
            user_ids: User IDs
            k: Number of final predictions
            beam_width: Beam search width
            exclude_seen: Exclude items in input sequence
            
        Returns:
            Top-k items and their scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, user_ids)
            logits = outputs['item_logits']
            
            batch_size = logits.size(0)
            
            # Apply exclusions
            if exclude_seen:
                for i, seq in enumerate(x):
                    seen_items = seq[seq != self.pad_token]
                    logits[i, seen_items] = float('-inf')
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Beam search: get top beam_width candidates
            beam_scores, beam_items = torch.topk(probs, min(beam_width, k * 2), dim=-1)
            
            # Re-rank using category consistency
            final_items = []
            final_scores = []
            
            for batch_idx in range(batch_size):
                # Get category predictions for context
                category_probs = F.softmax(outputs['category_logits'][batch_idx], dim=-1)
                top_category = torch.argmax(category_probs)
                
                # Score items based on item probability and category consistency
                item_candidates = beam_items[batch_idx]
                item_scores = beam_scores[batch_idx]
                
                # Get categories for candidate items
                candidate_categories = self.get_item_categories(item_candidates.unsqueeze(0)).squeeze(0)
                
                # Boost scores for items in predicted category
                category_boost = (candidate_categories == top_category).float() * 0.1
                adjusted_scores = item_scores + category_boost
                
                # Final top-k selection
                _, top_indices = torch.topk(adjusted_scores, min(k, len(adjusted_scores)))
                
                final_items.append(item_candidates[top_indices])
                final_scores.append(adjusted_scores[top_indices])
            
            return torch.stack(final_items), torch.stack(final_scores)
    
    def visualize_attention(self, x: torch.Tensor, user_ids: torch.Tensor = None, 
                           save_path: str = None) -> None:
        """
        Visualize attention weights
        
        Args:
            x: Input sequence (single sample)
            user_ids: User ID (single sample)
            save_path: Path to save visualization
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if user_ids is not None and user_ids.dim() == 0:
                user_ids = user_ids.unsqueeze(0)
            
            outputs = self.forward(x, user_ids, return_attention=True)
            attention_weights = outputs['attention_weights']
            
            # Get non-padded sequence
            seq = x[0]
            valid_positions = (seq != self.pad_token).nonzero().squeeze(-1)
            valid_seq = seq[valid_positions]
            
            if len(valid_positions) < 2:
                print("Sequence too short for attention visualization")
                return
            
            # Create visualization
            n_layers = len(attention_weights)
            n_heads = attention_weights[0].size(1)
            
            fig, axes = plt.subplots(n_layers, min(n_heads, 4), figsize=(16, 4 * n_layers))
            if n_layers == 1:
                axes = axes.reshape(1, -1)
            
            for layer_idx in range(n_layers):
                layer_attention = attention_weights[layer_idx][0]  # First batch item
                
                for head_idx in range(min(n_heads, 4)):
                    ax = axes[layer_idx, head_idx] if n_layers > 1 else axes[head_idx]
                    
                    # Extract attention for valid positions
                    head_attention = layer_attention[head_idx]
                    valid_attention = head_attention[valid_positions][:, valid_positions]
                    
                    # Plot heatmap
                    sns.heatmap(
                        valid_attention.cpu().numpy(),
                        ax=ax,
                        cmap='Blues',
                        xticklabels=[f'Item_{item.item()}' for item in valid_seq],
                        yticklabels=[f'Item_{item.item()}' for item in valid_seq],
                        cbar=True
                    )
                    
                    ax.set_title(f'Layer {layer_idx + 1}, Head {head_idx + 1}')
                    ax.set_xlabel('Key Positions')
                    ax.set_ylabel('Query Positions')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Attention visualization saved to {save_path}")
            else:
                plt.show()

class EnhancedTransformerTrainer:
    """Enhanced trainer with multi-task learning"""
    
    def __init__(self, model: EnhancedTransformerRecommender, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_step(self, batch_sequences: torch.Tensor, batch_labels: torch.Tensor,
                  batch_users: torch.Tensor, optimizer: torch.optim.Optimizer, 
                  item_criterion: nn.Module, category_criterion: nn.Module) -> Tuple[float, float, float]:
        """Enhanced training step with multi-task learning"""
        self.model.train()
        
        # Move to device
        batch_sequences = batch_sequences.to(self.device)
        batch_labels = batch_labels.to(self.device)
        batch_users = batch_users.to(self.device)
        
        try:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_sequences, batch_users)
            item_logits = outputs['item_logits']
            category_logits = outputs['category_logits']
            
            # Check for numerical issues
            if torch.isnan(item_logits).any() or torch.isinf(item_logits).any():
                return float('nan'), float('nan'), float('nan')
            
            # Multi-task losses
            item_loss = item_criterion(item_logits, batch_labels)
            
            # Category labels (derived from item labels)
            category_labels = self.model.get_item_categories(batch_labels.unsqueeze(1)).squeeze(1)
            category_loss = category_criterion(category_logits, category_labels)
            
            # Combined loss (item prediction is primary task)
            total_loss = item_loss + 0.3 * category_loss  # Category loss weight
            
            # Check losses
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                return float('nan'), float('nan'), float('nan')
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            optimizer.step()
            
            return total_loss.item(), item_loss.item(), category_loss.item()
            
        except Exception as e:
            print(f"Error in training step: {e}")
            return float('nan'), float('nan'), float('nan')
    
    def evaluate(self, test_loader, item_criterion: nn.Module, 
                category_criterion: nn.Module) -> Tuple[float, float, float, float]:
        """Enhanced evaluation with multi-task metrics"""
        self.model.eval()
        total_loss = 0
        item_loss_total = 0
        category_loss_total = 0
        correct_predictions = 0
        total_samples = 0
        valid_batches = 0
        
        with torch.no_grad():
            for batch_sequences, batch_labels, batch_users in test_loader:
                try:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    batch_users = batch_users.to(self.device)
                    
                    outputs = self.model(batch_sequences, batch_users)
                    item_logits = outputs['item_logits']
                    category_logits = outputs['category_logits']
                    
                    # Skip if numerical issues
                    if torch.isnan(item_logits).any() or torch.isinf(item_logits).any():
                        continue
                    
                    # Losses
                    item_loss = item_criterion(item_logits, batch_labels)
                    category_labels = self.model.get_item_categories(batch_labels.unsqueeze(1)).squeeze(1)
                    category_loss = category_criterion(category_logits, category_labels)
                    total_batch_loss = item_loss + 0.3 * category_loss
                    
                    if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                        continue
                    
                    total_loss += total_batch_loss.item()
                    item_loss_total += item_loss.item()
                    category_loss_total += category_loss.item()
                    valid_batches += 1
                    
                    # Accuracy
                    predictions = torch.argmax(item_logits, dim=1)
                    correct_predictions += (predictions == batch_labels).sum().item()
                    total_samples += batch_labels.size(0)
                    
                except Exception:
                    continue
        
        if valid_batches == 0:
            return float('nan'), float('nan'), float('nan'), 0.0
        
        avg_total_loss = total_loss / valid_batches
        avg_item_loss = item_loss_total / valid_batches
        avg_category_loss = category_loss_total / valid_batches
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return avg_total_loss, avg_item_loss, avg_category_loss, accuracy
