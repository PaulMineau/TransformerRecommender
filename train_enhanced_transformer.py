"""
Training script for enhanced transformer with user features, categories, attention visualization, and beam search
"""
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.enhanced_transformer import EnhancedTransformerRecommender, EnhancedTransformerTrainer

def create_enhanced_data_loaders(train_data, test_data, batch_size=32):
    """Create data loaders with user information"""
    
    # Convert sequences and labels
    train_sequences = torch.LongTensor(train_data['sequences'])
    train_labels = torch.LongTensor(train_data['labels'])
    train_users = torch.LongTensor(train_data['user_ids'])
    
    test_sequences = torch.LongTensor(test_data['sequences'])
    test_labels = torch.LongTensor(test_data['labels'])
    test_users = torch.LongTensor(test_data['user_ids'])
    
    # Create datasets
    train_dataset = TensorDataset(train_sequences, train_labels, train_users)
    test_dataset = TensorDataset(test_sequences, test_labels, test_users)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def calculate_enhanced_metrics(model, test_loader, device='cpu', k_values=[5, 10, 20]):
    """Calculate enhanced metrics including beam search performance"""
    model.eval()
    
    # Standard metrics
    hit_rates = {k: 0 for k in k_values}
    beam_hit_rates = {k: 0 for k in k_values}
    ndcg_scores = {k: 0 for k in k_values}
    beam_ndcg_scores = {k: 0 for k in k_values}
    
    total_samples = 0
    
    print("Calculating enhanced metrics...")
    with torch.no_grad():
        for batch_sequences, batch_labels, batch_users in tqdm(test_loader, desc="Evaluating"):
            try:
                batch_sequences = batch_sequences.to(device)
                batch_labels = batch_labels.to(device)
                batch_users = batch_users.to(device)
                
                batch_size = batch_labels.size(0)
                
                # Standard predictions
                outputs = model(batch_sequences, batch_users)
                logits = outputs['item_logits']
                
                # Exclude seen items
                for i, seq in enumerate(batch_sequences):
                    seen_items = seq[seq != model.pad_token]
                    logits[i, seen_items] = float('-inf')
                
                # Beam search predictions
                beam_items, beam_scores = model.beam_search_predict(
                    batch_sequences, batch_users, k=max(k_values), beam_width=10
                )
                
                for i in range(batch_size):
                    true_item = batch_labels[i].item()
                    
                    # Standard predictions
                    _, top_k_items = torch.topk(logits[i], max(k_values))
                    standard_preds = top_k_items.cpu().numpy()
                    
                    # Beam search predictions
                    beam_preds = beam_items[i].cpu().numpy()
                    
                    # Calculate metrics for different k values
                    for k in k_values:
                        # Standard Hit Rate and NDCG
                        std_top_k = standard_preds[:k]
                        if true_item in std_top_k:
                            hit_rates[k] += 1
                            rank = list(std_top_k).index(true_item) + 1
                            ndcg_scores[k] += 1.0 / np.log2(rank + 1)
                        
                        # Beam search Hit Rate and NDCG
                        beam_top_k = beam_preds[:k]
                        if true_item in beam_top_k:
                            beam_hit_rates[k] += 1
                            rank = list(beam_top_k).index(true_item) + 1
                            beam_ndcg_scores[k] += 1.0 / np.log2(rank + 1)
                    
                    total_samples += 1
                    
            except Exception as e:
                print(f"Error in metric calculation: {e}")
                continue
    
    # Calculate final metrics
    metrics = {}
    for k in k_values:
        metrics[f'hit_rate@{k}'] = hit_rates[k] / total_samples if total_samples > 0 else 0
        metrics[f'beam_hit_rate@{k}'] = beam_hit_rates[k] / total_samples if total_samples > 0 else 0
        metrics[f'ndcg@{k}'] = ndcg_scores[k] / total_samples if total_samples > 0 else 0
        metrics[f'beam_ndcg@{k}'] = beam_ndcg_scores[k] / total_samples if total_samples > 0 else 0
    
    return metrics

def visualize_sample_attention(model, test_loader, encoders, device='cpu'):
    """Visualize attention for a sample sequence"""
    model.eval()
    
    # Get a sample
    for batch_sequences, batch_labels, batch_users in test_loader:
        batch_sequences = batch_sequences.to(device)
        batch_users = batch_users.to(device)
        
        # Take first sample with sufficient length
        for i in range(len(batch_sequences)):
            seq = batch_sequences[i]
            user_id = batch_users[i]
            
            # Check if sequence has enough non-padding tokens
            valid_tokens = (seq != model.pad_token).sum().item()
            if valid_tokens >= 3:
                print(f"\nVisualizing attention for sequence with {valid_tokens} items")
                
                # Create attention visualization
                os.makedirs('visualizations', exist_ok=True)
                model.visualize_attention(
                    seq, user_id, 
                    save_path=f'visualizations/attention_sample_{i}.png'
                )
                return
    
    print("No suitable sequence found for attention visualization")

def main():
    """Train the enhanced transformer model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load preprocessed data
    print("Loading data...")
    with open('data/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open('data/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    print(f"Training samples: {len(train_data['sequences']):,}")
    print(f"Test samples: {len(test_data['sequences']):,}")
    print(f"Number of items: {encoders['n_items']:,}")
    print(f"Number of users: {encoders['n_users']:,}")
    print(f"Sequence length: {encoders['max_sequence_length']}")
    
    # Create enhanced data loaders
    batch_size = 32
    train_loader, test_loader = create_enhanced_data_loaders(train_data, test_data, batch_size)
    
    # Create enhanced transformer model
    model = EnhancedTransformerRecommender(
        n_items=encoders['n_items'],
        n_users=encoders['n_users'],
        n_categories=20,  # Assume 20 product categories
        d_model=128,      # Larger model for better performance
        n_heads=8,
        n_layers=4,       # More layers
        d_ff=256,
        max_seq_len=encoders['max_sequence_length'],
        dropout=0.1
    )
    
    print(f"Enhanced model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Enhanced training setup
    trainer = EnhancedTransformerTrainer(model, device)
    
    # Multi-task loss functions
    item_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    category_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Enhanced optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,  # Slightly higher learning rate for larger model
        weight_decay=1e-4,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Enhanced scheduler with warmup for longer training
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=len(train_loader),
        epochs=50,  # Extended training
        pct_start=0.05,  # Shorter warmup percentage for longer training
        div_factor=25.0,  # Start with lower learning rate
        final_div_factor=1000.0  # End with very low learning rate
    )
    
    # Training loop
    epochs = 50
    best_loss = float('inf')
    patience = 10  # Increased patience for longer training
    patience_counter = 0
    
    print(f"\nStarting enhanced training...")
    print(f"Features: User embeddings, Category embeddings, Position embeddings")
    print(f"Techniques: Multi-task learning, Beam search, Attention visualization")
    
    training_history = {
        'train_loss': [], 'item_loss': [], 'category_loss': [], 
        'test_loss': [], 'test_accuracy': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        item_losses = []
        category_losses = []
        nan_count = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_sequences, batch_labels, batch_users in pbar:
            total_loss, item_loss, category_loss = trainer.train_step(
                batch_sequences, batch_labels, batch_users, 
                optimizer, item_criterion, category_criterion
            )
            
            if np.isnan(total_loss):
                nan_count += 1
            else:
                train_losses.append(total_loss)
                item_losses.append(item_loss)
                category_losses.append(category_loss)
            
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'total_loss': f'{total_loss:.4f}' if not np.isnan(total_loss) else 'NaN',
                'item_loss': f'{item_loss:.4f}' if not np.isnan(item_loss) else 'NaN',
                'lr': f'{current_lr:.2e}',
                'nan_count': nan_count
            })
        
        # Evaluation
        test_loss, test_item_loss, test_category_loss, test_accuracy = trainer.evaluate(
            test_loader, item_criterion, category_criterion
        )
        
        # Record history
        training_history['train_loss'].append(np.mean(train_losses) if train_losses else float('nan'))
        training_history['item_loss'].append(np.mean(item_losses) if item_losses else float('nan'))
        training_history['category_loss'].append(np.mean(category_losses) if category_losses else float('nan'))
        training_history['test_loss'].append(test_loss)
        training_history['test_accuracy'].append(test_accuracy)
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {np.mean(train_losses):.4f} (NaN batches: {nan_count})')
        print(f'  Train Item Loss: {np.mean(item_losses):.4f}')
        print(f'  Train Category Loss: {np.mean(category_losses):.4f}')
        print(f'  Test Loss: {test_loss:.4f}')
        print(f'  Test Item Loss: {test_item_loss:.4f}')
        print(f'  Test Category Loss: {test_category_loss:.4f}')
        print(f'  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
        
        # Model saving with enhanced criteria
        if not np.isnan(test_loss):
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'n_items': encoders['n_items'],
                        'n_users': encoders['n_users'],
                        'n_categories': 20,
                        'd_model': 128,
                        'n_heads': 8,
                        'n_layers': 4,
                        'd_ff': 256,
                        'max_seq_len': encoders['max_sequence_length'],
                        'dropout': 0.1
                    },
                    'epoch': epoch,
                    'loss': test_loss,
                    'training_history': training_history
                }, 'models/enhanced_transformer.pth')
                print(f'  → Enhanced model saved! Loss: {test_loss:.4f}')
            else:
                patience_counter += 1
                print(f'  → No improvement ({patience_counter}/{patience})')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f'models/enhanced_transformer_epoch_{epoch+1}.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'n_items': encoders['n_items'],
                        'n_users': encoders['n_users'],
                        'n_categories': 20,
                        'd_model': 128,
                        'n_heads': 8,
                        'n_layers': 4,
                        'd_ff': 256,
                        'max_seq_len': encoders['max_sequence_length'],
                        'dropout': 0.1
                    },
                    'epoch': epoch,
                    'loss': test_loss,
                    'training_history': training_history
                }, checkpoint_path)
                print(f'  → Checkpoint saved: {checkpoint_path}')
        else:
            patience_counter += 1
            print(f'  → NaN loss detected ({patience_counter}/{patience})')
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
        
        print('-' * 70)
    
    print("Enhanced training completed!")
    
    # Final comprehensive evaluation
    if os.path.exists('models/enhanced_transformer.pth'):
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION")
        print("="*70)
        
        # Load best model
        checkpoint = torch.load('models/enhanced_transformer.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Basic metrics
        final_loss, final_item_loss, final_category_loss, final_accuracy = trainer.evaluate(
            test_loader, item_criterion, category_criterion
        )
        
        print(f"Final Test Loss: {final_loss:.4f}")
        print(f"Final Item Loss: {final_item_loss:.4f}")
        print(f"Final Category Loss: {final_category_loss:.4f}")
        print(f"Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        
        # Enhanced metrics with beam search
        enhanced_metrics = calculate_enhanced_metrics(model, test_loader, device, k_values=[5, 10, 20])
        
        print(f"\n=== ENHANCED METRICS COMPARISON ===")
        for k in [5, 10, 20]:
            std_hr = enhanced_metrics[f'hit_rate@{k}']
            beam_hr = enhanced_metrics[f'beam_hit_rate@{k}']
            std_ndcg = enhanced_metrics[f'ndcg@{k}']
            beam_ndcg = enhanced_metrics[f'beam_ndcg@{k}']
            
            print(f"Hit Rate @{k}:")
            print(f"  Standard: {std_hr:.4f} ({std_hr*100:.2f}%)")
            print(f"  Beam Search: {beam_hr:.4f} ({beam_hr*100:.2f}%) [+{((beam_hr-std_hr)/std_hr*100):.1f}%]")
            print(f"NDCG @{k}:")
            print(f"  Standard: {std_ndcg:.4f}")
            print(f"  Beam Search: {beam_ndcg:.4f} [+{((beam_ndcg-std_ndcg)/std_ndcg*100):.1f}%]")
            print()
        
        # Attention visualization
        print("=== ATTENTION VISUALIZATION ===")
        visualize_sample_attention(model, test_loader, encoders, device)
        
        # Performance comparison
        print("=== PERFORMANCE vs PREVIOUS MODELS ===")
        best_hr_10 = enhanced_metrics['beam_hit_rate@10']
        print(f"Enhanced Transformer (Beam): {best_hr_10*100:.2f}%")
        print(f"Stable Transformer: 3.08%")
        print(f"Simple LSTM: 2.21%")
        print(f"Broken Transformer: ~0%")
        
        if best_hr_10 > 0.05:  # 5%
            print(f"\n🎉 EXCELLENT! {((best_hr_10-0.0308)/0.0308*100):+.1f}% improvement over stable transformer!")
        elif best_hr_10 > 0.035:  # 3.5%
            print(f"\n✅ GOOD! {((best_hr_10-0.0308)/0.0308*100):+.1f}% improvement over stable transformer!")
        else:
            print(f"\n⚠️ Modest improvement: {((best_hr_10-0.0308)/0.0308*100):+.1f}%")
        
        # Target comparison
        print(f"\n=== PROGRESS TOWARD PRODUCTION TARGETS ===")
        target_hr_10 = 0.20  # 20% target
        progress = (best_hr_10 / target_hr_10) * 100
        print(f"Current: {best_hr_10*100:.2f}% | Target: {target_hr_10*100:.0f}% | Progress: {progress:.1f}%")
        
        if progress > 50:
            print("🚀 More than halfway to production quality!")
        elif progress > 25:
            print("📈 Good progress toward production targets!")
        else:
            print("📊 Early stage, but showing improvement!")
    
    else:
        print("❌ No enhanced model was successfully saved")

if __name__ == "__main__":
    main()
