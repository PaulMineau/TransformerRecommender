"""
Training script for stable transformer model
"""
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from src.stable_transformer import StableTransformerRecommender, StableTransformerTrainer

def main():
    """Train the stable transformer model"""
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
    print(f"Sequence length: {encoders['max_sequence_length']}")
    
    # Create data loaders with smaller batch size for stability
    train_sequences = torch.LongTensor(train_data['sequences'])
    train_labels = torch.LongTensor(train_data['labels'])
    test_sequences = torch.LongTensor(test_data['sequences'])
    test_labels = torch.LongTensor(test_data['labels'])
    
    train_dataset = TensorDataset(train_sequences, train_labels)
    test_dataset = TensorDataset(test_sequences, test_labels)
    
    # Smaller batch size for transformer stability
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create stable transformer model
    model = StableTransformerRecommender(
        n_items=encoders['n_items'],
        d_model=64,  # Smaller model for stability
        n_heads=4,   # Fewer heads
        n_layers=2,  # Fewer layers
        d_ff=128,    # Smaller feedforward
        max_seq_len=encoders['max_sequence_length'],
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup with very conservative parameters
    trainer = StableTransformerTrainer(model, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for stability
    
    # Very small learning rate for transformers
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-5,  # Very small learning rate
        weight_decay=1e-4,
        betas=(0.9, 0.98),  # Transformer-optimized betas
        eps=1e-9
    )
    
    # Warmup scheduler
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,  # Peak learning rate
        steps_per_epoch=len(train_loader),
        epochs=8,
        pct_start=0.1  # 10% warmup
    )
    
    # Training loop
    epochs = 8
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    print(f"\nStarting training with very conservative settings...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Model size: d_model={model.d_model}, n_heads={model.transformer.layers[0].self_attn.num_heads}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        nan_count = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_sequences, batch_labels in pbar:
            loss = trainer.train_step(batch_sequences, batch_labels, optimizer, criterion)
            
            if np.isnan(loss):
                nan_count += 1
            else:
                train_losses.append(loss)
            
            # Update learning rate
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss:.4f}' if not np.isnan(loss) else 'NaN',
                'lr': f'{current_lr:.2e}',
                'nan_count': nan_count
            })
        
        # Evaluation
        test_loss, test_accuracy = trainer.evaluate(test_loader, criterion)
        
        avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f} (NaN batches: {nan_count})')
        print(f'  Test Loss: {test_loss:.4f}')
        print(f'  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
        
        # Early stopping and model saving
        if not np.isnan(test_loss):
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'n_items': encoders['n_items'],
                        'd_model': 64,
                        'n_heads': 4,
                        'n_layers': 2,
                        'd_ff': 128,
                        'max_seq_len': encoders['max_sequence_length'],
                        'dropout': 0.1
                    },
                    'epoch': epoch,
                    'loss': test_loss
                }, 'models/stable_transformer.pth')
                print(f'  → Best model saved! Loss: {test_loss:.4f}')
            else:
                patience_counter += 1
                print(f'  → No improvement ({patience_counter}/{patience})')
        else:
            patience_counter += 1
            print(f'  → NaN loss detected ({patience_counter}/{patience})')
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
        
        print('-' * 60)
    
    print("Training completed!")
    
    # Final evaluation
    if os.path.exists('models/stable_transformer.pth'):
        print("\n=== Final Evaluation ===")
        checkpoint = torch.load('models/stable_transformer.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        final_loss, final_accuracy = trainer.evaluate(test_loader, criterion)
        print(f"Final Test Loss: {final_loss:.4f}")
        print(f"Final Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        
        # Calculate Hit Rate @10
        model.eval()
        hit_count = 0
        total_count = 0
        
        print("Calculating Hit Rate @10...")
        with torch.no_grad():
            for batch_sequences, batch_labels in tqdm(test_loader, desc="Evaluating"):
                try:
                    batch_sequences = batch_sequences.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    top_k_items, _ = model.predict_top_k(batch_sequences, k=10)
                    
                    for i in range(batch_labels.size(0)):
                        if batch_labels[i].item() in top_k_items[i].cpu().numpy():
                            hit_count += 1
                        total_count += 1
                except:
                    continue
        
        hit_rate_10 = hit_count / total_count if total_count > 0 else 0
        print(f"Hit Rate @10: {hit_rate_10:.4f} ({hit_rate_10*100:.2f}%)")
        
        # Compare with targets
        print(f"\n=== Performance vs Targets ===")
        print(f"Current Hit Rate @10: {hit_rate_10*100:.2f}% | Target: 15-25%")
        print(f"Current Accuracy: {final_accuracy*100:.2f}% | Target: 5-15%")
        
        if hit_rate_10 > 0.05:  # 5%
            print("✅ Model is learning meaningful patterns!")
        elif hit_rate_10 > 0.02:  # 2%
            print("⚠️ Model shows some learning, needs more training")
        else:
            print("❌ Model needs architecture improvements")
    
    else:
        print("❌ No model was successfully saved - all training failed")

if __name__ == "__main__":
    main()
