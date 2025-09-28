"""
Simple training script with stable model
"""
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from src.simple_model import SimpleRecommender, SimpleTrainer

def main():
    """Train the simple model"""
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
    
    # Create data loaders
    train_sequences = torch.LongTensor(train_data['sequences'])
    train_labels = torch.LongTensor(train_data['labels'])
    test_sequences = torch.LongTensor(test_data['sequences'])
    test_labels = torch.LongTensor(test_data['labels'])
    
    train_dataset = TensorDataset(train_sequences, train_labels)
    test_dataset = TensorDataset(test_sequences, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = SimpleRecommender(
        n_items=encoders['n_items'],
        embedding_dim=64,
        hidden_dim=128,
        max_seq_len=encoders['max_sequence_length'],
        dropout=0.2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    trainer = SimpleTrainer(model, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training loop
    epochs = 5
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_sequences, batch_labels in pbar:
            loss = trainer.train_step(batch_sequences, batch_labels, optimizer, criterion)
            if not np.isnan(loss):
                train_losses.append(loss)
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        # Evaluation
        test_loss, test_accuracy = trainer.evaluate(test_loader, criterion)
        
        avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Test Loss: {test_loss:.4f}')
        print(f'  Test Accuracy: {test_accuracy:.4f}')
        
        # Save best model
        if not np.isnan(test_loss) and test_loss < best_loss:
            best_loss = test_loss
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'n_items': encoders['n_items'],
                    'embedding_dim': 64,
                    'hidden_dim': 128,
                    'max_seq_len': encoders['max_sequence_length'],
                    'dropout': 0.2
                }
            }, 'models/simple_model.pth')
            print(f'  → Best model saved! Loss: {test_loss:.4f}')
        
        print('-' * 50)
    
    print("Training completed!")
    
    # Quick evaluation
    if os.path.exists('models/simple_model.pth'):
        print("\n=== Final Evaluation ===")
        checkpoint = torch.load('models/simple_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        final_loss, final_accuracy = trainer.evaluate(test_loader, criterion)
        print(f"Final Test Loss: {final_loss:.4f}")
        print(f"Final Test Accuracy: {final_accuracy:.4f}")
        
        # Calculate Hit Rate @10
        model.eval()
        hit_count = 0
        total_count = 0
        
        with torch.no_grad():
            for batch_sequences, batch_labels in test_loader:
                batch_sequences = batch_sequences.to(device)
                batch_labels = batch_labels.to(device)
                
                top_k_items, _ = model.predict_top_k(batch_sequences, k=10)
                
                for i in range(batch_labels.size(0)):
                    if batch_labels[i].item() in top_k_items[i].cpu().numpy():
                        hit_count += 1
                    total_count += 1
        
        hit_rate_10 = hit_count / total_count if total_count > 0 else 0
        print(f"Hit Rate @10: {hit_rate_10:.4f} ({hit_rate_10*100:.2f}%)")
        
        # Expected good metrics
        print("\n=== Expected Good Metrics ===")
        print("Hit Rate @10: 15-25% (0.15-0.25)")
        print("Test Accuracy: 5-15% (0.05-0.15)")
        print("Test Loss: 2-4 (stable, decreasing)")

if __name__ == "__main__":
    main()
