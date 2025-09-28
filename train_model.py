"""
Training script for transformer-based recommendation system
"""
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

from src.transformer_model import TransformerRecommender, RecommenderTrainer, calculate_metrics
from src.data_downloader import AmazonDataDownloader
from src.data_preprocessor import RecommendationDataPreprocessor

class RecommendationTrainingPipeline:
    """Complete training pipeline for recommendation system"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(self.config['model_dir'], exist_ok=True)
        os.makedirs(self.config['data_dir'], exist_ok=True)
        
    def prepare_data(self):
        """Download and preprocess data if not already done"""
        reviews_path = os.path.join(self.config['data_dir'], 'reviews.csv')
        train_data_path = os.path.join(self.config['data_dir'], 'train_data.pkl')
        
        # Download data if not exists
        if not os.path.exists(reviews_path):
            print("Downloading Amazon Beauty dataset...")
            downloader = AmazonDataDownloader(self.config['data_dir'])
            reviews_df, _ = downloader.download_and_process()
        else:
            print("Loading existing dataset...")
            import pandas as pd
            reviews_df = pd.read_csv(reviews_path)
        
        # Preprocess data if not exists
        if not os.path.exists(train_data_path):
            print("Preprocessing data...")
            preprocessor = RecommendationDataPreprocessor(
                min_interactions=self.config['min_interactions'],
                max_sequence_length=self.config['max_seq_len']
            )
            train_data, test_data = preprocessor.process_dataset(reviews_df, self.config['data_dir'])
        else:
            print("Loading preprocessed data...")
            with open(train_data_path, 'rb') as f:
                train_data = pickle.load(f)
            with open(os.path.join(self.config['data_dir'], 'test_data.pkl'), 'rb') as f:
                test_data = pickle.load(f)
        
        # Load encoders
        with open(os.path.join(self.config['data_dir'], 'encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
        
        return train_data, test_data, encoders
    
    def create_data_loaders(self, train_data: dict, test_data: dict):
        """Create PyTorch data loaders"""
        # Convert to tensors
        train_sequences = torch.LongTensor(train_data['sequences'])
        train_labels = torch.LongTensor(train_data['labels'])
        test_sequences = torch.LongTensor(test_data['sequences'])
        test_labels = torch.LongTensor(test_data['labels'])
        
        # Create datasets
        train_dataset = TensorDataset(train_sequences, train_labels)
        test_dataset = TensorDataset(test_sequences, test_labels)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=2
        )
        
        return train_loader, test_loader
    
    def create_model(self, n_items: int):
        """Create transformer model"""
        model = TransformerRecommender(
            n_items=n_items,
            d_model=self.config['d_model'],
            n_heads=self.config['n_heads'],
            n_layers=self.config['n_layers'],
            d_ff=self.config['d_ff'],
            max_seq_len=self.config['max_seq_len'],
            dropout=self.config['dropout']
        )
        return model
    
    def train_model(self, model: TransformerRecommender, train_loader, test_loader):
        """Train the model"""
        trainer = RecommenderTrainer(model, self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.config['scheduler_step'], 
            gamma=self.config['scheduler_gamma']
        )
        
        # Training history
        train_losses = []
        test_losses = []
        test_accuracies = []
        best_loss = float('inf')
        
        print(f"Starting training for {self.config['epochs']} epochs...")
        
        for epoch in range(self.config['epochs']):
            # Training
            model.train()
            epoch_train_loss = 0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')
            
            for batch_sequences, batch_labels in train_pbar:
                loss = trainer.train_step(batch_sequences, batch_labels, optimizer, criterion)
                epoch_train_loss += loss
                train_pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Evaluation
            test_loss, test_accuracy, _ = trainer.evaluate(test_loader, criterion)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            
            # Learning rate scheduling
            scheduler.step()
            
            print(f'Epoch {epoch+1}/{self.config["epochs"]}:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Test Loss: {test_loss:.4f}')
            print(f'  Test Accuracy: {test_accuracy:.4f}')
            print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            
            # Save best model (handle NaN losses)
            if not np.isnan(test_loss) and test_loss < best_loss:
                best_loss = test_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': test_loss,
                    'config': self.config
                }, os.path.join(self.config['model_dir'], 'best_model.pth'))
                print(f'  → New best model saved!')
            elif np.isnan(test_loss):
                print(f'  ⚠️ Warning: NaN loss detected!')
            
            print('-' * 60)
        
        return train_losses, test_losses, test_accuracies
    
    def evaluate_model(self, model: TransformerRecommender, test_loader):
        """Evaluate model with recommendation metrics"""
        print("Evaluating model with recommendation metrics...")
        
        metrics = calculate_metrics(
            model, 
            test_loader, 
            device=self.device, 
            k_values=[1, 5, 10, 20]
        )
        
        print("\n=== Recommendation Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def save_training_history(self, train_losses, test_losses, test_accuracies, metrics):
        """Save training history and create plots"""
        # Save history
        history = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies,
            'metrics': metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.config['model_dir'], 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
        ax1.set_title('Training and Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(epochs, test_accuracies, 'g-', label='Test Accuracy')
        ax2.set_title('Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Hit Rate metrics
        hit_rates = [metrics[f'hit_rate@{k}'] for k in [1, 5, 10, 20]]
        ax3.bar(['HR@1', 'HR@5', 'HR@10', 'HR@20'], hit_rates)
        ax3.set_title('Hit Rate at Different K')
        ax3.set_ylabel('Hit Rate')
        
        # NDCG metrics
        ndcg_scores = [metrics[f'ndcg@{k}'] for k in [1, 5, 10, 20]]
        ax4.bar(['NDCG@1', 'NDCG@5', 'NDCG@10', 'NDCG@20'], ndcg_scores)
        ax4.set_title('NDCG at Different K')
        ax4.set_ylabel('NDCG')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['model_dir'], 'training_plots.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_training(self):
        """Run complete training pipeline"""
        print("Starting recommendation system training pipeline...")
        
        # Prepare data
        train_data, test_data, encoders = self.prepare_data()
        train_loader, test_loader = self.create_data_loaders(train_data, test_data)
        
        print(f"\nDataset Statistics:")
        print(f"Number of users: {encoders['n_users']:,}")
        print(f"Number of items: {encoders['n_items']:,}")
        print(f"Training samples: {len(train_data['sequences']):,}")
        print(f"Test samples: {len(test_data['sequences']):,}")
        
        # Create model
        model = self.create_model(encoders['n_items'])
        print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        train_losses, test_losses, test_accuracies = self.train_model(model, train_loader, test_loader)
        
        # Load best model for evaluation (if it exists)
        best_model_path = os.path.join(self.config['model_dir'], 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded best model for evaluation")
        else:
            print("⚠️ No best model saved, using final model state")
        
        # Evaluate model
        metrics = self.evaluate_model(model, test_loader)
        
        # Save results
        self.save_training_history(train_losses, test_losses, test_accuracies, metrics)
        
        print(f"\nTraining completed! Best model saved to {self.config['model_dir']}/best_model.pth")
        return model, metrics

def main():
    """Main training function"""
    # Training configuration
    config = {
        # Data parameters
        'data_dir': 'data',
        'model_dir': 'models',
        'min_interactions': 5,
        'max_seq_len': 20,
        
        # Model parameters
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 512,
        'dropout': 0.1,
        
        # Training parameters
        'batch_size': 128,  # Smaller batch size for stability
        'learning_rate': 0.0001,  # Much smaller learning rate
        'weight_decay': 1e-6,  # Smaller weight decay
        'epochs': 10,  # Fewer epochs for testing
        'scheduler_step': 3,
        'scheduler_gamma': 0.8,
    }
    
    # Run training
    pipeline = RecommendationTrainingPipeline(config)
    model, metrics = pipeline.run_training()
    
    print("\n=== Training Summary ===")
    print(f"Final Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
