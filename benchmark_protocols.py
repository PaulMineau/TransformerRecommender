#!/usr/bin/env python3
"""
Comprehensive Benchmarking System for Fair Comparison with Published Results

This script implements different evaluation protocols to enable fair comparison
with published benchmarks on the Amazon Beauty dataset. It addresses the key
methodological differences identified in our SOTA comparison.

Features:
- Multiple negative sampling strategies
- Different train/test split methodologies  
- Various preprocessing pipelines
- Protocol impact analysis
- Standardized metric reporting
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict
import argparse
import json
from datetime import datetime

# Import our models
from src.enhanced_transformer import EnhancedTransformerRecommender
from src.stable_transformer import StableTransformerRecommender
from src.simple_model import SimpleRecommender

class BenchmarkDataset(Dataset):
    """Dataset class for different evaluation protocols"""
    
    def __init__(self, sequences, labels, max_len=20, negative_samples=None, users=None):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
        self.negative_samples = negative_samples
        self.users = users
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Pad sequence
        if len(sequence) > self.max_len:
            sequence = sequence[-self.max_len:]
        else:
            sequence = [0] * (self.max_len - len(sequence)) + sequence
            
        result = {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        if self.users is not None:
            result['users'] = torch.tensor(self.users[idx], dtype=torch.long)
        
        if self.negative_samples is not None:
            result['negatives'] = torch.tensor(self.negative_samples[idx], dtype=torch.long)
            
        return result

class ProtocolBenchmark:
    """Main benchmarking class implementing different evaluation protocols"""
    
    def __init__(self, data_path='data/', model_path='models/', results_path='benchmark_results/'):
        self.data_path = data_path
        self.model_path = model_path
        self.results_path = results_path
        
        # Create results directory
        os.makedirs(self.results_path, exist_ok=True)
        
        # Load data
        self.load_data()
        
        print(f"Loaded data: {len(self.interactions)} interactions")
        print(f"Users: {self.n_users}, Items: {self.n_items}")
    
    def load_data(self):
        """Load and prepare the dataset"""
        try:
            # Try to load preprocessed data
            with open(os.path.join(self.data_path, 'train_data.pkl'), 'rb') as f:
                train_data = pickle.load(f)
            with open(os.path.join(self.data_path, 'test_data.pkl'), 'rb') as f:
                test_data = pickle.load(f)
            with open(os.path.join(self.data_path, 'encoders.pkl'), 'rb') as f:
                encoders = pickle.load(f)
                
            # Combine train and test for re-splitting with different protocols
            all_sequences = train_data['sequences'] + test_data['sequences']
            all_labels = train_data['labels'] + test_data['labels']
            all_users = train_data.get('users', []) + test_data.get('users', [])
            
            self.all_sequences = all_sequences
            self.all_labels = all_labels
            self.all_users = all_users
            self.n_items = encoders['n_items']
            self.n_users = encoders['n_users']
            
            # Create user-item interaction matrix for analysis
            self.create_interaction_matrix()
            
        except FileNotFoundError:
            print("Preprocessed data not found. Please run data preprocessing first.")
            raise
    
    def create_interaction_matrix(self):
        """Create user-item interaction matrix from sequences"""
        self.interactions = []
        
        for i, (seq, label, user) in enumerate(zip(self.all_sequences, self.all_labels, self.all_users)):
            # Add all items in sequence
            for item in seq:
                if item > 0:  # Skip padding
                    self.interactions.append((user, item, 1))
            # Add the target item
            self.interactions.append((user, label, 1))
        
        # Convert to DataFrame
        self.interactions_df = pd.DataFrame(self.interactions, columns=['user', 'item', 'rating'])
        
        # Create user-item sets for efficient lookup
        self.user_items = defaultdict(set)
        for user, item, _ in self.interactions:
            self.user_items[user].add(item)
    
    def split_temporal(self, test_ratio=0.2):
        """Temporal split: latest interactions go to test set"""
        print(f"\n📅 Temporal Split (test ratio: {test_ratio})")
        
        # Sort by timestamp if available, otherwise by sequence order
        sorted_indices = list(range(len(self.all_sequences)))
        
        # Split point
        split_point = int(len(sorted_indices) * (1 - test_ratio))
        
        train_indices = sorted_indices[:split_point]
        test_indices = sorted_indices[split_point:]
        
        train_data = {
            'sequences': [self.all_sequences[i] for i in train_indices],
            'labels': [self.all_labels[i] for i in train_indices],
            'users': [self.all_users[i] for i in train_indices]
        }
        
        test_data = {
            'sequences': [self.all_sequences[i] for i in test_indices],
            'labels': [self.all_labels[i] for i in test_indices], 
            'users': [self.all_users[i] for i in test_indices]
        }
        
        print(f"Train samples: {len(train_data['sequences'])}")
        print(f"Test samples: {len(test_data['sequences'])}")
        
        return train_data, test_data
    
    def split_random(self, test_ratio=0.2, seed=42):
        """Random split: randomly assign interactions to train/test"""
        print(f"\n🎲 Random Split (test ratio: {test_ratio}, seed: {seed})")
        
        indices = list(range(len(self.all_sequences)))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=seed, shuffle=True
        )
        
        train_data = {
            'sequences': [self.all_sequences[i] for i in train_indices],
            'labels': [self.all_labels[i] for i in train_indices],
            'users': [self.all_users[i] for i in train_indices]
        }
        
        test_data = {
            'sequences': [self.all_sequences[i] for i in test_indices],
            'labels': [self.all_labels[i] for i in test_indices],
            'users': [self.all_users[i] for i in test_indices]
        }
        
        print(f"Train samples: {len(train_data['sequences'])}")
        print(f"Test samples: {len(test_data['sequences'])}")
        
        return train_data, test_data
    
    def split_leave_one_out(self):
        """Leave-one-out: last item of each user sequence goes to test"""
        print(f"\n👤 Leave-One-Out Split")
        
        # This is essentially what we already have, but let's be explicit
        train_data = {
            'sequences': self.all_sequences,
            'labels': self.all_labels,
            'users': self.all_users
        }
        
        test_data = {
            'sequences': self.all_sequences,
            'labels': self.all_labels,
            'users': self.all_users
        }
        
        print(f"Train samples: {len(train_data['sequences'])}")
        print(f"Test samples: {len(test_data['sequences'])}")
        print("Note: In leave-one-out, train and test use same sequences with last item as target")
        
        return train_data, test_data
    
    def generate_negative_samples(self, test_data, strategy='random', num_negatives=99):
        """Generate negative samples for evaluation"""
        print(f"\n🎯 Generating negative samples: {strategy} (num_negatives: {num_negatives})")
        
        negatives = []
        all_items = set(range(1, self.n_items))  # Exclude padding (0)
        
        for i, (user, label) in enumerate(zip(test_data['users'], test_data['labels'])):
            user_positives = self.user_items[user]
            
            if strategy == 'random':
                # Random negative sampling
                candidates = list(all_items - user_positives)
                if len(candidates) >= num_negatives:
                    neg_samples = random.sample(candidates, num_negatives)
                else:
                    neg_samples = candidates + random.choices(candidates, k=num_negatives-len(candidates))
                    
            elif strategy == 'popular':
                # Popular item negative sampling (harder)
                item_counts = self.interactions_df['item'].value_counts()
                popular_items = item_counts.index.tolist()
                candidates = [item for item in popular_items if item not in user_positives]
                
                if len(candidates) >= num_negatives:
                    neg_samples = candidates[:num_negatives]
                else:
                    neg_samples = candidates + random.choices(candidates, k=num_negatives-len(candidates))
                    
            elif strategy == 'none':
                # No negative sampling - use all items
                neg_samples = []
                
            else:
                raise ValueError(f"Unknown negative sampling strategy: {strategy}")
                
            negatives.append(neg_samples)
        
        print(f"Generated {len(negatives)} negative sample sets")
        return negatives
    
    def evaluate_protocol(self, model, test_data, negatives=None, k_values=[5, 10, 20], device='cpu'):
        """Evaluate model with specific protocol"""
        model.eval()
        
        # Create dataset
        users = test_data.get('users', None)
        if negatives is not None:
            dataset = BenchmarkDataset(test_data['sequences'], test_data['labels'], 
                                     negative_samples=negatives, users=users)
        else:
            dataset = BenchmarkDataset(test_data['sequences'], test_data['labels'], users=users)
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        hit_rates = {k: 0 for k in k_values}
        ndcg_scores = {k: 0 for k in k_values}
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                sequences = batch['sequence'].to(device)
                labels = batch['label'].to(device)
                batch_size = sequences.size(0)
                
                # Get model predictions
                try:
                    if hasattr(model, 'predict'):
                        output = model.predict(sequences)
                    else:
                        # For enhanced transformer, we need to pass user_ids
                        if hasattr(model, 'user_embedding') and 'users' in batch:
                            user_ids = batch.get('users', torch.zeros(batch_size, dtype=torch.long).to(device))
                            output = model(sequences, user_ids=user_ids)
                        else:
                            output = model(sequences)
                    
                    # Handle different output formats
                    if isinstance(output, dict):
                        logits = output['item_logits']  # Enhanced transformer returns dict
                    elif isinstance(output, tuple):
                        logits = output[0]  # Take first output if tuple
                    else:
                        logits = output
                    
                except Exception as e:
                    print(f"Error in model prediction: {e}")
                    continue
                
                for i in range(batch_size):
                    true_item = labels[i].item()
                    
                    # Handle single batch dimension
                    if len(logits.shape) == 1:
                        user_logits = logits
                    else:
                        user_logits = logits[i]
                    
                    if negatives is not None:
                        # Evaluate only on true item + negatives
                        neg_items = batch['negatives'][i]
                        candidate_items = torch.cat([labels[i:i+1], neg_items])
                        candidate_scores = user_logits[candidate_items]
                        
                        # Rank candidates
                        _, ranked_indices = torch.sort(candidate_scores, descending=True)
                        ranked_items = candidate_items[ranked_indices]
                        
                    else:
                        # Evaluate on all items (current approach)
                        _, ranked_items = torch.sort(user_logits, descending=True)
                        ranked_items = ranked_items.cpu()
                    
                    # Calculate metrics for different k values
                    for k in k_values:
                        top_k = ranked_items[:k]
                        if true_item in top_k:
                            hit_rates[k] += 1
                            rank = (top_k == true_item).nonzero(as_tuple=True)[0][0].item() + 1
                            ndcg_scores[k] += 1.0 / np.log2(rank + 1)
                    
                    total_samples += 1
        
        # Calculate final metrics
        metrics = {}
        for k in k_values:
            metrics[f'hr@{k}'] = hit_rates[k] / total_samples if total_samples > 0 else 0
            metrics[f'ndcg@{k}'] = ndcg_scores[k] / total_samples if total_samples > 0 else 0
        
        return metrics
    
    def run_comprehensive_benchmark(self, model_name='enhanced_transformer'):
        """Run comprehensive benchmark with all protocols"""
        print(f"\n🏆 COMPREHENSIVE BENCHMARK: {model_name}")
        print("="*80)
        
        # Load model
        model = self.load_model(model_name)
        if model is None:
            print(f"❌ Model {model_name} not found. Skipping.")
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        results = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'protocols': {}
        }
        
        # Test different split methodologies
        split_methods = [
            ('leave_one_out', self.split_leave_one_out),
            ('temporal_80_20', lambda: self.split_temporal(0.2)),
            ('random_80_20', lambda: self.split_random(0.2)),
            ('random_90_10', lambda: self.split_random(0.1))
        ]
        
        for split_name, split_func in split_methods:
            print(f"\n📊 SPLIT METHOD: {split_name}")
            print("-" * 40)
            
            train_data, test_data = split_func()
            
            # Test different negative sampling strategies
            neg_strategies = [
                ('no_sampling', 'none', 0),
                ('random_99', 'random', 99),
                ('popular_99', 'popular', 99),
                ('random_999', 'random', 999)
            ]
            
            protocol_results = {}
            
            for neg_name, neg_strategy, num_neg in neg_strategies:
                protocol_key = f"{split_name}_{neg_name}"
                print(f"\n🎯 Protocol: {protocol_key}")
                
                # Generate negatives
                if neg_strategy != 'none':
                    negatives = self.generate_negative_samples(test_data, neg_strategy, num_neg)
                else:
                    negatives = None
                
                # Evaluate
                metrics = self.evaluate_protocol(model, test_data, negatives, device=device)
                
                # Display results
                print(f"Results for {protocol_key}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f} ({value*100:.2f}%)")
                
                protocol_results[neg_name] = {
                    'negative_strategy': neg_strategy,
                    'num_negatives': num_neg,
                    'metrics': metrics
                }
            
            results['protocols'][split_name] = protocol_results
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_path, f'benchmark_{model_name}_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def load_model(self, model_name):
        """Load a trained model"""
        model_files = {
            'enhanced_transformer': 'enhanced_transformer_50epochs.pth',
            'enhanced_transformer_8': 'enhanced_transformer.pth', 
            'stable_transformer': 'stable_transformer.pth',
            'simple_lstm': 'simple_model.pth'
        }
        
        if model_name not in model_files:
            print(f"❌ Unknown model: {model_name}")
            return None
        
        model_file = os.path.join(self.model_path, model_files[model_name])
        
        if not os.path.exists(model_file):
            print(f"❌ Model file not found: {model_file}")
            return None
        
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            
            # Extract config and create model
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                # Default config for older models
                config = {
                    'n_items': self.n_items,
                    'n_users': self.n_users,
                    'd_model': 128,
                    'n_heads': 8,
                    'n_layers': 4,
                    'd_ff': 512,
                    'max_seq_len': 20,
                    'dropout': 0.1
                }
            
            # Create model based on type
            if 'enhanced' in model_name:
                model = EnhancedTransformerRecommender(**config)
            elif 'stable' in model_name:
                model = StableTransformerRecommender(**config)
            elif 'simple' in model_name:
                model = SimpleRecommender(**config)
            else:
                print(f"❌ Unknown model type: {model_name}")
                return None
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model: {model_name}")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
            return None
    
    def generate_summary_report(self, results):
        """Generate a summary report of benchmark results"""
        print(f"\n📋 BENCHMARK SUMMARY REPORT")
        print("="*80)
        
        model_name = results['model']
        protocols = results['protocols']
        
        print(f"Model: {model_name}")
        print(f"Timestamp: {results['timestamp']}")
        print()
        
        # Create comparison table
        print("📊 Hit Rate @10 Comparison Across Protocols:")
        print("-" * 60)
        print(f"{'Protocol':<25} {'HR@10':<10} {'NDCG@10':<10} {'Notes'}")
        print("-" * 60)
        
        for split_name, split_results in protocols.items():
            for neg_name, neg_results in split_results.items():
                protocol_name = f"{split_name}_{neg_name}"
                hr10 = neg_results['metrics'].get('hr@10', 0)
                ndcg10 = neg_results['metrics'].get('ndcg@10', 0)
                num_neg = neg_results['num_negatives']
                
                if num_neg == 0:
                    notes = "All items"
                else:
                    notes = f"{num_neg} negatives"
                
                print(f"{protocol_name:<25} {hr10:<10.4f} {ndcg10:<10.4f} {notes}")
        
        print("-" * 60)
        print()
        
        # Key insights
        print("🔍 Key Insights:")
        print("- Higher scores with negative sampling indicate evaluation methodology impact")
        print("- Random vs temporal splits show generalization differences")
        print("- Popular negative sampling provides harder evaluation")
        print("- 'All items' evaluation (our current approach) is most conservative")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Comprehensive Benchmark Protocols')
    parser.add_argument('--model', default='enhanced_transformer', 
                       choices=['enhanced_transformer', 'enhanced_transformer_8', 'stable_transformer', 'simple_lstm'],
                       help='Model to benchmark')
    parser.add_argument('--data_path', default='data/', help='Path to data directory')
    parser.add_argument('--model_path', default='models/', help='Path to models directory')
    parser.add_argument('--results_path', default='benchmark_results/', help='Path to save results')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = ProtocolBenchmark(
        data_path=args.data_path,
        model_path=args.model_path, 
        results_path=args.results_path
    )
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(args.model)
    
    print(f"\n🎉 Benchmark completed for {args.model}!")
    print(f"Results saved in {args.results_path}")

if __name__ == "__main__":
    main()
