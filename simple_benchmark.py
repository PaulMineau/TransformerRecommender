#!/usr/bin/env python3
"""
Simple Benchmark Demonstration

This demonstrates the key insight: how evaluation methodology affects metrics.
We'll show the impact of negative sampling on our model's reported performance.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import random
from torch.utils.data import DataLoader, Dataset

class SimpleBenchmarkDataset(Dataset):
    def __init__(self, sequences, labels, max_len=20):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
        
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
            
        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

def evaluate_with_negatives(model, test_data, n_items, num_negatives=99, device='cpu'):
    """Evaluate model with negative sampling (common in literature)"""
    model.eval()
    
    dataset = SimpleBenchmarkDataset(test_data['sequences'], test_data['labels'])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    hit_rate_10 = 0
    ndcg_10 = 0
    total_samples = 0
    
    all_items = list(range(1, n_items))  # Exclude padding (0)
    
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            batch_size = sequences.size(0)
            
            # Get model predictions - handle enhanced transformer output
            try:
                output = model(sequences)
                if isinstance(output, dict):
                    logits = output['item_logits']
                elif isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
            except:
                # If enhanced model fails, try simpler approach
                try:
                    logits = model(sequences)
                except:
                    print(f"Error with model prediction, skipping batch")
                    continue
            
            for i in range(batch_size):
                true_item = labels[i].item()
                
                # Sample random negatives
                negatives = random.sample([item for item in all_items if item != true_item], 
                                        min(num_negatives, len(all_items)-1))
                
                # Create candidate set: true item + negatives
                candidates = [true_item] + negatives
                candidate_tensor = torch.tensor(candidates, device=device)
                
                # Get scores for candidates
                candidate_scores = logits[i][candidate_tensor]
                
                # Rank candidates (higher score = better)
                _, ranked_indices = torch.sort(candidate_scores, descending=True)
                
                # Check if true item is in top 10
                true_item_rank = (ranked_indices == 0).nonzero(as_tuple=True)[0][0].item() + 1
                
                if true_item_rank <= 10:
                    hit_rate_10 += 1
                    ndcg_10 += 1.0 / np.log2(true_item_rank + 1)
                
                total_samples += 1
    
    hr_10 = hit_rate_10 / total_samples if total_samples > 0 else 0
    ndcg_10_score = ndcg_10 / total_samples if total_samples > 0 else 0
    
    return hr_10, ndcg_10_score

def evaluate_all_items(model, test_data, device='cpu'):
    """Evaluate model against all items (our current approach)"""
    model.eval()
    
    dataset = SimpleBenchmarkDataset(test_data['sequences'], test_data['labels'])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    hit_rate_10 = 0
    ndcg_10 = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            batch_size = sequences.size(0)
            
            # Get model predictions - handle enhanced transformer output
            try:
                output = model(sequences)
                if isinstance(output, dict):
                    logits = output['item_logits']
                elif isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
            except:
                print(f"Error with model prediction, skipping batch")
                continue
            
            for i in range(batch_size):
                true_item = labels[i].item()
                
                # Rank all items
                _, ranked_items = torch.sort(logits[i], descending=True)
                
                # Find true item rank
                true_item_rank = (ranked_items == true_item).nonzero(as_tuple=True)[0]
                
                if len(true_item_rank) > 0:
                    rank = true_item_rank[0].item() + 1
                    
                    if rank <= 10:
                        hit_rate_10 += 1
                        ndcg_10 += 1.0 / np.log2(rank + 1)
                
                total_samples += 1
    
    hr_10 = hit_rate_10 / total_samples if total_samples > 0 else 0
    ndcg_10_score = ndcg_10 / total_samples if total_samples > 0 else 0
    
    return hr_10, ndcg_10_score

def main():
    """Run the benchmark demonstration"""
    print("🎯 EVALUATION METHODOLOGY IMPACT DEMONSTRATION")
    print("="*60)
    print("This shows how different evaluation protocols affect reported metrics.")
    print()
    
    # Load data
    try:
        with open('data/test_data.pkl', 'rb') as f:
            test_data = pickle.load(f)
        with open('data/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
    except FileNotFoundError:
        print("❌ Data files not found. Please run data preprocessing first.")
        return
    
    n_items = encoders['n_items']
    print(f"Loaded test data: {len(test_data['sequences'])} samples")
    print(f"Total items in catalog: {n_items}")
    print()
    
    # Load model
    try:
        from src.enhanced_transformer import EnhancedTransformerRecommender
        
        # Try to load the best model
        model_file = 'models/enhanced_transformer_50epochs.pth'
        checkpoint = torch.load(model_file, map_location='cpu')
        
        config = checkpoint.get('config', {
            'n_items': n_items,
            'n_users': encoders['n_users'],
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 4,
            'd_ff': 512,
            'max_seq_len': 20,
            'dropout': 0.1
        })
        
        model = EnhancedTransformerRecommender(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("✅ Loaded Enhanced Transformer (50 epochs)")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    print()
    
    # Run different evaluation protocols
    print("🧪 RUNNING DIFFERENT EVALUATION PROTOCOLS")
    print("-" * 40)
    
    # 1. Our current approach (all items)
    print("1️⃣ Current Approach (All Items Ranking)")
    hr_all, ndcg_all = evaluate_all_items(model, test_data, device)
    print(f"   Hit Rate @10: {hr_all:.4f} ({hr_all*100:.2f}%)")
    print(f"   NDCG @10: {ndcg_all:.4f}")
    print("   → Most conservative/realistic for production")
    print()
    
    # 2. With 99 random negatives
    print("2️⃣ With 99 Random Negatives (Literature Standard)")
    hr_99, ndcg_99 = evaluate_with_negatives(model, test_data, n_items, 99, device)
    print(f"   Hit Rate @10: {hr_99:.4f} ({hr_99*100:.2f}%)")
    print(f"   NDCG @10: {ndcg_99:.4f}")
    improvement_99 = ((hr_99 / hr_all - 1) * 100) if hr_all > 0 else float('inf')
    print(f"   → {improvement_99:.0f}% higher than all-items approach")
    print()
    
    # 3. With fewer negatives (easier)
    print("3️⃣ With 9 Random Negatives (Easier Evaluation)")
    hr_9, ndcg_9 = evaluate_with_negatives(model, test_data, n_items, 9, device)
    print(f"   Hit Rate @10: {hr_9:.4f} ({hr_9*100:.2f}%)")
    print(f"   NDCG @10: {ndcg_9:.4f}")
    improvement_9 = ((hr_9 / hr_all - 1) * 100) if hr_all > 0 else float('inf')
    print(f"   → {improvement_9:.0f}% higher than all-items approach")
    print()
    
    # Summary
    print("📋 SUMMARY OF FINDINGS")
    print("=" * 40)
    print(f"{'Protocol':<25} {'HR@10':<12} {'NDCG@10':<12} {'vs All Items'}")
    print("-" * 55)
    print(f"{'All Items (Ours)':<25} {hr_all:<12.4f} {ndcg_all:<12.4f} {'Baseline'}")
    print(f"{'99 Negatives':<25} {hr_99:<12.4f} {ndcg_99:<12.4f} {f'+{improvement_99:.0f}%'}")
    print(f"{'9 Negatives':<25} {hr_9:<12.4f} {ndcg_9:<12.4f} {f'+{improvement_9:.0f}%'}")
    print("-" * 55)
    print()
    
    print("💡 KEY INSIGHTS:")
    print("• Negative sampling dramatically improves reported metrics")
    print("• Our conservative approach provides realistic production estimates")
    print("• Published benchmarks may use different evaluation protocols")
    print("• Always check evaluation methodology when comparing results!")
    print()
    
    # Save results
    with open('benchmark_results/methodology_impact.txt', 'w') as f:
        f.write("Evaluation Methodology Impact Analysis\n")
        f.write("="*40 + "\n\n")
        f.write(f"All Items Evaluation: HR@10={hr_all:.4f}, NDCG@10={ndcg_all:.4f}\n")
        f.write(f"99 Negatives: HR@10={hr_99:.4f}, NDCG@10={ndcg_99:.4f} (+{improvement_99:.0f}%)\n")
        f.write(f"9 Negatives: HR@10={hr_9:.4f}, NDCG@10={ndcg_9:.4f} (+{improvement_9:.0f}%)\n")
        f.write(f"\nConclusion: Negative sampling increases reported metrics by {improvement_99:.0f}%\n")
        f.write("This explains why our 'conservative' scores are lower than some published benchmarks.\n")
    
    print("📁 Results saved to: benchmark_results/methodology_impact.txt")

if __name__ == "__main__":
    main()
