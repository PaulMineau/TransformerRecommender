#!/usr/bin/env python3
"""
Quick Benchmark Comparison Script

This script runs the most important evaluation protocols to demonstrate
the impact of different methodologies on recommendation metrics.

Usage:
    python run_benchmark_comparison.py
    python run_benchmark_comparison.py --model stable_transformer
"""

import sys
import os
from benchmark_protocols import ProtocolBenchmark

def main():
    """Run a focused comparison of the most important protocols"""
    
    print("🏆 BENCHMARK PROTOCOL COMPARISON")
    print("="*60)
    print("This script demonstrates how evaluation methodology affects metrics")
    print("by comparing our Enhanced Transformer under different protocols.")
    print()
    
    # Check if model exists
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'enhanced_transformer'
    
    if not os.path.exists('models/enhanced_transformer_50epochs.pth') and model_name == 'enhanced_transformer':
        print("❌ Enhanced transformer model not found.")
        print("Please train the model first using: python train_50_epochs.py")
        return
    
    # Create benchmark
    try:
        benchmark = ProtocolBenchmark()
    except FileNotFoundError:
        print("❌ Data not found. Please ensure you have run data preprocessing.")
        print("Expected files: data/train_data.pkl, data/test_data.pkl, data/encoders.pkl")
        return
    
    print(f"📊 Benchmarking model: {model_name}")
    print()
    
    # Load model
    model = benchmark.load_model(model_name)
    if model is None:
        return
    
    # Quick protocol comparison (most important ones)
    protocols_to_test = [
        {
            'name': 'Current (Conservative)',
            'split': 'leave_one_out',
            'negatives': 'none',
            'description': 'Our current approach - most realistic/conservative'
        },
        {
            'name': 'Random Negatives (99)',
            'split': 'leave_one_out', 
            'negatives': 'random_99',
            'description': 'Common in literature - adds 99 random negative items'
        },
        {
            'name': 'Popular Negatives (99)',
            'split': 'leave_one_out',
            'negatives': 'popular_99', 
            'description': 'Harder evaluation - popular items as negatives'
        },
        {
            'name': 'Random Split + Negatives',
            'split': 'random_80_20',
            'negatives': 'random_99',
            'description': 'Different split methodology'
        }
    ]
    
    results = {}
    
    for protocol in protocols_to_test:
        print(f"\n📋 Testing: {protocol['name']}")
        print(f"   {protocol['description']}")
        print("-" * 50)
        
        # Get split
        if protocol['split'] == 'leave_one_out':
            train_data, test_data = benchmark.split_leave_one_out()
        elif protocol['split'] == 'random_80_20':
            train_data, test_data = benchmark.split_random(0.2)
        
        # Get negatives
        if protocol['negatives'] == 'none':
            negatives = None
        elif protocol['negatives'] == 'random_99':
            negatives = benchmark.generate_negative_samples(test_data, 'random', 99)
        elif protocol['negatives'] == 'popular_99':
            negatives = benchmark.generate_negative_samples(test_data, 'popular', 99)
        
        # Evaluate
        metrics = benchmark.evaluate_protocol(model, test_data, negatives)
        
        # Store and display results
        results[protocol['name']] = metrics
        
        print(f"Hit Rate @10: {metrics['hr@10']:.4f} ({metrics['hr@10']*100:.2f}%)")
        print(f"NDCG @10: {metrics['ndcg@10']:.4f}")
        print(f"Hit Rate @20: {metrics['hr@20']:.4f} ({metrics['hr@20']*100:.2f}%)")
    
    # Summary comparison
    print(f"\n🎯 PROTOCOL IMPACT SUMMARY")
    print("="*60)
    print(f"{'Protocol':<25} {'HR@10':<12} {'NDCG@10':<12} {'Improvement'}")
    print("-" * 60)
    
    baseline_hr = results['Current (Conservative)']['hr@10']
    
    for name, metrics in results.items():
        hr10 = metrics['hr@10']
        ndcg10 = metrics['ndcg@10']
        
        if name == 'Current (Conservative)':
            improvement = "Baseline"
        else:
            improvement = f"+{((hr10/baseline_hr - 1)*100):.0f}%" if baseline_hr > 0 else "N/A"
        
        print(f"{name:<25} {hr10:<12.4f} {ndcg10:<12.4f} {improvement}")
    
    print("-" * 60)
    print()
    
    # Key insights
    print("💡 KEY INSIGHTS:")
    print("• Higher scores with negative sampling show methodology impact")
    print("• Our 'conservative' approach provides realistic production estimates")
    print("• Published benchmarks may use different evaluation protocols")
    print("• Fair comparison requires identical evaluation methodologies")
    print()
    
    # Save quick results
    os.makedirs('benchmark_results', exist_ok=True)
    with open('benchmark_results/quick_comparison.txt', 'w') as f:
        f.write(f"Quick Benchmark Comparison - {model_name}\n")
        f.write("="*50 + "\n\n")
        
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            f.write(f"  HR@10: {metrics['hr@10']:.4f} ({metrics['hr@10']*100:.2f}%)\n")
            f.write(f"  NDCG@10: {metrics['ndcg@10']:.4f}\n\n")
    
    print("📁 Quick results saved to: benchmark_results/quick_comparison.txt")

if __name__ == "__main__":
    main()
