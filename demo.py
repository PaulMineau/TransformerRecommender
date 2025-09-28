"""
Quick demo script to test the recommendation system
"""
import os
import sys

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import torch
        import pandas as pd
        import numpy as np
        import streamlit
        import plotly
        import tqdm
        import requests
        import sklearn
        print("✅ All required packages are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_project_structure():
    """Check if project structure is correct"""
    required_dirs = ['src', 'data', 'models']
    required_files = [
        'src/data_downloader.py',
        'src/data_preprocessor.py', 
        'src/transformer_model.py',
        'train_model.py',
        'streamlit_app.py',
        'requirements.txt'
    ]
    
    print("\n📁 Checking project structure...")
    
    # Check directories
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory '{dir_name}' exists")
        else:
            print(f"❌ Directory '{dir_name}' missing")
            return False
    
    # Check files
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ File '{file_name}' exists")
        else:
            print(f"❌ File '{file_name}' missing")
            return False
    
    return True

def test_data_download():
    """Test data download functionality"""
    print("\n📥 Testing data download...")
    try:
        from src.data_downloader import AmazonDataDownloader
        
        # Just test the class initialization
        downloader = AmazonDataDownloader('data')
        print("✅ Data downloader initialized successfully")
        print(f"   - Data directory: {downloader.data_dir}")
        print(f"   - Available datasets: {list(downloader.urls.keys())}")
        return True
    except Exception as e:
        print(f"❌ Data download test failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\n🧠 Testing model creation...")
    try:
        from src.transformer_model import TransformerRecommender
        
        # Create a small test model
        model = TransformerRecommender(
            n_items=1000,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_seq_len=10
        )
        
        print("✅ Transformer model created successfully")
        print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Model size: ~{sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024:.1f} MB")
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

def test_preprocessing():
    """Test data preprocessing"""
    print("\n🔄 Testing data preprocessing...")
    try:
        from src.data_preprocessor import RecommendationDataPreprocessor
        import pandas as pd
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        sample_data = {
            'reviewerID': [f'user_{i//10}' for i in range(100)],
            'asin': [f'item_{np.random.randint(0, 20)}' for _ in range(100)],
            'unixReviewTime': [1000000 + i * 1000 for i in range(100)],
            'overall': [np.random.randint(1, 6) for _ in range(100)]
        }
        df = pd.DataFrame(sample_data)
        
        # Test preprocessor
        preprocessor = RecommendationDataPreprocessor(min_interactions=3, max_sequence_length=5)
        filtered_df = preprocessor.filter_data(df.copy())
        
        print("✅ Data preprocessing test passed")
        print(f"   - Original data: {len(df)} interactions")
        print(f"   - Filtered data: {len(filtered_df)} interactions")
        return True
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def show_usage_instructions():
    """Show usage instructions"""
    print("\n" + "="*60)
    print("🚀 RECOMMENDATION SYSTEM READY!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Install dependencies (if not done):")
    print("   pip install -r requirements.txt")
    
    print("\n2. Train the model:")
    print("   python train_model.py")
    print("   (This will download data and train the model - takes 10-30 minutes)")
    
    print("\n3. Run the Streamlit app:")
    print("   streamlit run streamlit_app.py")
    print("   (Open browser to http://localhost:8501)")
    
    print("\n📊 What to Expect:")
    print("• The training script will download ~50MB of Amazon Beauty data")
    print("• Training typically takes 15-30 minutes on CPU, 5-10 minutes on GPU")
    print("• The model achieves Hit Rate@10 of ~0.15-0.25 on the test set")
    print("• The Streamlit app provides an interactive interface for recommendations")
    
    print("\n🔧 Customization:")
    print("• Edit config in train_model.py to adjust model parameters")
    print("• Modify min_interactions and max_seq_len for different dataset sizes")
    print("• Increase epochs for better performance (at cost of training time)")
    
    print("\n💡 Tips:")
    print("• Use GPU if available for faster training")
    print("• Start with default parameters, then experiment")
    print("• Check models/ directory for saved models and training plots")

def main():
    """Run all demo tests"""
    print("🧪 TRANSFORMER RECOMMENDATION SYSTEM - DEMO")
    print("="*50)
    
    tests = [
        ("Requirements Check", check_requirements),
        ("Project Structure", check_project_structure),
        ("Data Download", test_data_download),
        ("Model Creation", test_model_creation),
        ("Data Preprocessing", test_preprocessing)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            all_passed = False
    
    if all_passed:
        show_usage_instructions()
    else:
        print("\n❌ Some tests failed. Please check the errors above and fix them.")
        print("Common fixes:")
        print("• Run: pip install -r requirements.txt")
        print("• Make sure you're in the correct directory")
        print("• Check that all files were created properly")

if __name__ == "__main__":
    main()
