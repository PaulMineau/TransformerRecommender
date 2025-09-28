"""
Setup script for the Transformer Recommendation System
"""
import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    dirs = ['data', 'models', 'notebooks']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def run_demo():
    """Run the demo script"""
    print("\n🧪 Running demo tests...")
    try:
        subprocess.check_call([sys.executable, "demo.py"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Transformer Recommendation System")
    print("="*50)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return
    
    # Run demo
    if run_demo():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. python train_model.py  # Train the model")
        print("2. streamlit run streamlit_app.py  # Run the app")
    else:
        print("❌ Setup completed with warnings. Check demo output above.")

if __name__ == "__main__":
    main()
