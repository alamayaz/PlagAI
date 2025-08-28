#!/usr/bin/env python3
"""
Setup script for Streamlit Plagiarism Detection App
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    requirements = [
        "streamlit>=1.28.0",
        "plotly>=5.15.0", 
        "pandas>=2.0.0",
        "openai>=1.3.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0"
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    try:
        import nltk
        import ssl
        
        # Handle SSL issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required datasets
        datasets = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
        
        for dataset in datasets:
            print(f"Downloading {dataset}...")
            nltk.download(dataset, quiet=True)
            print(f"✅ {dataset} downloaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def create_sample_env():
    """Create a sample .env file"""
    print("\n📝 Creating sample .env file...")
    
    env_content = """# Plagiarism Detection App Configuration
# Replace with your actual API keys

# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Google Custom Search (for web search functionality)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_google_custom_search_engine_id_here

# App Configuration
APP_TITLE="AI Plagiarism Detective"
APP_THEME="light"
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Sample .env file created")
        print("   Please edit .env file with your actual API keys")
    else:
        print("⚠️  .env file already exists - not overwriting")

def test_streamlit_installation():
    """Test if Streamlit is properly installed"""
    print("\n🧪 Testing Streamlit installation...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        # Test plotly
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
        
        # Test pandas
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_run_script():
    """Create a script to run the Streamlit app"""
    print("\n📄 Creating run script...")
    
    run_script_content = """#!/usr/bin/env python3
'''
Run script for Streamlit Plagiarism Detection App
'''

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("Starting AI Plagiarism Detective...")
    
    # Check if .env file exists
    if not Path(".env").exists():
        print("Warning: .env file not found. Please create one with your API keys.")
        print("   You can copy from .env.example if available.")
    
    # Check if required files exist
    required_files = ["streamlit_plagiarism_app.py", "plagiarism_detector.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        print("   Please ensure all project files are in the current directory.")
        return
    
    # Run Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_plagiarism_app.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\\nApp stopped by user")
    except Exception as e:
        print(f"Error running app: {e}")

if __name__ == "__main__":
    main()
"""
    
    with open("run_app.py", "w", encoding='utf-8') as f:
        f.write(run_script_content)
    
    # Make it executable on Unix systems
    if os.name != 'nt':
        os.chmod("run_app.py", 0o755)
    
    print("✅ Run script created: run_app.py")

def main():
    """Main setup function"""
    print("🎯 Setting up Streamlit Plagiarism Detection App")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check your internet connection and try again.")
        return
    
    # Step 2: Download NLTK data
    if not download_nltk_data():
        print("⚠️  NLTK data download failed. You may need to download manually.")
    
    # Step 3: Create sample .env file
    create_sample_env()
    
    # Step 4: Test installation
    if not test_streamlit_installation():
        print("❌ Streamlit installation test failed.")
        return
    
    # Step 5: Create run script
    create_run_script()
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit the .env file with your actual API keys")
    print("2. Ensure these files are in your directory:")
    print("   - streamlit_plagiarism_app.py")
    print("   - plagiarism_detector.py")
    print("   - enhanced_plagiarism_detector.py (optional)")
    print("\n🚀 To run the app:")
    print("   python run_app.py")
    print("   OR")
    print("   streamlit run streamlit_plagiarism_app.py")
    print("\n🌐 The app will open in your browser at http://localhost:8501")

if __name__ == "__main__":
    main()