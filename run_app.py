#!/usr/bin/env python3
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
        print("\nApp stopped by user")
    except Exception as e:
        print(f"Error running app: {e}")

if __name__ == "__main__":
    main()
