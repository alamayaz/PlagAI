#!/usr/bin/env python3
"""
Quick test script for plagiarism detection system
Run this to verify everything is working correctly
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_installation():
    """Test if all dependencies are installed correctly"""
    print("🔧 Testing installation...")
    
    try:
        import langgraph
        print("✅ LangGraph installed")
        
        import openai
        print("✅ OpenAI installed")
        
        import sentence_transformers
        print("✅ Sentence Transformers installed")
        
        import sklearn
        print("✅ Scikit-learn installed")
        
        import nltk
        print("✅ NLTK installed")
        
        print("🎉 All dependencies installed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def test_api_keys():
    """Test if API keys are properly set"""
    print("\n🔑 Testing API keys...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    google_cse = os.getenv("GOOGLE_CSE_ID")
    
    if openai_key and openai_key != "your_openai_api_key_here":
        print("✅ OpenAI API key found")
    else:
        print("❌ OpenAI API key missing or not set")
        return False
    
    if google_key and google_key != "your_google_api_key_here":
        print("✅ Google API key found")
    else:
        print("⚠️  Google API key missing (optional but recommended)")
    
    if google_cse and google_cse != "your_custom_search_engine_id_here":
        print("✅ Google CSE ID found")
    else:
        print("⚠️  Google CSE ID missing (optional but recommended)")
    
    return True

def test_basic_functionality():
    """Test basic plagiarism detection functionality"""
    print("\n🔍 Testing basic functionality...")
    
    try:
        # Import our plagiarism detector
        from plagiarism_detector import PlagiarismDetector
        
        # Initialize with API keys
        detector = PlagiarismDetector(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_cse_id=os.getenv("GOOGLE_CSE_ID")
        )
        
        print("✅ Plagiarism detector initialized")
        
        # Test with a simple text
        test_text = """
        Machine learning is a subset of artificial intelligence that enables 
        computers to learn and improve from experience without being explicitly 
        programmed. It focuses on the development of algorithms that can access 
        data and use it to learn for themselves.
        """
        
        print("🔍 Running plagiarism analysis...")
        
        # Run analysis (without web search for quick test)
        results = detector.analyze_text(test_text, web_search_enabled=False)
        
        print(f"✅ Analysis completed!")
        print(f"   Plagiarism percentage: {results['overall_plagiarism_percentage']:.1f}%")
        print(f"   Chunks analyzed: {len(results['detected_plagiarism'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in functionality test: {e}")
        return False

def test_with_web_search():
    """Test with web search enabled"""
    print("\n🌐 Testing web search functionality...")
    
    google_key = os.getenv("GOOGLE_API_KEY")
    google_cse = os.getenv("GOOGLE_CSE_ID")
    
    if not google_key or not google_cse:
        print("⚠️  Skipping web search test - Google credentials not available")
        return True
    
    try:
        from plagiarism_detector import PlagiarismDetector
        
        detector = PlagiarismDetector(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=google_key,
            google_cse_id=google_cse
        )
        
        # Test with a well-known text snippet
        test_text = "Climate change is a long-term change in the average weather patterns."
        
        print("🔍 Running analysis with web search...")
        results = detector.analyze_text(test_text, web_search_enabled=True)
        
        print(f"✅ Web search analysis completed!")
        print(f"   Sources checked: {results['summary']['web_sources_checked']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in web search test: {e}")
        print("   This might be due to API quota or network issues")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Plagiarism Detection System Tests")
    print("=" * 50)
    
    # Test 1: Installation
    if not test_installation():
        print("\n❌ Installation test failed. Please install missing dependencies.")
        return
    
    # Test 2: API Keys
    if not test_api_keys():
        print("\n❌ API key test failed. Please check your .env file.")
        return
    
    # Test 3: Basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality test failed.")
        return
    
    # Test 4: Web search (optional)
    test_with_web_search()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed! Your system is ready to use.")
    print("\nNext steps:")
    print("1. Try the simple example: python simple_example.py")
    print("2. Use the LangGraph agent for advanced features")
    print("3. Check the documentation for more usage examples")

if __name__ == "__main__":
    main()
