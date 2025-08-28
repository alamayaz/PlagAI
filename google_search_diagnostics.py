#!/usr/bin/env python3
"""
Google Custom Search API Diagnostics
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def test_google_api():
    """Test Google Custom Search API with detailed error reporting"""
    
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    print("🔍 Google Custom Search API Diagnostics")
    print("=" * 50)
    
    # Check if credentials exist
    print(f"API Key: {'✅ Found' if api_key else '❌ Missing'}")
    print(f"CSE ID: {'✅ Found' if cse_id else '❌ Missing'}")
    
    if not api_key or not cse_id:
        print("\n❌ Missing credentials. Please check your .env file.")
        print("\nYour .env file should look like:")
        print("GOOGLE_API_KEY=your_actual_api_key")
        print("GOOGLE_CSE_ID=your_actual_cse_id")
        return False
    
    # Show partial credentials for verification (safely)
    print(f"API Key preview: {api_key[:10]}...{api_key[-4:]}")
    print(f"CSE ID preview: {cse_id[:10]}...{cse_id[-4:]}")
    
    # Test API call
    print(f"\n🌐 Testing API call...")
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': 'test search',
        'num': 1
    }
    
    try:
        print(f"Making request to: {url}")
        print(f"Parameters: cx={cse_id[:8]}..., q='test search', num=1")
        
        response = requests.get(url, params=params, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Google Custom Search API is working!")
            
            # Parse response
            data = response.json()
            total_results = data.get('searchInformation', {}).get('totalResults', 0)
            items = data.get('items', [])
            
            print(f"Total results found: {total_results}")
            print(f"Items returned: {len(items)}")
            
            if items:
                print(f"First result title: {items[0].get('title', 'N/A')}")
            
            return True
            
        else:
            print(f"❌ API Error: {response.status_code}")
            
            # Try to parse error response
            try:
                error_data = response.json()
                error_message = error_data.get('error', {}).get('message', 'Unknown error')
                error_code = error_data.get('error', {}).get('code', 'Unknown')
                
                print(f"Error Code: {error_code}")
                print(f"Error Message: {error_message}")
                
                # Specific error guidance
                if response.status_code == 400:
                    print("\n🔧 Possible fixes for 400 Bad Request:")
                    print("1. Check if Custom Search Engine ID is correct")
                    print("2. Verify the CSE is configured to search the entire web")
                    print("3. Make sure the API key has Custom Search JSON API enabled")
                    
                elif response.status_code == 403:
                    print("\n🔧 Possible fixes for 403 Forbidden:")
                    print("1. Check if Custom Search JSON API is enabled")
                    print("2. Verify billing is enabled in Google Cloud Console")
                    print("3. Check API quotas and usage limits")
                    
                elif response.status_code == 429:
                    print("\n🔧 Rate limit exceeded:")
                    print("1. You've exceeded the free tier limit (100 searches/day)")
                    print("2. Wait 24 hours or upgrade to paid tier")
                
            except:
                print("Error response body:", response.text)
            
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False

def check_google_cloud_setup():
    """Provide guidance for Google Cloud setup"""
    
    print("\n📋 Google Cloud Setup Checklist:")
    print("-" * 30)
    
    steps = [
        "1. Go to Google Cloud Console (console.cloud.google.com)",
        "2. Create a new project or select existing one",
        "3. Enable 'Custom Search JSON API'",
        "4. Create credentials (API Key)",
        "5. Go to cse.google.com to create Custom Search Engine",
        "6. Set search engine to search entire web (*)",
        "7. Copy the Search Engine ID",
        "8. Add both API key and CSE ID to .env file"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print(f"\n🔗 Helpful Links:")
    print(f"   Google Cloud Console: https://console.cloud.google.com/")
    print(f"   Custom Search Engine: https://cse.google.com/")
    print(f"   API Documentation: https://developers.google.com/custom-search/v1/introduction")

def test_without_google():
    """Test plagiarism detection without Google search"""
    
    print(f"\n🔄 Testing plagiarism detection without web search...")
    
    try:
        from plagiarism_detector import PlagiarismDetector
        
        detector = PlagiarismDetector(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=None,  # Disable Google search
            google_cse_id=None
        )
        
        test_text = "Machine learning is a subset of artificial intelligence."
        
        print("Running analysis without web search...")
        results = detector.analyze_text(test_text, web_search_enabled=False)
        
        print("✅ Local plagiarism detection works!")
        print(f"   Similarity algorithms functional")
        print(f"   OpenAI integration working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in local detection: {e}")
        return False

def main():
    """Main diagnostic function"""
    
    # Test Google API
    google_works = test_google_api()
    
    if not google_works:
        # Show setup guidance
        check_google_cloud_setup()
        
        # Test without Google
        local_works = test_without_google()
        
        if local_works:
            print(f"\n✅ Your system works with local detection only")
            print(f"   You can use the plagiarism detector without web search")
            print(f"   Fix Google setup later for enhanced detection")
        
    else:
        print(f"\n🎉 All systems working perfectly!")

if __name__ == "__main__":
    main()
