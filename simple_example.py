#!/usr/bin/env python3
"""
Simple example of plagiarism detection
"""

import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

def simple_text_check():
    """Simple example: Check text for plagiarism"""
    
    # Import the detector
    from plagiarism_detector import PlagiarismDetector
    
    # Initialize detector with your API keys
    detector = PlagiarismDetector(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),  # Optional
        google_cse_id=os.getenv("GOOGLE_CSE_ID")     # Optional
    )
    
    # Text to analyze (replace with your own)
    sample_text = """
    Artificial intelligence has become one of the most transformative technologies 
    of our time. Machine learning algorithms can process vast amounts of data to 
    identify patterns and make predictions. Deep learning represents a subset of 
    machine learning that uses neural networks to mimic the human brain's 
    functionality. These technologies have applications across various sectors 
    including healthcare, finance, transportation, and education.
    """
    
    print("🔍 Analyzing text for plagiarism...")
    print("Text preview:", sample_text[:100] + "...")
    print("-" * 50)
    
    # Run the analysis
    results = detector.analyze_text(sample_text, web_search_enabled=True)
    
    # Display results
    print(f"📊 RESULTS:")
    print(f"Plagiarism Percentage: {results['overall_plagiarism_percentage']:.1f}%")
    print(f"Chunks Analyzed: {len(results['detected_plagiarism'])}")
    print(f"Sources Checked: {results['summary']['web_sources_checked']}")
    
    # Show plagiarized sections if any
    plagiarized_count = len([chunk for chunk in results['detected_plagiarism'] if chunk['is_plagiarized']])
    if plagiarized_count > 0:
        print(f"⚠️  Plagiarized Sections Found: {plagiarized_count}")
        
        # Show suggestions if available
        if results.get('suggestions'):
            print(f"💡 Suggestions Available: {len(results['suggestions'])}")
            print("First suggestion preview:")
            print(results['suggestions'][0][:200] + "...")
    else:
        print("✅ No significant plagiarism detected!")
    
    # Generate and display full report
    print("\n" + "="*60)
    print("DETAILED REPORT")
    print("="*60)
    
    report = detector.format_report(results)
    print(report)
    
    return results

def analyze_your_text():
    """Interactive function to analyze your own text"""
    
    print("\n" + "="*50)
    print("ANALYZE YOUR OWN TEXT")
    print("="*50)
    
    print("Enter your text to analyze (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    
    user_text = "\n".join(lines)
    
    if not user_text.strip():
        print("No text entered. Using sample text.")
        return simple_text_check()
    
    # Analyze user's text
    from plagiarism_detector import PlagiarismDetector
    
    detector = PlagiarismDetector(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        google_cse_id=os.getenv("GOOGLE_CSE_ID")
    )
    
    print(f"\n🔍 Analyzing your text ({len(user_text)} characters)...")
    
    results = detector.analyze_text(user_text, web_search_enabled=True)
    
    # Quick summary
    print(f"\n📊 QUICK RESULTS:")
    print(f"Plagiarism Score: {results['overall_plagiarism_percentage']:.1f}%")
    
    if results['overall_plagiarism_percentage'] > 30:
        print("🔴 High plagiarism detected!")
    elif results['overall_plagiarism_percentage'] > 15:
        print("🟡 Moderate plagiarism detected")
    else:
        print("🟢 Low plagiarism detected")
    
    # Detailed report
    report = detector.format_report(results)
    print("\n" + report)
    
    return results

def main():
    """Main function"""
    print("🚀 Plagiarism Detection - Simple Example")
    print("="*50)
    
    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please check your .env file")
        return
    
    try:
        # Option 1: Run with sample text
        print("Choose an option:")
        print("1. Test with sample text")
        print("2. Analyze your own text")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "2":
            results = analyze_your_text()
        else:
            results = simple_text_check()
        
        # Save results
        import json
        with open("plagiarism_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to 'plagiarism_results.json'")
        
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your API keys and internet connection")

if __name__ == "__main__":
    main()
