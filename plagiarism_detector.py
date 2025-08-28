import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from typing import List, Dict, Tuple, Any
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import json
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

@dataclass
class PlagiarismResult:
    similarity_score: float
    matched_content: str
    source: str
    detection_method: str
    suggestions: List[str]

class PlagiarismDetector:
    def __init__(self, openai_api_key: str, google_api_key: str = None, google_cse_id: str = None):
        """
        Initialize the plagiarism detector with API keys
        
        Args:
            openai_api_key: OpenAI API key
            google_api_key: Google Custom Search API key (optional)
            google_cse_id: Google Custom Search Engine ID (optional)
        """
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Google Search API setup
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        
        # Initialize sentence transformer model
        print("Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        
        # Stop words for text cleaning
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        print("Plagiarism detector initialized successfully!")
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 3) -> List[str]:
        """Split text into sentences and create overlapping chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(len(sentences)):
            # Create chunks of consecutive sentences
            chunk = ' '.join(sentences[i:i+chunk_size])
            if len(chunk.strip()) > 20:  # Only include substantial chunks
                chunks.append(chunk.strip())
        
        return chunks
    
    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        # Tokenize and convert to sets
        words1 = set(word_tokenize(text1.lower()))
        words2 = set(word_tokenize(text2.lower()))
        
        # Remove stop words
        words1 = words1 - self.stop_words
        words2 = words2 - self.stop_words
        
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def cosine_similarity_tfidf(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity using TF-IDF"""
        try:
            # Fit and transform texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence embeddings"""
        try:
            # Generate embeddings
            embeddings = self.sentence_model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Search web using Google Custom Search API"""
        if not self.google_api_key or not self.google_cse_id:
            print("Google API credentials not provided. Skipping web search.")
            return []
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_cse_id,
                'q': query,
                'num': num_results
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                results = response.json()
                web_results = []
                
                for item in results.get('items', []):
                    web_results.append({
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'link': item.get('link', ''),
                        'content': item.get('snippet', '')  # Using snippet as content
                    })
                
                return web_results
            else:
                print(f"Web search failed: {response.status_code}")
                return []
        
        except Exception as e:
            print(f"Web search error: {str(e)}")
            return []
    
    def check_smallseotools(self, text: str) -> Dict[str, Any]:
        """
        Placeholder for SmallSEOTools API integration
        Note: SmallSEOTools doesn't have a public API, this is a placeholder
        """
        # This would require web scraping or finding an alternative API
        # For now, returning a placeholder
        return {
            'plagiarism_percentage': 0,
            'sources': [],
            'available': False,
            'message': 'SmallSEOTools API not available - consider web scraping implementation'
        }
    
    def generate_suggestions(self, original_text: str, plagiarized_parts: List[str]) -> List[str]:
        """Generate suggestions to remove plagiarism using OpenAI"""
        try:
            suggestions = []
            
            for part in plagiarized_parts[:3]:  # Limit to first 3 parts to avoid token limits
                prompt = f"""
                The following text appears to be plagiarized:
                "{part}"
                
                Please provide 3 different ways to rewrite this text to remove plagiarism while maintaining the original meaning:
                1. Paraphrasing with different sentence structure
                2. Using synonyms and different vocabulary
                3. Restructuring the content with original analysis
                
                Provide only the rewritten versions, numbered 1-3.
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a writing assistant that helps rewrite text to avoid plagiarism while maintaining meaning and quality."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                suggestion = response.choices[0].message.content.strip()
                suggestions.append(f"For text: '{part[:50]}...'\n{suggestion}\n")
            
            return suggestions
            
        except Exception as e:
            print(f"Error generating suggestions: {str(e)}")
            return ["Unable to generate suggestions. Please try manual paraphrasing."]
    
    def analyze_text(self, input_text: str, web_search_enabled: bool = True) -> Dict[str, Any]:
        """
        Main method to analyze text for plagiarism
        
        Args:
            input_text: Text to analyze
            web_search_enabled: Whether to perform web search
            
        Returns:
            Dictionary containing analysis results
        """
        print("Starting plagiarism analysis...")
        
        # Clean and chunk the input text
        cleaned_text = self.clean_text(input_text)
        chunks = self.chunk_text(cleaned_text)
        
        results = {
            'overall_plagiarism_percentage': 0.0,
            'detected_plagiarism': [],
            'analysis_methods': {},
            'suggestions': [],
            'summary': {}
        }
        
        plagiarized_parts = []
        total_similarity_scores = []
        
        print(f"Analyzing {len(chunks)} text chunks...")
        
        # Analyze each chunk
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            chunk_results = {
                'text': chunk,
                'similarities': [],
                'max_similarity': 0.0,
                'is_plagiarized': False,
                'sources': []
            }
            
            # Web search analysis
            if web_search_enabled:
                # Extract key phrases for search
                words = word_tokenize(chunk.lower())
                key_phrases = [word for word in words if word not in self.stop_words and len(word) > 3]
                search_query = ' '.join(key_phrases[:6])  # Use first 6 meaningful words
                
                web_results = self.search_web(search_query)
                
                for result in web_results:
                    web_content = result['content']
                    
                    # Calculate multiple similarities
                    jaccard_sim = self.jaccard_similarity(chunk, web_content)
                    cosine_sim = self.cosine_similarity_tfidf(chunk, web_content)
                    semantic_sim = self.semantic_similarity(chunk, web_content)
                    
                    # Weighted average of similarities
                    combined_similarity = (jaccard_sim * 0.2 + cosine_sim * 0.4 + semantic_sim * 0.4)
                    
                    if combined_similarity > 0.3:  # Threshold for potential plagiarism
                        chunk_results['similarities'].append({
                            'source': result['link'],
                            'title': result['title'],
                            'jaccard': jaccard_sim,
                            'cosine_tfidf': cosine_sim,
                            'semantic': semantic_sim,
                            'combined': combined_similarity,
                            'matched_content': web_content
                        })
                        
                        chunk_results['max_similarity'] = max(chunk_results['max_similarity'], combined_similarity)
            
            # Determine if chunk is plagiarized
            if chunk_results['max_similarity'] > 0.5:  # Threshold for plagiarism
                chunk_results['is_plagiarized'] = True
                plagiarized_parts.append(chunk)
            
            total_similarity_scores.append(chunk_results['max_similarity'])
            results['detected_plagiarism'].append(chunk_results)
        
        # Calculate overall plagiarism percentage
        if total_similarity_scores:
            results['overall_plagiarism_percentage'] = np.mean(total_similarity_scores) * 100
        
        # Secondary check with SmallSEOTools (placeholder)
        smallseo_results = self.check_smallseotools(cleaned_text)
        results['analysis_methods']['smallseotools'] = smallseo_results
        
        # Generate suggestions if plagiarism detected
        if plagiarized_parts:
            print("Generating suggestions for plagiarized content...")
            results['suggestions'] = self.generate_suggestions(cleaned_text, plagiarized_parts)
        
        # Create summary
        plagiarized_chunks = len([chunk for chunk in results['detected_plagiarism'] if chunk['is_plagiarized']])
        results['summary'] = {
            'total_chunks_analyzed': len(chunks),
            'plagiarized_chunks': plagiarized_chunks,
            'plagiarism_percentage': results['overall_plagiarism_percentage'],
            'highest_similarity': max(total_similarity_scores) if total_similarity_scores else 0,
            'web_sources_checked': len(web_results) if web_search_enabled else 0
        }
        
        print("Analysis complete!")
        return results
    
    def format_report(self, results: Dict[str, Any]) -> str:
        """Format analysis results into a readable report"""
        report = []
        report.append("=" * 60)
        report.append("PLAGIARISM DETECTION REPORT")
        report.append("=" * 60)
        
        summary = results['summary']
        report.append(f"\nOVERALL ANALYSIS:")
        report.append(f"• Plagiarism Percentage: {summary['plagiarism_percentage']:.1f}%")
        report.append(f"• Total Chunks Analyzed: {summary['total_chunks_analyzed']}")
        report.append(f"• Plagiarized Chunks: {summary['plagiarized_chunks']}")
        report.append(f"• Highest Similarity: {summary['highest_similarity']:.1f}%")
        report.append(f"• Web Sources Checked: {summary['web_sources_checked']}")
        
        # Detailed results
        if results['detected_plagiarism']:
            report.append(f"\nDETAILED ANALYSIS:")
            for i, chunk_result in enumerate(results['detected_plagiarism']):
                if chunk_result['is_plagiarized']:
                    report.append(f"\n--- PLAGIARIZED CONTENT #{i+1} ---")
                    report.append(f"Text: {chunk_result['text'][:100]}...")
                    report.append(f"Max Similarity: {chunk_result['max_similarity']:.1f}%")
                    
                    if chunk_result['similarities']:
                        best_match = max(chunk_result['similarities'], key=lambda x: x['combined'])
                        report.append(f"Best Match Source: {best_match['source']}")
                        report.append(f"Similarity Scores:")
                        report.append(f"  - Jaccard: {best_match['jaccard']:.3f}")
                        report.append(f"  - TF-IDF Cosine: {best_match['cosine_tfidf']:.3f}")
                        report.append(f"  - Semantic: {best_match['semantic']:.3f}")
        
        # Suggestions
        if results['suggestions']:
            report.append(f"\nSUGGESTIONS TO REMOVE PLAGIARISM:")
            report.append("-" * 40)
            for suggestion in results['suggestions']:
                report.append(f"\n{suggestion}")
        
        return "\n".join(report)

# Example usage and testing
def main():
    # Initialize detector (you need to provide your API keys)
    OPENAI_API_KEY = "your-openai-api-key-here"
    GOOGLE_API_KEY = "your-google-api-key-here"  # Optional
    GOOGLE_CSE_ID = "your-cse-id-here"  # Optional
    
    detector = PlagiarismDetector(
        openai_api_key=OPENAI_API_KEY,
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID
    )
    
    # Example text to analyze
    sample_text = """
    Artificial intelligence is transforming the way we live and work. 
    Machine learning algorithms can process vast amounts of data to identify 
    patterns and make predictions. Deep learning, a subset of machine learning, 
    uses neural networks to mimic human brain functions. This technology has 
    applications in healthcare, finance, transportation, and many other sectors.
    """
    
    print("Testing plagiarism detector...")
    results = detector.analyze_text(sample_text, web_search_enabled=True)
    
    # Print formatted report
    report = detector.format_report(results)
    print(report)
    
    # Save results to JSON
    with open('plagiarism_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to 'plagiarism_results.json'")

if __name__ == "__main__":
    main()