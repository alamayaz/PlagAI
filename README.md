# Plagiarism Detection System Setup Guide

## 📋 Requirements

### Python Dependencies

Create a `requirements.txt` file:

```
# Core dependencies
langgraph>=0.0.40
langchain>=0.1.0
openai>=1.0.0
requests>=2.31.0
numpy>=1.24.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
nltk>=3.8.0

# File processing
PyPDF2>=3.0.0
python-docx>=0.8.11
openpyxl>=3.1.0

# Additional utilities
python-dotenv>=1.0.0
tqdm>=4.65.0
pandas>=2.0.0
```

### Installation

```bash
# Create virtual environment
python -m venv plagiarism_env
source plagiarism_env/bin/activate  # On Windows: plagiarism_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 🔑 API Keys Setup

### 1. OpenAI API Key (Required)
- Go to [OpenAI API](https://platform.openai.com/api-keys)
- Create an account and generate an API key
- Add billing information for usage

### 2. Google Custom Search API (Optional but Recommended)
- Go to [Google Cloud Console](https://console.cloud.google.com/)
- Enable the Custom Search JSON API
- Create credentials (API key)
- Set up a Custom Search Engine at [Google CSE](https://cse.google.com/)

### 3. Environment Variables
Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here
```

## 🚀 Usage Examples

### 1. Basic Text Analysis

```python
from plagiarism_detector import PlagiarismDetector
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize detector
detector = PlagiarismDetector(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID")
)

# Analyze text
text = "Your text to analyze here..."
results = detector.analyze_text(text)

# Print report
report = detector.format_report(results)
print(report)
```

### 2. Using LangGraph Agent

```python
from langgraph_plagiarism_agent import PlagiarismDetectionAgent
import os

# Initialize agent
agent = PlagiarismDetectionAgent(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID")
)

# Analyze text
results = agent.run_analysis(input_text="Your text here...")
print(f"Plagiarism: {results['plagiarism_percentage']:.1f}%")

# Analyze file
results = agent.run_analysis(input_file="document.pdf")
```

### 3. Jupyter Notebook Interface

```python
from langgraph_plagiarism_agent import PlagiarismDetectorNotebook
import os

# Initialize notebook interface
detector = PlagiarismDetectorNotebook(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID")
)

# Check text with automatic display
results = detector.check_text("Your text to analyze...")

# Get suggestions if needed
detector.get_suggestions(results)

# Compare two texts directly
similarity = detector.compare_texts(text1, text2)
print(similarity)
```

### 4. Command Line Interface

```bash
# Analyze text
python langgraph_plagiarism_agent.py \
    --text "Your text here" \
    --openai-key "your_key" \
    --output "results.json"

# Analyze file
python langgraph_plagiarism_agent.py \
    --file "document.pdf" \
    --openai-key "your_key" \
    --google-key "your_google_key" \
    --google-cse "your_cse_id"

# Batch processing
python langgraph_plagiarism_agent.py \
    --batch "batch_inputs.json" \
    --openai-key "your_key"
```

### 5. Batch Input Format

Create `batch_inputs.json`:

```json
[
    {"text": "First text to analyze..."},
    {"file": "document1.pdf"},
    {"text": "Another text sample..."},
    {"file": "document2.docx"}
]
```

## 🔧 Configuration Options

### Detection Thresholds

You can modify these in the code:

```python
# Similarity thresholds
PLAGIARISM_THRESHOLD = 0.5  # Consider as plagiarized
WEB_SEARCH_THRESHOLD = 0.3  # Include in web search results
SUGGESTION_THRESHOLD = 20   # Generate suggestions above this percentage

# Text processing
CHUNK_SIZE = 3  # Number of sentences per chunk
MAX_SEARCH_RESULTS = 5  # Web search results per query
```

### Supported File Formats

- **PDF**: `.pdf` files
- **Word**: `.docx` files  
- **Text**: `.txt` files
- **Direct text**: String input

## 📊 Output Format

### Analysis Results Structure

```json
{
    "status": "completed",
    "plagiarism_percentage": 25.5,
    "plagiarized_sections_count": 3,
    "sources_found": ["url1", "url2"],
    "suggestions": ["Suggestion 1...", "Suggestion 2..."],
    "methods_used": ["jaccard_similarity", "semantic_similarity", "web_search"],
    "detailed_report": "Full formatted report...",
    "full_results": {
        "overall_plagiarism_percentage": 25.5,
        "detected_plagiarism": [...],
        "analysis_methods": {...},
        "suggestions": [...],
        "summary": {...}
    }
}
```

## ⚠️ Important Notes

### API Costs
- **OpenAI**: ~$0.002 per 1K tokens (GPT-3.5-turbo)
- **Google Custom Search**: Free tier: 100 queries/day, then $5/1000 queries
- Monitor usage to avoid unexpected charges

### Rate Limits
- OpenAI: 3,500 requests/minute (tier dependent)
- Google CSE: 100 queries/day (free), 10,000/day (paid)

### Privacy Considerations
- Web search sends query terms to Google
- OpenAI processes text for suggestions
- Consider local-only mode for sensitive content

### Performance Tips
- Use smaller text chunks for faster processing
- Limit web search results for cost efficiency
- Cache results for repeated analysis
- Use batch processing for multiple documents

## 🐛 Troubleshooting

### Common Issues

1. **NLTK Data Missing**
   ```bash
   python -c "import nltk; nltk.download('all')"
   ```

2. **Sentence Transformers Model Download**
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')  # Downloads automatically
   ```

3. **Google API Quota Exceeded**
   - Check Google Cloud Console quota
   - Verify billing is enabled
   - Consider upgrading to paid tier

4. **OpenAI API Errors**
   - Verify API key is valid
   - Check account billing status
   - Monitor rate limits

### Getting Help

1. Check the error messages in the output
2. Verify all API keys are correctly set
3. Ensure all dependencies are installed
4. Test with simple examples first

## 🔄 Updates and Maintenance

### Keeping Dependencies Updated

```bash
pip install --upgrade -r requirements.txt
```

### Model Updates

The sentence transformer model may be updated periodically. To force a fresh download:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=None)
```

This comprehensive setup should get you started with a robust plagiarism detection system!