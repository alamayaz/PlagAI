from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Any, Dict, List
import operator
from enum import Enum
import asyncio
import json
from pathlib import Path
import PyPDF2
import docx
import io

# Import our plagiarism detector
from plagiarism_detector import PlagiarismDetector, PlagiarismResult

class DetectionStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class AgentState(TypedDict):
    # Input
    input_text: str
    input_file_path: str
    file_type: str
    
    # Processing state
    status: DetectionStatus
    current_step: str
    error_message: str
    
    # Extracted content
    extracted_text: str
    text_chunks: List[str]
    
    # Detection results
    plagiarism_results: Dict[str, Any]
    similarity_scores: List[float]
    plagiarized_sections: List[Dict[str, Any]]
    
    # Final output
    plagiarism_percentage: float
    detailed_report: str
    suggestions: List[str]
    sources_found: List[str]
    
    # Metadata
    processing_time: float
    methods_used: List[str]

class PlagiarismDetectionAgent:
    def __init__(self, openai_api_key: str, google_api_key: str = None, google_cse_id: str = None):
        """Initialize the plagiarism detection agent"""
        self.detector = PlagiarismDetector(
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
            google_cse_id=google_cse_id
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("parse_input", self.parse_input)
        workflow.add_node("extract_text", self.extract_text)
        workflow.add_node("preprocess_text", self.preprocess_text)
        workflow.add_node("detect_plagiarism", self.detect_plagiarism)
        workflow.add_node("analyze_results", self.analyze_results)
        workflow.add_node("generate_suggestions", self.generate_suggestions)
        workflow.add_node("create_report", self.create_report)
        workflow.add_node("handle_error", self.handle_error)
        
        # Define the workflow
        workflow.set_entry_point("parse_input")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "parse_input",
            self.should_extract_text,
            {
                "extract": "extract_text",
                "preprocess": "preprocess_text",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("extract_text", "preprocess_text")
        workflow.add_edge("preprocess_text", "detect_plagiarism")
        
        workflow.add_conditional_edges(
            "detect_plagiarism",
            self.should_analyze_results,
            {
                "analyze": "analyze_results",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "analyze_results",
            self.should_generate_suggestions,
            {
                "suggestions": "generate_suggestions",
                "report": "create_report"
            }
        )
        
        workflow.add_edge("generate_suggestions", "create_report")
        workflow.add_edge("create_report", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def parse_input(self, state: AgentState) -> AgentState:
        """Parse and validate input"""
        print("📄 Parsing input...")
        
        try:
            state["status"] = DetectionStatus.PROCESSING
            state["current_step"] = "parsing_input"
            state["methods_used"] = []
            
            # Determine input type
            if state.get("input_file_path"):
                file_path = Path(state["input_file_path"])
                if file_path.exists():
                    state["file_type"] = file_path.suffix.lower()
                else:
                    state["status"] = DetectionStatus.FAILED
                    state["error_message"] = f"File not found: {state['input_file_path']}"
                    return state
            elif state.get("input_text"):
                state["file_type"] = "text"
            else:
                state["status"] = DetectionStatus.FAILED
                state["error_message"] = "No input text or file provided"
                return state
                
            print(f"✅ Input parsed successfully. Type: {state['file_type']}")
            return state
            
        except Exception as e:
            state["status"] = DetectionStatus.FAILED
            state["error_message"] = f"Error parsing input: {str(e)}"
            return state
    
    def extract_text(self, state: AgentState) -> AgentState:
        """Extract text from various file formats"""
        print("📖 Extracting text from file...")
        
        try:
            state["current_step"] = "extracting_text"
            file_path = state["input_file_path"]
            file_type = state["file_type"]
            
            if file_type == ".pdf":
                state["extracted_text"] = self._extract_from_pdf(file_path)
            elif file_type == ".docx":
                state["extracted_text"] = self._extract_from_docx(file_path)
            elif file_type == ".txt":
                state["extracted_text"] = self._extract_from_txt(file_path)
            else:
                state["status"] = DetectionStatus.FAILED
                state["error_message"] = f"Unsupported file type: {file_type}"
                return state
            
            if not state["extracted_text"].strip():
                state["status"] = DetectionStatus.FAILED
                state["error_message"] = "No text extracted from file"
                return state
            
            print(f"✅ Text extracted successfully. Length: {len(state['extracted_text'])} characters")
            return state
            
        except Exception as e:
            state["status"] = DetectionStatus.FAILED
            state["error_message"] = f"Error extracting text: {str(e)}"
            return state
    
    def preprocess_text(self, state: AgentState) -> AgentState:
        """Preprocess and clean text"""
        print("🔧 Preprocessing text...")
        
        try:
            state["current_step"] = "preprocessing_text"
            
            # Use extracted text or input text
            text = state.get("extracted_text") or state.get("input_text", "")
            
            # Clean and chunk text using detector methods
            cleaned_text = self.detector.clean_text(text)
            chunks = self.detector.chunk_text(cleaned_text)
            
            state["extracted_text"] = cleaned_text
            state["text_chunks"] = chunks
            
            print(f"✅ Text preprocessed. Chunks created: {len(chunks)}")
            return state
            
        except Exception as e:
            state["status"] = DetectionStatus.FAILED
            state["error_message"] = f"Error preprocessing text: {str(e)}"
            return state
    
    def detect_plagiarism(self, state: AgentState) -> AgentState:
        """Run plagiarism detection"""
        print("🔍 Running plagiarism detection...")
        
        try:
            state["current_step"] = "detecting_plagiarism"
            text = state["extracted_text"]
            
            # Run the main detection
            results = self.detector.analyze_text(text, web_search_enabled=True)
            state["plagiarism_results"] = results
            
            # Extract key metrics
            state["plagiarism_percentage"] = results["overall_plagiarism_percentage"]
            state["plagiarized_sections"] = [
                chunk for chunk in results["detected_plagiarism"] 
                if chunk["is_plagiarized"]
            ]
            
            # Track methods used
            methods = ["jaccard_similarity", "cosine_similarity_tfidf", "semantic_similarity"]
            if results["summary"]["web_sources_checked"] > 0:
                methods.append("web_search")
            state["methods_used"] = methods
            
            print(f"✅ Plagiarism detection completed. Score: {state['plagiarism_percentage']:.1f}%")
            return state
            
        except Exception as e:
            state["status"] = DetectionStatus.FAILED
            state["error_message"] = f"Error in plagiarism detection: {str(e)}"
            return state
    
    def analyze_results(self, state: AgentState) -> AgentState:
        """Analyze detection results"""
        print("📊 Analyzing results...")
        
        try:
            state["current_step"] = "analyzing_results"
            results = state["plagiarism_results"]
            
            # Extract sources
            sources = set()
            for section in state["plagiarized_sections"]:
                for similarity in section.get("similarities", []):
                    sources.add(similarity["source"])
            
            state["sources_found"] = list(sources)
            
            # Extract similarity scores
            similarity_scores = []
            for chunk in results["detected_plagiarism"]:
                similarity_scores.append(chunk["max_similarity"])
            
            state["similarity_scores"] = similarity_scores
            
            print(f"✅ Analysis completed. Sources found: {len(sources)}")
            return state
            
        except Exception as e:
            state["status"] = DetectionStatus.FAILED
            state["error_message"] = f"Error analyzing results: {str(e)}"
            return state
    
    def generate_suggestions(self, state: AgentState) -> AgentState:
        """Generate suggestions to remove plagiarism"""
        print("💡 Generating suggestions...")
        
        try:
            state["current_step"] = "generating_suggestions"
            
            if state["plagiarism_results"].get("suggestions"):
                state["suggestions"] = state["plagiarism_results"]["suggestions"]
            else:
                state["suggestions"] = ["No specific suggestions available."]
            
            print(f"✅ Generated {len(state['suggestions'])} suggestions")
            return state
            
        except Exception as e:
            state["status"] = DetectionStatus.FAILED
            state["error_message"] = f"Error generating suggestions: {str(e)}"
            return state
    
    def create_report(self, state: AgentState) -> AgentState:
        """Create final report"""
        print("📝 Creating final report...")
        
        try:
            state["current_step"] = "creating_report"
            state["detailed_report"] = self.detector.format_report(state["plagiarism_results"])
            state["status"] = DetectionStatus.COMPLETED
            
            print("✅ Report created successfully!")
            return state
            
        except Exception as e:
            state["status"] = DetectionStatus.FAILED
            state["error_message"] = f"Error creating report: {str(e)}"
            return state
    
    def handle_error(self, state: AgentState) -> AgentState:
        """Handle errors"""
        print(f"❌ Error: {state.get('error_message', 'Unknown error')}")
        state["status"] = DetectionStatus.FAILED
        return state
    
    # Conditional edge functions
    def should_extract_text(self, state: AgentState) -> str:
        """Decide if text extraction is needed"""
        if state["status"] == DetectionStatus.FAILED:
            return "error"
        elif state["file_type"] != "text":
            return "extract"
        else:
            return "preprocess"
    
    def should_analyze_results(self, state: AgentState) -> str:
        """Decide if results analysis should proceed"""
        return "error" if state["status"] == DetectionStatus.FAILED else "analyze"
    
    def should_generate_suggestions(self, state: AgentState) -> str:
        """Decide if suggestions should be generated"""
        if state["plagiarism_percentage"] > 20:  # Threshold for generating suggestions
            return "suggestions"
        else:
            return "report"
    
    # Helper methods for file extraction
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def run_analysis(self, input_text: str = None, input_file: str = None) -> Dict[str, Any]:
        """Run the complete plagiarism analysis"""
        
        # Initialize state
        initial_state = {
            "input_text": input_text,
            "input_file_path": input_file,
            "status": DetectionStatus.PENDING,
            "methods_used": [],
            "suggestions": [],
            "sources_found": []
        }
        
        print("🚀 Starting plagiarism detection workflow...")
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        print("🎉 Workflow completed!")
        
        return {
            "status": final_state["status"].value,
            "plagiarism_percentage": final_state.get("plagiarism_percentage", 0),
            "plagiarized_sections_count": len(final_state.get("plagiarized_sections", [])),
            "sources_found": final_state.get("sources_found", []),
            "suggestions": final_state.get("suggestions", []),
            "detailed_report": final_state.get("detailed_report", ""),
            "methods_used": final_state.get("methods_used", []),
            "error_message": final_state.get("error_message", ""),
            "full_results": final_state.get("plagiarism_results", {})
        }

    def run_batch_analysis(self, inputs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Run analysis on multiple inputs"""
        results = []
        
        for i, input_data in enumerate(inputs):
            print(f"\n--- Processing batch item {i+1}/{len(inputs)} ---")
            
            result = self.run_analysis(
                input_text=input_data.get("text"),
                input_file=input_data.get("file")
            )
            
            result["batch_id"] = i + 1
            results.append(result)
        
        return results

    def save_results(self, results: Dict[str, Any], output_path: str = "plagiarism_analysis.json"):
        """Save analysis results to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📄 Results saved to {output_path}")

# Usage example and CLI interface
import argparse
import time

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Plagiarism Detection Agent")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument("--openai-key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--google-key", type=str, help="Google API key (optional)")
    parser.add_argument("--google-cse", type=str, help="Google CSE ID (optional)")
    parser.add_argument("--output", type=str, default="results.json", help="Output file")
    parser.add_argument("--batch", type=str, help="JSON file with batch inputs")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = PlagiarismDetectionAgent(
        openai_api_key=args.openai_key,
        google_api_key=args.google_key,
        google_cse_id=args.google_cse
    )
    
    start_time = time.time()
    
    if args.batch:
        # Batch processing
        with open(args.batch, 'r') as f:
            batch_inputs = json.load(f)
        
        results = agent.run_batch_analysis(batch_inputs)
        agent.save_results({"batch_results": results}, args.output)
        
    else:
        # Single analysis
        if not args.text and not args.file:
            print("❌ Please provide either --text or --file argument")
            return
        
        results = agent.run_analysis(
            input_text=args.text,
            input_file=args.file
        )
        
        # Print summary
        print("\n" + "="*60)
        print("PLAGIARISM DETECTION SUMMARY")
        print("="*60)
        print(f"Status: {results['status']}")
        print(f"Plagiarism Percentage: {results['plagiarism_percentage']:.1f}%")
        print(f"Plagiarized Sections: {results['plagiarized_sections_count']}")
        print(f"Sources Found: {len(results['sources_found'])}")
        print(f"Methods Used: {', '.join(results['methods_used'])}")
        
        if results['error_message']:
            print(f"Error: {results['error_message']}")
        
        if results['suggestions']:
            print(f"\nSuggestions Available: {len(results['suggestions'])}")
        
        print(f"\nProcessing Time: {time.time() - start_time:.2f} seconds")
        
        # Print detailed report if available
        if results['detailed_report']:
            print("\n" + results['detailed_report'])
        
        # Save results
        agent.save_results(results, args.output)

# Jupyter notebook friendly interface
class PlagiarismDetectorNotebook:
    """Simplified interface for Jupyter notebooks"""
    
    def __init__(self, openai_api_key: str, google_api_key: str = None, google_cse_id: str = None):
        self.agent = PlagiarismDetectionAgent(
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
            google_cse_id=google_cse_id
        )
    
    def check_text(self, text: str, show_details: bool = True) -> Dict[str, Any]:
        """Check text for plagiarism with simple interface"""
        results = self.agent.run_analysis(input_text=text)
        
        if show_details:
            self._display_results(results)
        
        return results
    
    def check_file(self, file_path: str, show_details: bool = True) -> Dict[str, Any]:
        """Check file for plagiarism with simple interface"""
        results = self.agent.run_analysis(input_file=file_path)
        
        if show_details:
            self._display_results(results)
        
        return results
    
    def _display_results(self, results: Dict[str, Any]):
        """Display results in a formatted way"""
        print("🔍 PLAGIARISM DETECTION RESULTS")
        print("-" * 40)
        
        # Status indicator
        status = results['status']
        if status == 'completed':
            status_emoji = "✅"
        elif status == 'failed':
            status_emoji = "❌"
        else:
            status_emoji = "⏳"
        
        print(f"{status_emoji} Status: {status.upper()}")
        
        if results['error_message']:
            print(f"❌ Error: {results['error_message']}")
            return
        
        # Main metrics
        percentage = results['plagiarism_percentage']
        if percentage < 15:
            percentage_emoji = "🟢"
        elif percentage < 30:
            percentage_emoji = "🟡"
        else:
            percentage_emoji = "🔴"
        
        print(f"{percentage_emoji} Plagiarism Score: {percentage:.1f}%")
        print(f"📄 Plagiarized Sections: {results['plagiarized_sections_count']}")
        print(f"🌐 Sources Found: {len(results['sources_found'])}")
        print(f"🔧 Methods Used: {', '.join(results['methods_used'])}")
        
        # Recommendations
        if percentage > 20:
            print(f"\n💡 Suggestions Available: {len(results['suggestions'])}")
            print("   Run .get_suggestions() to see recommendations")
        
        if results['sources_found']:
            print(f"\n🔗 Top Sources:")
            for i, source in enumerate(results['sources_found'][:3], 1):
                print(f"   {i}. {source}")
    
    def get_suggestions(self, results: Dict[str, Any] = None):
        """Display suggestions for removing plagiarism"""
        if not results:
            print("Please provide results from a previous analysis")
            return
        
        suggestions = results.get('suggestions', [])
        if not suggestions:
            print("No suggestions available for this analysis")
            return
        
        print("💡 SUGGESTIONS TO REMOVE PLAGIARISM")
        print("=" * 50)
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion}")
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, float]:
        """Compare two texts directly"""
        detector = self.agent.detector
        
        return {
            "jaccard_similarity": detector.jaccard_similarity(text1, text2),
            "cosine_similarity": detector.cosine_similarity_tfidf(text1, text2),
            "semantic_similarity": detector.semantic_similarity(text1, text2)
        }

# Example usage functions
def demo_text_analysis():
    """Demo function for text analysis"""
    # Example API keys (replace with your own)
    OPENAI_KEY = "your-openai-api-key"
    GOOGLE_KEY = "your-google-api-key"  # Optional
    GOOGLE_CSE = "your-cse-id"  # Optional
    
    # Initialize the notebook interface
    detector = PlagiarismDetectorNotebook(
        openai_api_key=OPENAI_KEY,
        google_api_key=GOOGLE_KEY,
        google_cse_id=GOOGLE_CSE
    )
    
    # Sample text to analyze
    sample_text = """
    Climate change refers to long-term shifts in global or regional climate patterns. 
    It is primarily attributed to human activities, particularly the burning of fossil fuels, 
    which increases the levels of greenhouse gases in the atmosphere. The effects of climate 
    change include rising sea levels, more frequent extreme weather events, and shifts in 
    precipitation patterns. Addressing climate change requires immediate action from 
    governments, businesses, and individuals worldwide.
    """
    
    print("🔍 Analyzing sample text for plagiarism...")
    results = detector.check_text(sample_text)
    
    # Show suggestions if needed
    if results['plagiarism_percentage'] > 20:
        detector.get_suggestions(results)
    
    return results

def demo_file_analysis():
    """Demo function for file analysis"""
    # This would analyze a file
    detector = PlagiarismDetectorNotebook(
        openai_api_key="your-key-here"
    )
    
    # Example: detector.check_file("sample_document.pdf")
    print("File analysis demo - provide a file path to analyze")

if __name__ == "__main__":
    main()
        