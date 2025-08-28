#!/usr/bin/env python3
"""
LangGraph Visualization for Plagiarism Detection Agent
Creates both Mermaid diagrams and ASCII art for the workflow
"""

from typing import List, Dict, Any
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph_plagiarism_agent import AgentState, DetectionStatus
import os

# Load environment variables
load_dotenv()

# Node names as constants
PARSE_INPUT = "parse_input"
EXTRACT_TEXT = "extract_text"
PREPROCESS_TEXT = "preprocess_text"
DETECT_PLAGIARISM = "detect_plagiarism"
ANALYZE_RESULTS = "analyze_results"
GENERATE_SUGGESTIONS = "generate_suggestions"
CREATE_REPORT = "create_report"
HANDLE_ERROR = "handle_error"

def parse_input_node(state: AgentState) -> AgentState:
    """Parse and validate input"""
    print("📄 Parsing input...")
    
    try:
        state["status"] = DetectionStatus.PROCESSING
        state["current_step"] = "parsing_input"
        state["methods_used"] = []
        
        # Determine input type
        if state.get("input_file_path"):
            from pathlib import Path
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

def extract_text_node(state: AgentState) -> AgentState:
    """Extract text from various file formats"""
    print("📖 Extracting text from file...")
    
    try:
        state["current_step"] = "extracting_text"
        file_path = state["input_file_path"]
        file_type = state["file_type"]
        
        # Simulate text extraction
        if file_type in [".pdf", ".docx", ".txt"]:
            state["extracted_text"] = f"Extracted text content from {file_path}"
        else:
            state["status"] = DetectionStatus.FAILED
            state["error_message"] = f"Unsupported file type: {file_type}"
            return state
        
        print(f"✅ Text extracted successfully. Length: {len(state['extracted_text'])} characters")
        return state
        
    except Exception as e:
        state["status"] = DetectionStatus.FAILED
        state["error_message"] = f"Error extracting text: {str(e)}"
        return state

def preprocess_text_node(state: AgentState) -> AgentState:
    """Preprocess and clean text"""
    print("🔧 Preprocessing text...")
    
    try:
        state["current_step"] = "preprocessing_text"
        
        # Use extracted text or input text
        text = state.get("extracted_text") or state.get("input_text", "")
        
        # Simulate text preprocessing
        state["extracted_text"] = text.strip()
        state["text_chunks"] = [text[i:i+100] for i in range(0, len(text), 100)]  # Simple chunking
        
        print(f"✅ Text preprocessed. Chunks created: {len(state['text_chunks'])}")
        return state
        
    except Exception as e:
        state["status"] = DetectionStatus.FAILED
        state["error_message"] = f"Error preprocessing text: {str(e)}"
        return state

def detect_plagiarism_node(state: AgentState) -> AgentState:
    """Run plagiarism detection"""
    print("🔍 Running plagiarism detection...")
    
    try:
        state["current_step"] = "detecting_plagiarism"
        
        # Simulate plagiarism detection
        import random
        state["plagiarism_percentage"] = random.uniform(0, 50)
        state["plagiarized_sections"] = []
        
        # Track methods used
        methods = ["jaccard_similarity", "cosine_similarity_tfidf", "semantic_similarity"]
        state["methods_used"] = methods
        
        print(f"✅ Plagiarism detection completed. Score: {state['plagiarism_percentage']:.1f}%")
        return state
        
    except Exception as e:
        state["status"] = DetectionStatus.FAILED
        state["error_message"] = f"Error in plagiarism detection: {str(e)}"
        return state

def analyze_results_node(state: AgentState) -> AgentState:
    """Analyze detection results"""
    print("📊 Analyzing results...")
    
    try:
        state["current_step"] = "analyzing_results"
        
        # Simulate results analysis
        state["sources_found"] = ["example.com", "wikipedia.org"]
        state["similarity_scores"] = [0.3, 0.7, 0.2]
        
        print(f"✅ Analysis completed. Sources found: {len(state['sources_found'])}")
        return state
        
    except Exception as e:
        state["status"] = DetectionStatus.FAILED
        state["error_message"] = f"Error analyzing results: {str(e)}"
        return state

def generate_suggestions_node(state: AgentState) -> AgentState:
    """Generate suggestions to remove plagiarism"""
    print("💡 Generating suggestions...")
    
    try:
        state["current_step"] = "generating_suggestions"
        
        # Simulate suggestion generation
        state["suggestions"] = [
            "Rewrite the flagged sections using different vocabulary",
            "Add proper citations for referenced content",
            "Paraphrase the content while maintaining the original meaning"
        ]
        
        print(f"✅ Generated {len(state['suggestions'])} suggestions")
        return state
        
    except Exception as e:
        state["status"] = DetectionStatus.FAILED
        state["error_message"] = f"Error generating suggestions: {str(e)}"
        return state

def create_report_node(state: AgentState) -> AgentState:
    """Create final report"""
    print("📝 Creating final report...")
    
    try:
        state["current_step"] = "creating_report"
        
        # Create a simple report
        state["detailed_report"] = f"""
PLAGIARISM DETECTION REPORT
==========================
Plagiarism Score: {state.get('plagiarism_percentage', 0):.1f}%
Sources Found: {len(state.get('sources_found', []))}
Methods Used: {', '.join(state.get('methods_used', []))}
Suggestions: {len(state.get('suggestions', []))}
"""
        
        state["status"] = DetectionStatus.COMPLETED
        
        print("✅ Report created successfully!")
        return state
        
    except Exception as e:
        state["status"] = DetectionStatus.FAILED
        state["error_message"] = f"Error creating report: {str(e)}"
        return state

def handle_error_node(state: AgentState) -> AgentState:
    """Handle errors"""
    print(f"❌ Error: {state.get('error_message', 'Unknown error')}")
    state["status"] = DetectionStatus.FAILED
    return state

# Conditional edge functions
def should_extract_text(state: AgentState) -> str:
    """Decide if text extraction is needed"""
    if state["status"] == DetectionStatus.FAILED:
        return HANDLE_ERROR
    elif state.get("file_type") != "text":
        return EXTRACT_TEXT
    else:
        return PREPROCESS_TEXT

def should_analyze_results(state: AgentState) -> str:
    """Decide if results analysis should proceed"""
    return HANDLE_ERROR if state["status"] == DetectionStatus.FAILED else ANALYZE_RESULTS

def should_generate_suggestions(state: AgentState) -> str:
    """Decide if suggestions should be generated"""
    plagiarism_percentage = state.get("plagiarism_percentage", 0)
    if plagiarism_percentage > 20:  # Threshold for generating suggestions
        return GENERATE_SUGGESTIONS
    else:
        return CREATE_REPORT

def create_plagiarism_detection_graph():
    """Create the plagiarism detection workflow graph"""
    
    # Create StateGraph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node(PARSE_INPUT, parse_input_node)
    graph.add_node(EXTRACT_TEXT, extract_text_node)
    graph.add_node(PREPROCESS_TEXT, preprocess_text_node)
    graph.add_node(DETECT_PLAGIARISM, detect_plagiarism_node)
    graph.add_node(ANALYZE_RESULTS, analyze_results_node)
    graph.add_node(GENERATE_SUGGESTIONS, generate_suggestions_node)
    graph.add_node(CREATE_REPORT, create_report_node)
    graph.add_node(HANDLE_ERROR, handle_error_node)
    
    # Set entry point
    graph.set_entry_point(PARSE_INPUT)
    
    # Add conditional edges
    graph.add_conditional_edges(
        PARSE_INPUT,
        should_extract_text,
        {
            EXTRACT_TEXT: EXTRACT_TEXT,
            PREPROCESS_TEXT: PREPROCESS_TEXT,
            HANDLE_ERROR: HANDLE_ERROR
        }
    )
    
    graph.add_conditional_edges(
        DETECT_PLAGIARISM,
        should_analyze_results,
        {
            ANALYZE_RESULTS: ANALYZE_RESULTS,
            HANDLE_ERROR: HANDLE_ERROR
        }
    )
    
    graph.add_conditional_edges(
        ANALYZE_RESULTS,
        should_generate_suggestions,
        {
            GENERATE_SUGGESTIONS: GENERATE_SUGGESTIONS,
            CREATE_REPORT: CREATE_REPORT
        }
    )
    
    # Add direct edges
    graph.add_edge(EXTRACT_TEXT, PREPROCESS_TEXT)
    graph.add_edge(PREPROCESS_TEXT, DETECT_PLAGIARISM)
    graph.add_edge(GENERATE_SUGGESTIONS, CREATE_REPORT)
    graph.add_edge(CREATE_REPORT, END)
    graph.add_edge(HANDLE_ERROR, END)
    
    return graph

def main():
    """Main function to create and visualize the graph"""
    
    print("🎨 PLAGIARISM DETECTION WORKFLOW VISUALIZER")
    print("=" * 60)
    
    # Create the graph
    print("🔧 Creating workflow graph...")
    graph = create_plagiarism_detection_graph()
    
    # Compile the graph
    print("⚙️ Compiling graph...")
    app = graph.compile()
    
    print("\n" + "=" * 60)
    print("📊 MERMAID DIAGRAM")
    print("=" * 60)
    print("Copy this to https://mermaid.live/ for interactive visualization:")
    print()
    
    # Generate Mermaid diagram
    try:
        mermaid_diagram = app.get_graph().draw_mermaid()
        print(mermaid_diagram)
    except Exception as e:
        print(f"❌ Error generating Mermaid diagram: {e}")
    
    print("\n" + "=" * 60)
    print("🎨 ASCII DIAGRAM")
    print("=" * 60)
    
    # Generate ASCII diagram
    try:
        ascii_diagram = app.get_graph().print_ascii()
        if ascii_diagram:
            print(ascii_diagram)
        else:
            print("ASCII diagram generated (check above output)")
    except Exception as e:
        print(f"❌ Error generating ASCII diagram: {e}")
    
    print("\n" + "=" * 60)
    print("📋 WORKFLOW SUMMARY")
    print("=" * 60)
    
    # Print workflow summary
    nodes = [
        PARSE_INPUT, EXTRACT_TEXT, PREPROCESS_TEXT, DETECT_PLAGIARISM,
        ANALYZE_RESULTS, GENERATE_SUGGESTIONS, CREATE_REPORT, HANDLE_ERROR
    ]
    
    print(f"📊 Total Nodes: {len(nodes)}")
    print(f"🚀 Entry Point: {PARSE_INPUT}")
    print(f"🏁 End Points: {CREATE_REPORT}, {HANDLE_ERROR}")
    print(f"🔀 Conditional Nodes: 3")
    print(f"⚡ Processing Nodes: 8")
    
    print(f"\n🏗️ Node Details:")
    node_descriptions = {
        PARSE_INPUT: "📄 Validate and parse input",
        EXTRACT_TEXT: "📖 Extract text from files",
        PREPROCESS_TEXT: "🔧 Clean and chunk text",
        DETECT_PLAGIARISM: "🔍 Run similarity algorithms",
        ANALYZE_RESULTS: "📊 Process detection results",
        GENERATE_SUGGESTIONS: "💡 Generate AI suggestions",
        CREATE_REPORT: "📝 Format final report",
        HANDLE_ERROR: "❌ Handle workflow errors"
    }
    
    for i, (node, description) in enumerate(node_descriptions.items(), 1):
        print(f"  {i}. {description}")
    
    print(f"\n🔗 Decision Logic:")
    print(f"  • File extraction: Based on input type (PDF/DOCX/TXT vs text)")
    print(f"  • Error handling: Any step can trigger error flow")
    print(f"  • Suggestion generation: Only if plagiarism > 20%")
    
    # Test the workflow with sample data
    print(f"\n🧪 TESTING WORKFLOW")
    print("=" * 30)
    
    try:
        # Create sample state
        initial_state = {
            "input_text": "This is a sample text for plagiarism detection testing.",
            "status": DetectionStatus.PENDING,
            "methods_used": [],
            "suggestions": [],
            "sources_found": []
        }
        
        print("🔍 Running sample analysis...")
        result = app.invoke(initial_state)
        
        print("✅ Workflow completed successfully!")
        print(f"📊 Final Status: {result.get('status', 'Unknown')}")
        print(f"📈 Plagiarism Score: {result.get('plagiarism_percentage', 0):.1f}%")
        print(f"💡 Suggestions Generated: {len(result.get('suggestions', []))}")
        print(f"🔧 Methods Used: {', '.join(result.get('methods_used', []))}")
        
        if result.get('detailed_report'):
            print(f"\n📄 Sample Report Generated:")
            print(result['detailed_report'])
        
    except Exception as e:
        print(f"❌ Error testing workflow: {e}")
        print("(This is expected if dependencies are missing)")
    
    print(f"\n🎉 Visualization Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
