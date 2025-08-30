#!/usr/bin/env python3
"""
Streamlit Web Application for Plagiarism Detection
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import time
from io import StringIO
import tempfile
from pathlib import Path

# Import your plagiarism detector
try:
    from plagiarism_detector import PlagiarismDetector
except ImportError:
    st.error("Please ensure plagiarism_detector.py is in the same directory")
    st.stop()

# Check for LangGraph availability
try:
    import langgraph
    from langgraph_plagiarism_agent import PlagiarismDetectionAgent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    PlagiarismDetectionAgent = None

# Page configuration
st.set_page_config(
    page_title="AI Plagiarism Detective",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        margin-bottom: 2rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .plagiarism-low { border-left-color: #4CAF50 !important; }
    .plagiarism-medium { border-left-color: #FF9800 !important; }
    .plagiarism-high { border-left-color: #F44336 !important; }
    
    .source-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    
    .suggestion-card {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    
    .chunk-analysis {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'detector_type' not in st.session_state:
        st.session_state.detector_type = "Standard"

def create_detector(api_keys, detector_type="Standard"):
    """Create plagiarism detection agent instance"""
    try:
        if LANGGRAPH_AVAILABLE:
            # Use LangGraph agent for structured workflow
            agent = PlagiarismDetectionAgent(
                openai_api_key=api_keys['openai'],
                google_api_key=api_keys.get('google'),
                google_cse_id=api_keys.get('google_cse')
            )
            return agent
        else:
            # Fallback to basic detector
            return PlagiarismDetector(
                openai_api_key=api_keys['openai'],
                google_api_key=api_keys.get('google'),
                google_cse_id=api_keys.get('google_cse')
            )
    except Exception as e:
        st.error(f"Error initializing detector: {str(e)}")
        return None

def get_plagiarism_color_and_icon(percentage):
    """Get color and icon based on plagiarism percentage"""
    if percentage < 15:
        return "#4CAF50", "✅", "Low"
    elif percentage < 30:
        return "#FF9800", "⚠️", "Medium"
    else:
        return "#F44336", "🔴", "High"

def create_plagiarism_gauge(percentage):
    """Create a gauge chart for plagiarism percentage"""
    color, icon, level = get_plagiarism_color_and_icon(percentage)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{icon} Plagiarism Level: {level}"},
        delta = {'reference': 20},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 15], 'color': "lightgray"},
                {'range': [15, 30], 'color': "yellow"},
                {'range': [30, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_similarity_comparison_chart(chunk_results):
    """Create comparison chart for different similarity methods"""
    if not chunk_results:
        return None
    
    methods = ['jaccard', 'cosine_tfidf', 'semantic']
    method_names = ['Jaccard Similarity', 'TF-IDF Cosine', 'Semantic Similarity']
    
    data = []
    for i, chunk in enumerate(chunk_results):
        if chunk.get('similarities'):
            avg_scores = {}
            for method in methods:
                scores = [sim.get(method, 0) for sim in chunk['similarities']]
                avg_scores[method] = sum(scores) / len(scores) if scores else 0
            
            for method, method_name in zip(methods, method_names):
                data.append({
                    'Chunk': f"Chunk {i+1}",
                    'Method': method_name,
                    'Score': avg_scores.get(method, 0)
                })
    
    if data:
        df = pd.DataFrame(data)
        fig = px.bar(df, x='Chunk', y='Score', color='Method',
                    title="Similarity Scores by Detection Method",
                    labels={'Score': 'Similarity Score'},
                    height=400)
        return fig
    return None

def create_sources_distribution_chart(results):
    """Create chart showing distribution of sources"""
    sources = []
    for chunk in results.get('detected_plagiarism', []):
        for similarity in chunk.get('similarities', []):
            source = similarity.get('source', 'Unknown')
            # Extract domain from URL
            try:
                from urllib.parse import urlparse
                domain = urlparse(source).netloc
                sources.append(domain)
            except:
                sources.append('Unknown')
    
    if sources:
        source_counts = pd.Series(sources).value_counts().head(10)
        fig = px.pie(values=source_counts.values, names=source_counts.index,
                    title="Top 10 Source Domains")
        return fig
    return None

def display_detailed_analysis(results):
    """Display detailed analysis results"""
    st.header("📊 Detailed Analysis Results")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    summary = results.get('summary', {})
    percentage = summary.get('plagiarism_percentage', 0)
    color, icon, level = get_plagiarism_color_and_icon(percentage)
    
    with col1:
        st.metric(
            label="Plagiarism Score",
            value=f"{percentage:.1f}%",
            delta=f"{level} Risk"
        )
    
    with col2:
        st.metric(
            label="Chunks Analyzed",
            value=summary.get('total_chunks_analyzed', 0)
        )
    
    with col3:
        st.metric(
            label="Flagged Chunks",
            value=summary.get('plagiarized_chunks', 0)
        )
    
    with col4:
        st.metric(
            label="Sources Found",
            value=summary.get('web_sources_checked', 0)
        )
    
    # Plagiarism gauge
    st.subheader("🎯 Plagiarism Risk Assessment")
    gauge_fig = create_plagiarism_gauge(percentage)
    st.plotly_chart(gauge_fig, use_container_width=True)
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        similarity_chart = create_similarity_comparison_chart(results.get('detected_plagiarism', []))
        if similarity_chart:
            st.plotly_chart(similarity_chart, use_container_width=True)
    
    with col2:
        sources_chart = create_sources_distribution_chart(results)
        if sources_chart:
            st.plotly_chart(sources_chart, use_container_width=True)

def display_chunk_analysis(results):
    """Display detailed chunk-by-chunk analysis"""
    st.header("🔍 Chunk-by-Chunk Analysis")
    
    plagiarized_chunks = [chunk for chunk in results.get('detected_plagiarism', []) 
                         if chunk.get('is_plagiarized', False)]
    
    if not plagiarized_chunks:
        st.success("🎉 No significant plagiarism detected in any text chunks!")
        return
    
    for i, chunk in enumerate(plagiarized_chunks):
        st.markdown(f"### 🚨 Flagged Chunk #{i+1}")
        
        # Create expandable section for each chunk
        with st.expander(f"View details for chunk {i+1} (Similarity: {chunk['max_similarity']:.1%})"):
            # Display the flagged text
            st.markdown("**📝 Flagged Text:**")
            st.markdown(f"*{chunk['text']}*")
            
            # Display similarity scores
            if chunk.get('similarities'):
                st.markdown("**📊 Similarity Analysis:**")
                
                similarity_data = []
                for sim in chunk['similarities']:
                    similarity_data.append({
                        'Source': sim.get('title', 'Unknown')[:50] + '...',
                        'URL': sim.get('source', ''),
                        'Jaccard': f"{sim.get('jaccard', 0):.3f}",
                        'TF-IDF': f"{sim.get('cosine_tfidf', 0):.3f}",
                        'Semantic': f"{sim.get('semantic', 0):.3f}",
                        'Combined': f"{sim.get('combined', 0):.3f}"
                    })
                
                df = pd.DataFrame(similarity_data)
                st.dataframe(df, use_container_width=True)
                
                # Show matched content
                if chunk['similarities']:
                    best_match = max(chunk['similarities'], key=lambda x: x.get('combined', 0))
                    st.markdown("**🎯 Best Match Content:**")
                    st.info(f"Source: {best_match.get('source', 'Unknown')}")
                    st.markdown(f"*{best_match.get('matched_content', 'No content available')}*")

def display_analysis_methods(results):
    """Display results from different analysis methods"""
    st.header("🛠️ Analysis Methods Results")
    
    methods = results.get('analysis_methods', {})
    
    if not methods:
        st.info("No secondary analysis methods were run.")
        return
    
    for method_name, method_results in methods.items():
        if method_results.get('available', False):
            st.markdown(f"### {method_name.title().replace('_', ' ')}")
            
            if method_name == 'phrase_matching':
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Phrases Checked", method_results.get('phrases_checked', 0))
                with col2:
                    st.metric("Matches Found", method_results.get('phrases_found', 0))
                with col3:
                    st.metric("Match Rate", f"{method_results.get('match_percentage', 0):.1f}%")
                
                # Show matched phrases
                sources = method_results.get('sources', [])
                if sources:
                    st.markdown("**🔗 Matched Phrases:**")
                    for source in sources[:5]:  # Show top 5
                        st.markdown(f"- **{source.get('matched_phrase', 'N/A')}** found in: {source.get('title', 'Unknown')}")
            
            elif method_name == 'academic_check':
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Academic Phrases Checked", method_results.get('checked_phrases', 0))
                with col2:
                    st.metric("ArXiv Matches", method_results.get('matches_found', 0))
                
                matches = method_results.get('matches', [])
                if matches:
                    st.markdown("**📚 Academic Matches:**")
                    for match in matches:
                        st.markdown(f"- Phrase: *{match.get('phrase', 'N/A')}*")
            
            elif method_name == 'content_fingerprint':
                st.metric("Content Fingerprint", method_results.get('fingerprint', 'N/A'))
                st.info(method_results.get('note', 'Content fingerprinting completed'))
        else:
            st.markdown(f"### {method_name.title().replace('_', ' ')}")
            st.warning(f"Method not available: {method_results.get('message', 'Unknown error')}")

def display_suggestions(results):
    """Display AI-generated suggestions"""
    st.header("💡 AI-Powered Suggestions")
    
    suggestions = results.get('suggestions', [])
    plagiarism_percentage = results.get('overall_plagiarism_percentage', 0)
    flagged_chunks = len([chunk for chunk in results.get('detected_plagiarism', []) if chunk.get('is_plagiarized')])
    
    # Show status information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Plagiarism Score", f"{plagiarism_percentage:.1f}%")
    with col2:
        st.metric("Flagged Chunks", flagged_chunks)
    with col3:
        threshold = 20  # Current threshold for suggestions
        st.metric("Suggestion Threshold", f"{threshold}%")
    
    if not suggestions or (len(suggestions) == 1 and "Unable to generate" in suggestions[0]):
        if plagiarism_percentage < 20:
            st.success("🎉 **Great news!** Your text appears to be original!")
            st.info("💡 **Why no suggestions?** Your plagiarism score is below 20%, indicating original content. Suggestions are only generated for higher plagiarism scores.")
            
            # Option to force suggestions for improvement
            if st.button("🔧 Generate Writing Improvement Suggestions Anyway"):
                with st.spinner("Generating writing improvement suggestions..."):
                    try:
                        # Force generate suggestions for writing improvement
                        detector = st.session_state.get('detector')
                        if detector:
                            # Create fake plagiarized parts from the original text for improvement suggestions
                            chunks = [chunk['text'] for chunk in results.get('detected_plagiarism', [])][:3]
                            if chunks:
                                improvement_suggestions = detector.generate_suggestions(
                                    "Original text", chunks
                                )
                                if improvement_suggestions:
                                    st.markdown("### ✍️ Writing Improvement Suggestions")
                                    for i, suggestion in enumerate(improvement_suggestions):
                                        with st.expander(f"Improvement suggestion {i+1}"):
                                            st.markdown(suggestion)
                    except Exception as e:
                        st.error(f"Error generating improvement suggestions: {e}")
        else:
            st.warning("⚠️ **Suggestions unavailable** despite flagged content.")
            st.info("This might be due to:")
            st.markdown("""
            - 🔑 **OpenAI API issue** - Check your API key and billing
            - 🌐 **Network connectivity** - Verify internet connection  
            - 🚫 **Rate limiting** - You may have exceeded API limits
            - 🔧 **Technical error** - Try running the analysis again
            """)
            
            if st.button("🔄 Retry Generating Suggestions"):
                st.experimental_rerun()
        
        return
    
    # Display actual suggestions
    st.markdown("**🤖 Here are AI-generated suggestions to improve your text and reduce plagiarism:**")
    
    suggestion_count = len(suggestions)
    st.info(f"📝 Generated {suggestion_count} suggestion{'s' if suggestion_count != 1 else ''} for flagged content")
    
    for i, suggestion in enumerate(suggestions):
        st.markdown(f"### 💡 Suggestion {i+1}")
        
        with st.expander(f"View detailed suggestion {i+1}", expanded=True):
            # Parse the suggestion to separate the original text and suggestions
            lines = suggestion.split('\n')
            original_text = ""
            rewrite_suggestions = []
            
            for line in lines:
                line = line.strip()
                if line:
                    if line.startswith('For text:'):
                        original_text = line[10:].strip().strip('"\'')
                        st.markdown(f"**📝 Original Text:**")
                        st.markdown(f"> {original_text}")
                        st.markdown("**✍️ Rewrite Options:**")
                    elif line.startswith(('1.', '2.', '3.')):
                        st.markdown(f"**{line}**")
                    elif line and not line.startswith('For text:'):
                        st.markdown(line)
    
    # Additional helpful information
    st.markdown("---")
    st.markdown("### 📚 How to Use These Suggestions")
    st.info("""
    **Tips for implementing suggestions:**
    - 📖 **Read each option** and choose the one that best fits your writing style
    - 🎯 **Maintain meaning** while changing structure and vocabulary  
    - ✅ **Check citations** if the original content should be quoted instead
    - 🔄 **Run analysis again** after making changes to verify improvement
    """)
    
    # Export suggestions
    if st.button("📥 Export Suggestions as Text"):
        suggestions_text = "\n\n".join([f"Suggestion {i+1}:\n{sugg}" for i, sugg in enumerate(suggestions)])
        st.download_button(
            label="Download Suggestions",
            data=suggestions_text,
            file_name=f"plagiarism_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def display_export_options(results):
    """Display export options for results"""
    st.header("📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON export
        if st.button("📄 Export as JSON"):
            json_str = json.dumps(results, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON Report",
                data=json_str,
                file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        # PDF export (simplified)
        if st.button("📑 Generate Text Report"):
            try:
                if hasattr(st.session_state, 'detector') and st.session_state.detector:
                    if hasattr(st.session_state.detector, 'detector'):
                        # LangGraph agent - access the underlying detector
                        report = st.session_state.detector.detector.format_report(results)
                    else:
                        # Basic detector
                        report = st.session_state.detector.format_report(results)
                    
                    st.download_button(
                        label="Download Text Report",
                        data=report,
                        file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Detector not available for report generation")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    with col3:
        # CSV export of chunk analysis
        if st.button("📊 Export Chunk Analysis as CSV"):
            chunk_data = []
            for i, chunk in enumerate(results.get('detected_plagiarism', [])):
                chunk_data.append({
                    'Chunk_Number': i+1,
                    'Text': chunk.get('text', ''),
                    'Is_Plagiarized': chunk.get('is_plagiarized', False),
                    'Max_Similarity': chunk.get('max_similarity', 0),
                    'Source_Count': len(chunk.get('similarities', []))
                })
            
            if chunk_data:
                df = pd.DataFrame(chunk_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Analysis",
                    data=csv,
                    file_name=f"chunk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 AI Plagiarism Detective</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Advanced AI and Multiple Detection Algorithms")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys section
        st.subheader("🔑 API Keys")
        
        openai_key = st.text_input("OpenAI API Key", type="password", 
                                  help="Required for AI-powered suggestions")
        
        with st.expander("🌐 Google Search (Optional)"):
            google_key = st.text_input("Google API Key", type="password",
                                      help="For web search functionality")
            google_cse = st.text_input("Google CSE ID",
                                      help="Custom Search Engine ID")
        
        # Detector type selection
        st.subheader("🛠️ Detection Mode")
        detector_type = st.selectbox(
            "Choose Detector Type",
            ["Standard", "Enhanced"],
            help="Enhanced mode includes additional academic and content fingerprint checks"
        )
        st.session_state.detector_type = detector_type
        
        # Analysis settings
        st.subheader("📊 Analysis Settings")
        enable_web_search = st.checkbox("Enable Web Search", value=True,
                                       help="Search the web for similar content")
        
        chunk_size = st.slider("Text Chunk Size", min_value=2, max_value=6, value=3,
                              help="Number of sentences per analysis chunk")
        
        # Analysis history
        if st.session_state.analysis_history:
            st.subheader("📈 Analysis History")
            for i, analysis in enumerate(st.session_state.analysis_history[-5:]):  # Show last 5
                timestamp = analysis.get('timestamp', 'Unknown')
                percentage = analysis.get('percentage', 0)
                method = analysis.get('method', 'Unknown')
                st.write(f"{i+1}. {timestamp}: {percentage:.1f}% ({method})")
    
    # Show current detection method
    detection_method = "🤖 LangGraph Agent Workflow" if LANGGRAPH_AVAILABLE else "⚡ Basic Detection"
    st.sidebar.success(f"Using: {detection_method}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "📁 File Analysis", "📊 Batch Analysis"])
    
    with tab1:
        st.header("📝 Text Analysis")
        
        # Text input
        input_text = st.text_area(
            "Enter text to analyze for plagiarism:",
            height=200,
            placeholder="Paste your text here... (minimum 50 characters recommended)"
        )
        
        # Analysis button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze for Plagiarism", type="primary", use_container_width=True)
        
        # Perform analysis
        if analyze_button:
            if not openai_key:
                st.error("❌ Please provide an OpenAI API key in the sidebar.")
            elif len(input_text.strip()) < 20:
                st.warning("⚠️ Please provide more text (at least 20 characters) for meaningful analysis.")
            else:
                # Create detector
                api_keys = {
                    'openai': openai_key,
                    'google': google_key if google_key else None,
                    'google_cse': google_cse if google_cse else None
                }
                
                detector = create_detector(api_keys, detector_type)
                if detector:
                    st.session_state.detector = detector
                    
                    # Show progress
                    with st.spinner("🔍 Analyzing text for plagiarism..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Initializing analysis...")
                        progress_bar.progress(20)
                        time.sleep(0.5)
                        
                        status_text.text("Processing text chunks...")
                        progress_bar.progress(40)
                        
                        # Run analysis using LangGraph agent
                        if hasattr(detector, 'run_analysis'):
                            # Using LangGraph agent
                            results = detector.run_analysis(input_text=input_text)
                            
                            # Convert agent results to expected format
                            if results.get('full_results'):
                                analysis_results = results['full_results']
                                analysis_results['overall_plagiarism_percentage'] = results.get('plagiarism_percentage', 0)
                                analysis_results['suggestions'] = results.get('suggestions', [])
                            else:
                                analysis_results = {
                                    'overall_plagiarism_percentage': results.get('plagiarism_percentage', 0),
                                    'detected_plagiarism': [],
                                    'suggestions': results.get('suggestions', []),
                                    'summary': {
                                        'plagiarism_percentage': results.get('plagiarism_percentage', 0),
                                        'total_chunks_analyzed': 0,
                                        'plagiarized_chunks': results.get('plagiarized_sections_count', 0),
                                        'web_sources_checked': len(results.get('sources_found', []))
                                    }
                                }
                        else:
                            # Using basic detector
                            analysis_results = detector.analyze_text(input_text, web_search_enabled=enable_web_search)
                        progress_bar.progress(80)
                        
                        status_text.text("Generating suggestions...")
                        progress_bar.progress(100)
                        
                        # Store results
                        st.session_state.analysis_results = analysis_results
                        
                        # Add to history
                        st.session_state.analysis_history.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'percentage': analysis_results.get('overall_plagiarism_percentage', 0),
                            'text_preview': input_text[:50] + '...',
                            'method': 'LangGraph Agent' if hasattr(detector, 'run_analysis') else 'Basic Detector'
                        })
                        
                        progress_bar.empty()
                        status_text.empty()
                    
                    st.success("✅ Analysis completed!")
        
        # Display results if available
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Display all analysis sections
            display_detailed_analysis(results)
            st.markdown("---")
            
            display_chunk_analysis(results)
            st.markdown("---")
            
            display_analysis_methods(results)
            st.markdown("---")
            
            display_suggestions(results)
            st.markdown("---")
            
            display_export_options(results)
    
    with tab2:
        st.header("📁 File Analysis")
        st.info("🚧 File upload functionality - Upload PDF, DOCX, or TXT files")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size} bytes")
            
            if st.button("🔍 Analyze File", type="primary"):
                if not openai_key:
                    st.error("❌ Please provide an OpenAI API key in the sidebar.")
                else:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Create detector
                        api_keys = {
                            'openai': openai_key,
                            'google': google_key if google_key else None,
                            'google_cse': google_cse if google_cse else None
                        }
                        
                        detector = create_detector(api_keys, detector_type)
                        if detector:
                            with st.spinner("📄 Processing file..."):
                                # Use LangGraph agent for file processing
                                if hasattr(detector, 'run_analysis'):
                                    # LangGraph agent can handle file input directly
                                    results = detector.run_analysis(input_file=tmp_file_path)
                                    
                                    # Convert agent results to expected format
                                    if results.get('full_results'):
                                        analysis_results = results['full_results']
                                        analysis_results['overall_plagiarism_percentage'] = results.get('plagiarism_percentage', 0)
                                    else:
                                        analysis_results = {
                                            'overall_plagiarism_percentage': results.get('plagiarism_percentage', 0),
                                            'detected_plagiarism': [],
                                            'suggestions': results.get('suggestions', []),
                                            'summary': {
                                                'plagiarism_percentage': results.get('plagiarism_percentage', 0),
                                                'total_chunks_analyzed': 0,
                                                'plagiarized_chunks': results.get('plagiarized_sections_count', 0),
                                                'web_sources_checked': len(results.get('sources_found', []))
                                            }
                                        }
                                    
                                    st.session_state.analysis_results = analysis_results
                                    st.success("✅ File analysis completed using LangGraph workflow!")
                                    st.rerun()
                                    
                                else:
                                    # Fallback to basic file processing
                                    if uploaded_file.type == "text/plain":
                                        file_text = str(uploaded_file.read(), "utf-8")
                                        analysis_results = detector.analyze_text(file_text, web_search_enabled=enable_web_search)
                                        st.session_state.analysis_results = analysis_results
                                        st.success("✅ File analysis completed!")
                                        st.rerun()
                                    else:
                                        st.error("File text extraction not implemented for this file type in basic mode. Please use text analysis or upgrade to LangGraph agent.")

                    
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
    
    with tab3:
        st.header("📊 Batch Analysis")
        st.info("🚧 Batch processing functionality - Analyze multiple texts at once")
        
        # Batch input methods
        batch_method = st.radio(
            "Choose batch input method:",
            ["Text List", "CSV Upload", "JSON Upload"]
        )
        
        if batch_method == "Text List":
            st.subheader("Enter multiple texts (one per line)")
            batch_texts = st.text_area(
                "Texts to analyze:",
                height=200,
                placeholder="Enter each text on a new line...\nText 1 here\nText 2 here\nText 3 here"
            )
            
            if st.button("🔍 Analyze Batch") and batch_texts:
                texts = [text.strip() for text in batch_texts.split('\n') if text.strip()]
                if not openai_key:
                    st.error("❌ Please provide an OpenAI API key.")
                elif len(texts) == 0:
                    st.warning("⚠️ No valid texts found.")
                else:
                    st.info(f"📊 Processing {len(texts)} texts...")
                    
                    # Process each text
                    batch_results = []
                    progress_bar = st.progress(0)
                    
                    for i, text in enumerate(texts):
                        if len(text) > 20:  # Only process substantial texts
                            api_keys = {
                                'openai': openai_key,
                                'google': google_key if google_key else None,
                                'google_cse': google_cse if google_cse else None
                            }
                            
                            detector = create_detector(api_keys, detector_type)
                            if detector and hasattr(detector, 'run_analysis'):
                                # Use LangGraph agent for batch processing
                                batch_inputs = [{"text": text} for text in texts if len(text) > 20]
                                batch_results = detector.run_batch_analysis(batch_inputs)
                                
                                # Convert to display format
                                display_results = []
                                for result in batch_results:
                                    display_results.append({
                                        'text_preview': result.get('text_preview', 'N/A'),
                                        'plagiarism_percentage': result.get('plagiarism_percentage', 0),
                                        'flagged_chunks': result.get('plagiarized_sections_count', 0),
                                        'method': 'LangGraph Agent'
                                    })
                                batch_results = display_results
                                
                            elif detector:
                                # Fallback to basic processing
                                for i, text in enumerate(texts):
                                    if len(text) > 20:
                                        result = detector.analyze_text(text, web_search_enabled=enable_web_search)
                                        batch_results.append({
                                            'text_preview': text[:50] + '...',
                                            'plagiarism_percentage': result.get('overall_plagiarism_percentage', 0),
                                            'flagged_chunks': len([c for c in result.get('detected_plagiarism', []) if c.get('is_plagiarized')]),
                                            'method': 'Basic Detector'
                                        })
                        
                        progress_bar.progress((i + 1) / len(texts))
                    
                    # Display batch results
                    if batch_results:
                        st.success(f"✅ Batch analysis completed! Processed {len(batch_results)} texts.")
                        
                        df = pd.DataFrame(batch_results)
                        st.dataframe(df, use_container_width=True)
                        
                        # Batch statistics
                        avg_plagiarism = df['plagiarism_percentage'].mean()
                        max_plagiarism = df['plagiarism_percentage'].max()
                        flagged_count = len(df[df['plagiarism_percentage'] > 20])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Plagiarism", f"{avg_plagiarism:.1f}%")
                        with col2:
                            st.metric("Highest Score", f"{max_plagiarism:.1f}%")
                        with col3:
                            st.metric("High-Risk Texts", flagged_count)
        
        else:
            st.info("CSV and JSON upload functionality coming soon!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔍 AI Plagiarism Detective | Powered by LangGraph Workflow, OpenAI, Google Search & Advanced ML</p>
        <p>Built with Streamlit & LangGraph | © 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()