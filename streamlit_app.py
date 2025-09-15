import streamlit as st
import os
import json
from io import StringIO

# Simple RCA (Root Cause Analysis) App
st.title("Root Cause Analysis Agent")
st.write("Upload a document and ask questions to identify root causes.")

# File uploader
uploaded_file = st.file_uploader("Choose a document", type=['txt', 'md'])

if uploaded_file is not None:
    # Read the uploaded file
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    file_content = stringio.read()
    
    st.subheader("Document Content:")
    st.text_area("Content", file_content, height=200, disabled=True)
    
    # Simple Q&A interface
    st.subheader("Ask Questions:")
    question = st.text_input("Enter your question about the document:")
    
    if question and st.button("Analyze"):
        st.subheader("Analysis:")
        
        # Simple keyword-based analysis (placeholder for LangChain integration)
        keywords = question.lower().split()
        relevant_sentences = []
        
        for sentence in file_content.split('.'):
            sentence = sentence.strip()
            if sentence and any(keyword in sentence.lower() for keyword in keywords):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            st.write("**Relevant information found:**")
            for i, sentence in enumerate(relevant_sentences[:3], 1):
                st.write(f"{i}. {sentence}...")
        else:
            st.write("No directly relevant information found in the document.")
            st.write("Try asking about: causes, issues, problems, failures, or reasons.")
    
    # ============================================================================
    # HYPOTHESIS GENERATOR SECTION
    # ============================================================================
    # TODO: Future expansion - integrate with LLM and RAG for sophisticated hypothesis generation
    # TODO: Add vector embeddings for semantic similarity matching
    # TODO: Implement hypothesis validation and ranking mechanisms
    
    st.markdown("---")
    st.subheader("🔬 Hypothesis Generator")
    st.write("Generate testable hypotheses based on the document content and your RCA query.")
    
    hypothesis_query = st.text_input("Enter your RCA query for hypothesis generation:", 
                                   placeholder="e.g., Why did the system fail? What caused the incident?")
    
    if hypothesis_query and st.button("Generate Hypotheses"):
        st.subheader("Generated Hypotheses:")
        
        # Simple keyword-based mock hypothesis generator
        # TODO: Replace with LLM-based hypothesis generation
        def generate_mock_hypotheses(query, content):
            """Mock hypothesis generator - placeholder for future LLM integration"""
            
            # Extract key terms from query and content for hypothesis generation
            query_keywords = set(query.lower().split())
            
            # Basic hypothesis templates based on common RCA patterns
            hypotheses = []
            
            # Hypothesis 1: Process/Procedure related
            if any(word in query.lower() for word in ['fail', 'error', 'problem', 'issue']):
                hypotheses.append({
                    "id": "H001",
                    "claim": "Process deviation or procedural non-compliance caused the incident",
                    "rationale": f"Based on query keywords: {', '.join(list(query_keywords)[:3])}, process-related factors are common root causes",
                    "tests": [
                        "Review process documentation and adherence",
                        "Audit recent procedural changes",
                        "Interview process stakeholders"
                    ]
                })
            
            # Hypothesis 2: Technical/System related
            hypotheses.append({
                "id": "H002",
                "claim": "Technical system malfunction or configuration error contributed to the issue",
                "rationale": "System-related causes account for significant portion of operational incidents",
                "tests": [
                    "Analyze system logs and error patterns",
                    "Review recent system changes",
                    "Perform technical diagnostics"
                ]
            })
            
            # Hypothesis 3: Human factors
            if any(word in content.lower() for word in ['training', 'staff', 'employee', 'user', 'operator']):
                hypotheses.append({
                    "id": "H003",
                    "claim": "Human factors such as training gaps or workload contributed to the incident",
                    "rationale": "Document mentions staff/training elements, suggesting human factors relevance",
                    "tests": [
                        "Assess training records and competency",
                        "Evaluate workload and staffing levels",
                        "Review communication patterns"
                    ]
                })
            
            # Hypothesis 4: Environmental/External factors (if relevant keywords found)
            if any(word in query.lower() for word in ['environment', 'external', 'weather', 'supplier']):
                hypotheses.append({
                    "id": "H004",
                    "claim": "External environmental factors or supplier issues influenced the outcome",
                    "rationale": "Query suggests external factors may be relevant to the incident",
                    "tests": [
                        "Review external conditions during incident timeframe",
                        "Audit supplier performance and deliverables",
                        "Assess environmental monitoring data"
                    ]
                })
            
            # Return 3-6 hypotheses (limit based on relevance)
            return hypotheses[:min(6, len(hypotheses))]
        
        # Generate hypotheses using mock agent
        generated_hypotheses = generate_mock_hypotheses(hypothesis_query, file_content)
        
        # Display hypotheses count
        st.info(f"Generated {len(generated_hypotheses)} testable hypotheses")
        
        # Display hypotheses as formatted JSON
        st.json(generated_hypotheses)
        
        # Optional: Display hypotheses in a more readable format
        with st.expander("📋 Structured Hypothesis View", expanded=False):
            for hyp in generated_hypotheses:
                st.markdown(f"**{hyp['id']}: {hyp['claim']}**")
                st.write(f"*Rationale:* {hyp['rationale']}")
                st.write("*Suggested Tests:*")
                for i, test in enumerate(hyp['tests'], 1):
                    st.write(f"  {i}. {test}")
                st.markdown("---")
else:
    st.info("Please upload a document to begin analysis.")

# Footer
st.markdown("---")
st.markdown("*This is a minimal RCA application built with Streamlit.*")
st.markdown("*Hypothesis Generator: Mock implementation - Future versions will integrate LLM and RAG capabilities.*")
