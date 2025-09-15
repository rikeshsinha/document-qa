import streamlit as st
import os
from io import StringIO

# Simple RCA (Root Cause Analysis) App with Streamlit
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

else:
    st.info("Please upload a document to begin analysis.")

# Footer
st.markdown("---")
st.markdown("*This is a minimal RCA application built with Streamlit.*")
