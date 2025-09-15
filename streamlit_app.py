import streamlit as st
import os
import json
from io import StringIO
import numpy as np
from typing import List, Dict
import re

# For RAG functionality - can be swapped with actual implementations
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("OpenAI not installed. Using mock LLM responses.")

# Mock vector store for semantic similarity (replace with FAISS/PGVector in production)
class MockVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_document(self, doc_id: str, content: str, chunks: List[str]):
        """Add document chunks to the vector store with mock embeddings"""
        for i, chunk in enumerate(chunks):
            # Mock embedding - in production, use actual embedding model
            mock_embedding = np.random.rand(384)  # Simulate 384-dim embedding
            self.documents.append({
                'doc_id': doc_id,
                'chunk_id': f"{doc_id}_chunk_{i}",
                'content': chunk,
                'embedding': mock_embedding
            })
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Mock semantic search - replace with actual vector similarity in production"""
        if not self.documents:
            return []
        
        # Mock relevance scoring based on keyword overlap
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in self.documents:
            doc_words = set(doc['content'].lower().split())
            # Simple Jaccard similarity as mock semantic similarity
            similarity = len(query_words.intersection(doc_words)) / len(query_words.union(doc_words))
            scored_docs.append((similarity, doc))
        
        # Sort by similarity and return top_k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

def chunk_document(content: str, chunk_size: int = 500) -> List[str]:
    """Split document into chunks for better retrieval granularity"""
    # Simple sentence-based chunking
    sentences = re.split(r'[.!?]+', content)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk + sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_hypotheses_with_llm(query: str, context_chunks: List[str]) -> List[Dict]:
    """Generate RCA hypotheses using LLM with retrieved context"""
    
    # Prepare context from retrieved chunks
    context_text = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
    
    # Example schema from previous implementation
    example_output = '''[
        {
            "hypothesis": "Technical system failure in critical component",
            "likelihood": "High",
            "evidence": "System logs show repeated failures, performance degradation observed",
            "next_steps": "Analyze system logs, check hardware diagnostics, review recent changes"
        },
        {
            "hypothesis": "Process breakdown in workflow",
            "likelihood": "Medium", 
            "evidence": "Timeline analysis shows delays, communication gaps identified",
            "next_steps": "Map current process, identify bottlenecks, interview stakeholders"
        }
    ]'''
    
    prompt = f"""You are an expert Root Cause Analysis assistant. Based on the user's query and the provided context, generate 2-4 specific, actionable hypotheses for potential root causes.

User Query: {query}

Relevant Context:
{context_text}

Generate hypotheses in the exact JSON format below. Each hypothesis should be specific, evidence-based, and actionable:

{example_output}

Ensure each hypothesis includes:
- A clear, specific hypothesis statement
- Likelihood assessment (High/Medium/Low)
- Supporting evidence from the context
- Concrete next steps for validation

Respond with only the JSON array, no additional text."""
    
    if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
        try:
            # Use OpenAI API for real LLM generation
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert Root Cause Analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            result = response.choices[0].message.content.strip()
            return json.loads(result)
            
        except Exception as e:
            st.warning(f"LLM API error: {e}. Using mock response.")
    
    # Mock response when LLM is not available - maintains same schema
    return [
        {
            "hypothesis": "System configuration issue based on retrieved context",
            "likelihood": "High",
            "evidence": f"Context analysis suggests configuration problems. Retrieved {len(context_chunks)} relevant chunks.",
            "next_steps": "Review system configuration, validate against retrieved documentation, check recent changes"
        },
        {
            "hypothesis": "Process or procedural gap identified in knowledge base", 
            "likelihood": "Medium",
            "evidence": f"Knowledge store contains relevant information pointing to process issues. Query: {query[:50]}...",
            "next_steps": "Analyze process documentation from knowledge store, identify gaps, validate with stakeholders"
        }
    ]

# Initialize session state for knowledge store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = MockVectorStore()
if 'knowledge_docs' not in st.session_state:
    st.session_state.knowledge_docs = {}

# Main App
st.title("Root Cause Analysis Agent with Knowledge Store")
st.write("Upload documents to build a knowledge base, then perform RCA queries with retrieval-augmented generation.")

# === KNOWLEDGE STORE SECTION ===
st.header("📚 Knowledge Store")

# File uploader for knowledge base
uploaded_file = st.file_uploader(
    "Add document to knowledge store", 
    type=['txt', 'md'],
    help="Upload documents to build your RCA knowledge base"
)

if uploaded_file is not None:
    # Read and process the uploaded file
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    file_content = stringio.read()
    doc_id = uploaded_file.name
    
    if st.button("Add to Knowledge Store"):
        # Chunk the document for better retrieval
        chunks = chunk_document(file_content)
        
        # Add to vector store (mock implementation)
        st.session_state.vector_store.add_document(doc_id, file_content, chunks)
        
        # Store in session state for display
        st.session_state.knowledge_docs[doc_id] = {
            'content': file_content,
            'chunks': chunks,
            'chunk_count': len(chunks)
        }
        
        st.success(f"Added '{doc_id}' to knowledge store ({len(chunks)} chunks)")
        st.rerun()

# Display current knowledge store
if st.session_state.knowledge_docs:
    st.subheader("Current Knowledge Base")
    
    for doc_id, doc_info in st.session_state.knowledge_docs.items():
        with st.expander(f"📄 {doc_id} ({doc_info['chunk_count']} chunks)"):
            st.text_area(
                "Content Preview", 
                doc_info['content'][:500] + "..." if len(doc_info['content']) > 500 else doc_info['content'],
                height=200, 
                disabled=True,
                key=f"preview_{doc_id}"
            )
            
            # Show chunks for debugging/verification
            if st.checkbox(f"Show chunks for {doc_id}", key=f"chunks_{doc_id}"):
                for i, chunk in enumerate(doc_info['chunks']):
                    st.text(f"Chunk {i+1}: {chunk[:100]}...")
else:
    st.info("No documents in knowledge store yet. Upload documents above to get started.")

# === RCA QUERY SECTION ===
st.header("🔍 Root Cause Analysis Query")

if st.session_state.knowledge_docs:
    rca_query = st.text_input(
        "Enter your RCA query:",
        placeholder="e.g., Why did the system fail last week?",
        help="Ask questions about issues you want to analyze. The system will search your knowledge base for relevant context."
    )
    
    if rca_query and st.button("Generate RCA Hypotheses"):
        with st.spinner("Retrieving relevant context and generating hypotheses..."):
            
            # === RAG RETRIEVAL STEP ===
            # Search vector store for relevant chunks
            relevant_chunks = st.session_state.vector_store.search(rca_query, top_k=3)
            
            if relevant_chunks:
                st.subheader("📋 Retrieved Context")
                context_texts = []
                
                for i, chunk_info in enumerate(relevant_chunks):
                    with st.expander(f"Context {i+1} (from {chunk_info['doc_id']})"):
                        st.write(chunk_info['content'])
                    context_texts.append(chunk_info['content'])
                
                # === LLM GENERATION STEP ===
                st.subheader("🎯 Generated RCA Hypotheses")
                
                # Generate hypotheses using LLM with retrieved context
                hypotheses = generate_hypotheses_with_llm(rca_query, context_texts)
                
                # Display hypotheses in the same format as before
                for i, hypothesis in enumerate(hypotheses, 1):
                    with st.expander(f"Hypothesis {i}: {hypothesis['hypothesis']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Likelihood:** {hypothesis['likelihood']}")
                            st.write(f"**Evidence:** {hypothesis['evidence']}")
                        
                        with col2:
                            st.write(f"**Next Steps:** {hypothesis['next_steps']}")
            else:
                st.warning("No relevant context found in knowledge store. Try adding more documents or refining your query.")
else:
    st.info("Please add documents to the knowledge store before performing RCA queries.")

# === SYSTEM INFO ===
with st.sidebar:
    st.header("System Information")
    st.write(f"**Documents in store:** {len(st.session_state.knowledge_docs)}")
    st.write(f"**Total chunks:** {sum(doc['chunk_count'] for doc in st.session_state.knowledge_docs.values())}")
    
    if st.button("Clear Knowledge Store"):
        st.session_state.knowledge_docs = {}
        st.session_state.vector_store = MockVectorStore()
        st.success("Knowledge store cleared!")
        st.rerun()
    
    # Technical notes
    with st.expander("🔧 Technical Notes"):
        st.write("""
        **Current Implementation:**
        - Mock vector embeddings (replace with actual embeddings)
        - Simple keyword-based similarity (replace with semantic similarity)
        - OpenAI integration ready (set OPENAI_API_KEY)
        - FAISS/PGVector ready for production swap
        
        **Production Upgrades:**
        - Replace MockVectorStore with FAISS or PGVector
        - Use sentence-transformers for real embeddings
        - Add document preprocessing and cleaning
        - Implement persistent storage
        """)
