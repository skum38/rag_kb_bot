# Knowledge Graph RAG Chatbot
A powerful document question-answering system that combines Vector RAG, Knowledge Graphs, and Web Search for intelligent multi-hop reasoning over PDF documents.

🌟 Features
# Core Capabilities

📄 PDF Processing: Upload and process multi-page PDF documents
🖼️ Image Support: OCR-based text extraction from images
🕸️ Knowledge Graph: Automatic entity and relationship extraction
🔍 Hybrid Search: FAISS vector search + BM25 keyword search
🌐 Web Search: Fallback to DuckDuckGo for external knowledge
💬 Multi-modal: Support for text, images, tables, and diagrams

# Advanced Features
Multi-hop Reasoning: Answer complex questions requiring multiple inference steps
Chapter Extraction: Automatic table of contents detection
Chapter Summarization: Generate summaries of specific chapters
Real-time Toggle: Enable/disable Knowledge Graph and Web Search during chat
Context Caching: Fast responses for repeated queries
Visual Q&A: Answer questions about diagrams, tables, and images

## Start the Application
- streamlit run app.py
