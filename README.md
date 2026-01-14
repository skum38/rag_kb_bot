# Solution 1 - Knowledge Graph RAG Chatbot
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




# Solution-2 📄 Hybrid RAG Chatbot with Vision & Web Search 

A powerful multimodal chatbot built with Streamlit that combines **Retrieval-Augmented Generation (RAG)**, **Vision AI**, and **Web Search** to answer questions from PDFs, images, and the internet.

---

## 🌟 Features

### 📚 **PDF Processing**
- ✅ Extract text, images, and tables from PDF documents
- ✅ Automatic chapter detection and table of contents extraction
- ✅ Hybrid retrieval using FAISS (vector search) + BM25 (keyword search)
- ✅ Smart chunking with overlap for better context

### 🖼️ **Image Analysis**
- ✅ Vision AI (GPT-4o-mini) for image understanding
- ✅ OCR extraction using Tesseract for text-heavy images
- ✅ Automatic fallback to vision-only mode for complex images

### 🌐 **Web Search Integration**
- ✅ DuckDuckGo search for science topics
- ✅ Automatic web search when document lacks information
- ✅ Toggle web search on/off via sidebar

### 🧠 **Smart Question Routing**
- ✅ Detects question type (chapter summary, definition, concept, etc.)
- ✅ Prevents hallucination by validating question-content match
- ✅ Clear error messages when information unavailable

### 🚀 **Performance Optimizations**
- ✅ Response caching for faster repeated queries
- ✅ Limited page processing for images/tables (configurable)
- ✅ Progress bar for PDF indexing
- ✅ Session state management for file persistence


## Start the Application
- streamlit run app.py
