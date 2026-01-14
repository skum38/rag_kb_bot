# 📄 Hybrid RAG Chatbot with Vision & Web Search + Knowledge Graph RAG Chatbot

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
