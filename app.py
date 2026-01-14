import streamlit as st
import tempfile
import os
import re
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

from pdf2image import convert_from_path
import pytesseract
import camelot
import base64
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# ✅ Knowledge Graph imports
from kg_utils import (
    SimpleKnowledgeGraph, 
    build_kg_from_chunks, 
    get_kg_context_enhanced,
    extract_keywords_from_question,
    set_llm
)

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(page_title="KG-RAG Chatbot", layout="wide")
st.title("📄 Knowledge Graph RAG Chatbot")

# -------------------------------------------------
# Session State
# -------------------------------------------------
st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("bm25", None)
st.session_state.setdefault("all_chunks", [])
st.session_state.setdefault("chapter_titles", [])
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pdf_indexed", False)
st.session_state.setdefault("upload_status", "Awaiting PDF upload")
st.session_state.setdefault("retrieval_cache", {})
st.session_state.setdefault("answer_cache", {})
st.session_state.setdefault("vision_only", False)
st.session_state.setdefault("uploaded_image_base64", None)
st.session_state.setdefault("last_uploaded_file", None)
st.session_state.setdefault("current_file_type", None)
st.session_state.setdefault("web_search_enabled", True)
st.session_state.setdefault("image_chunks", [])
st.session_state.setdefault("table_chunks", [])

# ✅ Knowledge Graph session state
st.session_state.setdefault("knowledge_graph", None)
st.session_state.setdefault("kg_enabled", True)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    if st.button("🔄 Reset App"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown("### 📌 Upload Status")
    st.info(st.session_state.upload_status)
    
    st.markdown("### 🌐 Web Search")
    st.session_state.web_search_enabled = st.checkbox(
        "Enable web search for science topics", 
        value=st.session_state.web_search_enabled
    )
    
    # ✅ Knowledge Graph Toggle
    st.markdown("### 🕸️ Knowledge Graph")
    st.session_state.kg_enabled = st.checkbox(
        "Enable Knowledge Graph RAG",
        value=st.session_state.kg_enabled,
        help="Use graph relationships for better multi-hop reasoning"
    )
    
    # Show KG stats if available
    if st.session_state.knowledge_graph:
        stats = st.session_state.knowledge_graph.get_stats()
        st.write(f"**Entities:** {stats['entities']}")
        st.write(f"**Relationships:** {stats['relationships']}")
        
        if st.button("📊 Visualize Graph"):
            with st.spinner("Creating visualization..."):
                st.session_state.knowledge_graph.visualize("current_kg.png")
                if os.path.exists("current_kg.png"):
                    st.image("current_kg.png", caption="Knowledge Graph")
    
    st.markdown("### 🔍 Debug Info")
    st.write(f"Vision Only: {st.session_state.vision_only}")
    st.write(f"PDF Indexed: {st.session_state.pdf_indexed}")
    st.write(f"Current File Type: {st.session_state.current_file_type}")
    st.write(f"Chunks Count: {len(st.session_state.all_chunks)}")
    st.write(f"Web Search: {st.session_state.web_search_enabled}")
    st.write(f"KG Enabled: {st.session_state.kg_enabled}")
    st.write(f"Chapters: {len(st.session_state.chapter_titles)}")

# -------------------------------------------------
# Models
# -------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm.invoke("warm up")
    return llm

@st.cache_resource
def load_web_search():
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
        search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
        return search_tool
    except Exception as e:
        st.sidebar.warning(f"⚠️ Web search unavailable: {e}")
        return None

embeddings = load_embeddings()
answer_llm = load_llm()
web_search = load_web_search()

# ✅ Set LLM for KG extraction
set_llm(answer_llm)

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def reset_rag_state():
    """Clear all state including KG"""
    st.session_state.vectorstore = None
    st.session_state.bm25 = None
    st.session_state.all_chunks = []
    st.session_state.chapter_titles = []
    st.session_state.retrieval_cache = {}
    st.session_state.answer_cache = {}
    st.session_state.messages = []
    st.session_state.uploaded_image_base64 = None
    st.session_state.pdf_indexed = False
    st.session_state.vision_only = False
    st.session_state.current_file_type = None
    st.session_state.knowledge_graph = None

def image_to_base64(uploaded_file):
    uploaded_file.seek(0)
    return base64.b64encode(uploaded_file.read()).decode("utf-8")

def llm_text(out):
    return out.content if hasattr(out, "content") else str(out)

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def is_summary_request(q):
    return any(p in q.lower() for p in ["summarize", "summary", "summarise"])

def extract_chapter_name_from_query(q):
    match = re.search(r"summarize\s+(?:the\s+)?(.+?)\s+chapter", q.lower())
    return match.group(1).strip() if match else None

def is_chapter_titles_request(q):
    """Enhanced detection for chapter list requests"""
    chapter_keywords = [
        "chapter titles", "titles of all chapters", "list of chapters",
        "chapters in the document", "table of contents", "index of chapters",
        "all chapters", "show chapters", "what chapters", "list chapters",
        "chapter names", "chapters are", "chapters in", "provide chapters",
        "give me chapters", "show me chapters", "tell me chapters"
    ]
    q_lower = q.lower().strip()
    
    # Direct match
    if any(keyword in q_lower for keyword in chapter_keywords):
        return True
    
    # Pattern matching
    patterns = [
        r'\blist\s+(?:all\s+)?(?:the\s+)?chapters?\b',
        r'\bshow\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?chapters?\b',
        r'\bwhat\s+(?:are\s+)?(?:all\s+)?(?:the\s+)?chapters?\b',
        r'\bchapters?\s+(?:in\s+)?(?:the\s+)?(?:document|book|pdf)\b',
    ]
    
    return any(re.search(pattern, q_lower) for pattern in patterns)

def is_definition_question(q):
    return q.lower().startswith(("who", "what", "where", "when", "how"))

def is_visual_question(q):
    return any(k in q.lower() for k in [
        "table", "diagram", "figure", "graph", "chart", "image", "illustration"
    ])

def is_image_specific_question(q):
    image_indicators = [
        "image", "picture", "photo", "diagram", "this", "what's in", "what is in",
        "describe", "show", "see", "visible", "displayed", "shown", "appears",
        "what does", "can you see", "in the image", "in this", "from the image",
        "analyze", "explain this", "what's this", "what is this"
    ]
    return any(indicator in q.lower() for indicator in image_indicators)

def is_multi_hop_question(q):
    """Detect questions that require connecting multiple pieces of information"""
    multi_hop_patterns = [
        "who discovered", "who invented", "who created", "what did.*do",
        "how are.*related", "what is the relationship", "connection between",
        "what else", "also", "besides", "in addition to", "and.*discover",
        "both.*and", "along with", "as well as"
    ]
    return any(re.search(pattern, q.lower()) for pattern in multi_hop_patterns)

SCIENCE_KEYWORDS = {
    "force", "motion", "velocity", "acceleration", "gravity",
    "energy", "work", "power", "circle", "circular",
    "atom", "molecule", "chemical", "reaction", "element", "compound",
    "cell", "DNA", "organism", "photosynthesis", "respiration",
    "electricity", "current", "voltage", "resistance", "circuit",
    "wave", "light", "sound", "frequency", "wavelength",
    "newton", "joule", "watt", "momentum", "friction",
    "pressure", "density", "mass", "volume", "temperature",
    "doppler", "effect", "quantum", "relativity", "eclipse",
    "magnetic", "magnet", "electromagnetic", "electron", "proton", "neutron",
    "gravitation", "gravitational", "laws", "inertia"
}

def is_science_question(q):
    return any(k in q.lower() for k in SCIENCE_KEYWORDS)

def is_insufficient_answer(answer):
    insufficient_phrases = [
        "does not contain", "not mentioned", "not found", "cannot explain",
        "no information", "not provided", "not available", "i cannot",
        "unable to", "insufficient information", "cannot provide"
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in insufficient_phrases) or len(answer.strip()) < 80

def web_search_science(query):
    if not web_search:
        return None
    try:
        search_query = f"{query} science explanation"
        results = web_search.run(search_query)
        return results
    except:
        return None

def extract_uploaded_image_text(image_file):
    try:
        image_file.seek(0)
        img = Image.open(image_file).convert("RGB")
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

def fallback_extract_chapters_regex(pages):
    text = "\n".join(p.page_content for p in pages[:10])
    patterns = [
        r"(Chapter\s+\d+[:\-]?\s+.+)",
        r"(\d+\.\s+[A-Z][^\n]+)",
    ]
    chapters = []
    for pat in patterns:
        chapters += re.findall(pat, text)
    return list(dict.fromkeys(c.strip() for c in chapters))

def extract_chapters_enhanced(pages):
    """Enhanced chapter extraction using multiple patterns"""
    all_text = "\n".join(p.page_content for p in pages[:15])  # Check first 15 pages
    
    chapters = []
    
    # Pattern 1: "Chapter 1: Title" or "Chapter 1 - Title"
    pattern1 = re.findall(r'Chapter\s+(\d+)[:\-\s]+([A-Z][^\n]{5,60})', all_text, re.IGNORECASE)
    if pattern1:
        chapters = [f"Chapter {num}: {title.strip()}" for num, title in pattern1]
    
    # Pattern 2: "1. Title" (numbered list style)
    if not chapters:
        pattern2 = re.findall(r'^(\d+)\.\s+([A-Z][^\n]{5,60})', all_text, re.MULTILINE)
        if pattern2:
            chapters = [f"{num}. {title.strip()}" for num, title in pattern2 if int(num) <= 20]
    
    # Pattern 3: All caps titles (TABLE OF CONTENTS style)
    if not chapters:
        pattern3 = re.findall(r'^([A-Z\s]{10,60})$', all_text, re.MULTILINE)
        if pattern3:
            chapters = [title.strip().title() for title in pattern3[:15]]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_chapters = []
    for ch in chapters:
        ch_normalized = ch.lower().strip()
        if ch_normalized not in seen and len(ch.strip()) > 5:
            seen.add(ch_normalized)
            unique_chapters.append(ch.strip())
    
    return unique_chapters[:20]  # Max 20 chapters

def normalize_chapter_name(name):
    return re.sub(r"(chapter\s*\d+[:\-]?)", "", name.lower()).strip()

def retrieve_chapter_content(chapter_name):
    chapter_key = normalize_chapter_name(chapter_name)
    chunks = [c for c in st.session_state.all_chunks if c.metadata.get("page", 0) >= 5]
    
    start_idx = None
    for i, c in enumerate(chunks):
        text = c.page_content.lower()
        if (re.search(rf"\b\d+\.\s*{re.escape(chapter_key)}", text) or 
            text.strip().startswith(chapter_key)):
            start_idx = i
            break
    
    if start_idx is None:
        return []
    
    end_idx = len(chunks)
    for j in range(start_idx + 1, len(chunks)):
        if re.search(r"\b\d+\.\s+[a-z]", chunks[j].page_content.lower()):
            end_idx = j
            break
    
    return chunks[start_idx:end_idx][:12]

def extract_toc_at_ingestion(pages):
    raw_text = "\n\n".join(p.page_content for p in pages[:8])[:12000]
    prompt = f"""Extract ONLY the chapter titles from the Table of Contents or Index.
Do NOT invent. Ignore page numbers. Return as a numbered list.
If none exist, return EXACTLY: NONE.

Text: {raw_text}"""
    
    result = llm_text(answer_llm.invoke(prompt)).strip()
    if result.upper() == "NONE":
        return []
    return [line.strip() for line in result.split("\n") 
            if re.match(r"^\d+[\.\)]\s+", line.strip())]

def extract_images_as_text(pdf_path, max_pages=5):
    image_docs = []
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=min(max_pages, 5))
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            if len(text.strip()) > 40:
                image_docs.append({"text": text, "page": i + 1})
    except Exception as e:
        st.warning(f"Image extraction skipped: {e}")
    return image_docs

def extract_tables_as_text(pdf_path, max_pages=10):
    table_docs = []
    try:
        pages_str = f"1-{min(max_pages, 10)}"
        tables = camelot.read_pdf(pdf_path, pages=pages_str, flavor="lattice")
        if tables.n == 0:
            tables = camelot.read_pdf(pdf_path, pages=pages_str, flavor="stream")
        
        for table in tables:
            text = table.df.to_string(index=False)
            if len(text.strip()) > 40:
                table_docs.append({"text": text, "page": table.page})
    except Exception as e:
        st.warning(f"Table extraction skipped: {e}")
    return table_docs

# -------------------------------------------------
# File Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF or Image",
    type=["pdf", "png", "jpg", "jpeg"],
    key="file_uploader"
)

if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.type}"
    
    if st.session_state.last_uploaded_file != file_id or not st.session_state.pdf_indexed:
        reset_rag_state()
        st.session_state.last_uploaded_file = file_id
        
        # IMAGE PROCESSING
        if uploaded_file.type.startswith("image"):
            st.session_state.current_file_type = "image"
            st.session_state.upload_status = "⏳ Processing image..."
            st.session_state.uploaded_image_base64 = image_to_base64(uploaded_file)
            image_text = extract_uploaded_image_text(uploaded_file)
            
            if image_text.strip() and len(image_text.strip()) > 20:
                image_doc = Document(
                    page_content=f"[UPLOADED IMAGE OCR]\n{image_text}",
                    metadata={"page": 1, "type": "image"}
                )
                st.session_state.vectorstore = FAISS.from_documents([image_doc], embeddings)
                st.session_state.bm25 = BM25Okapi([tokenize(image_doc.page_content)])
                st.session_state.all_chunks = [image_doc]
                st.session_state.pdf_indexed = True
                st.session_state.vision_only = False
                st.session_state.upload_status = "✅ Image ready (OCR + Vision mode)"
            else:
                st.session_state.pdf_indexed = True
                st.session_state.vision_only = True
                st.session_state.upload_status = "✅ Image ready (Vision mode only)"
        
        # PDF PROCESSING WITH KG
        elif uploaded_file.type == "application/pdf":
            st.session_state.current_file_type = "pdf"
            st.session_state.upload_status = "⏳ Indexing PDF..."
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name
            
            status_text.text("📄 Loading PDF pages...")
            progress_bar.progress(10)
            pages = PyPDFLoader(pdf_path).load()
            
            # ✅ ENHANCED: Chapter extraction with multiple methods
            status_text.text("📚 Extracting table of contents...")
            progress_bar.progress(20)
            
            titles = []
            
            # Method 1: LLM-based extraction
            try:
                titles = extract_toc_at_ingestion(pages)
                if titles:
                    st.sidebar.success(f"✅ Found {len(titles)} chapters (LLM)")
            except Exception as e:
                st.sidebar.warning(f"LLM extraction failed: {e}")
            
            # Method 2: Regex fallback
            if not titles:
                try:
                    titles = fallback_extract_chapters_regex(pages)
                    if titles:
                        st.sidebar.success(f"✅ Found {len(titles)} chapters (Regex)")
                except Exception as e:
                    st.sidebar.warning(f"Regex extraction failed: {e}")
            
            # Method 3: Enhanced pattern matching
            if not titles:
                try:
                    titles = extract_chapters_enhanced(pages)
                    if titles:
                        st.sidebar.success(f"✅ Found {len(titles)} chapters (Enhanced)")
                except Exception as e:
                    st.sidebar.warning(f"Enhanced extraction failed: {e}")
            
            st.session_state.chapter_titles = titles
            
            # Show preview in sidebar
            if titles:
                with st.sidebar.expander(f"📚 Found {len(titles)} Chapters"):
                    for i, title in enumerate(titles, 1):
                        st.write(f"{i}. {title[:60]}...")
            else:
                st.sidebar.warning("⚠️ No chapters detected.")
            
            status_text.text("✂️ Splitting text into chunks...")
            progress_bar.progress(40)
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=200)
            chunks = splitter.split_documents(pages)
            
            status_text.text("🖼️ Extracting images...")
            progress_bar.progress(55)
            for img in extract_images_as_text(pdf_path, max_pages=5):
                chunks.append(Document(
                    page_content=f"[IMAGE CONTENT]\n{img['text']}",
                    metadata={"page": img["page"], "type": "image"}
                ))
            
            status_text.text("📊 Extracting tables...")
            progress_bar.progress(65)
            for tbl in extract_tables_as_text(pdf_path, max_pages=10):
                chunks.append(Document(
                    page_content=f"[TABLE CONTENT]\n{tbl['text']}",
                    metadata={"page": tbl["page"], "type": "table"}
                ))
            
            os.remove(pdf_path)
            
            # ✅ Build Knowledge Graph
            if st.session_state.kg_enabled:
                status_text.text("🕸️ Building Knowledge Graph...")
                progress_bar.progress(75)
                
                try:
                    st.session_state.knowledge_graph = build_kg_from_chunks(
                        chunks, 
                        answer_llm, 
                        max_chunks=30
                    )
                    stats = st.session_state.knowledge_graph.get_stats()
                    status_text.text(f"✅ KG built: {stats['entities']} entities, {stats['relationships']} relations")
                except Exception as e:
                    st.warning(f"⚠️ KG building failed: {e}")
                    st.session_state.knowledge_graph = None
            
            status_text.text("🔍 Building vector index...")
            progress_bar.progress(85)
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.all_chunks = chunks
            
            status_text.text("📑 Building BM25 index...")
            progress_bar.progress(95)
            tokenized = [tokenize(c.page_content) for c in chunks]
            st.session_state.bm25 = BM25Okapi(tokenized)
            
            st.session_state.pdf_indexed = True
            progress_bar.progress(100)
            status_text.text("✅ PDF ready!")
            st.session_state.upload_status = "✅ PDF indexed and ready"
            
            st.success(f"✅ Indexed {len(chunks)} chunks from {len(pages)} pages")
            if st.session_state.chapter_titles:
                st.info(f"📚 Found {len(st.session_state.chapter_titles)} chapters")
            if st.session_state.knowledge_graph:
                kg_stats = st.session_state.knowledge_graph.get_stats()
                st.info(f"🕸️ Knowledge Graph: {kg_stats['entities']} entities, {kg_stats['relationships']} relationships")

# -------------------------------------------------
# Hybrid Retrieval
# -------------------------------------------------
def hybrid_retrieve(query):
    """Hybrid retrieval combining FAISS and BM25"""
    key = query.lower().strip()
    
    if key in st.session_state.retrieval_cache:
        return st.session_state.retrieval_cache[key]

    faiss_docs = st.session_state.vectorstore.similarity_search(query, k=6)
    scores = st.session_state.bm25.get_scores(tokenize(query))

    bm25_docs = [
        st.session_state.all_chunks[i]
        for i in np.argsort(scores)[::-1][:6]
        if scores[i] > 0
    ]

    merged, seen = [], set()
    for d in faiss_docs + bm25_docs:
        sig = d.page_content[:120]
        if sig not in seen:
            seen.add(sig)
            merged.append(d)
        if len(merged) >= 8:
            break

    st.session_state.retrieval_cache[key] = merged
    return merged

# -------------------------------------------------
# ✅ COMPLETE FIXED RAG ANSWER FUNCTION
# -------------------------------------------------
def rag_answer(question):
    """
    Main RAG answer function with:
    - Real-time KG enable/disable support
    - Knowledge Graph integration
    - Strict grounding to document
    - Web search fallback
    - Multi-modal support
    """
    qkey = question.lower().strip()

    # Check cache (but not for dynamic queries)
    if (qkey in st.session_state.answer_cache and 
        not is_chapter_titles_request(question) and 
        not is_summary_request(question)):
        return st.session_state.answer_cache[qkey]

    # Handle chapter titles request
    if is_chapter_titles_request(question):
        if st.session_state.chapter_titles:
            chapters_list = "\n".join([f"{i+1}. {ch}" for i, ch in enumerate(st.session_state.chapter_titles)])
            return f"📚 **Chapters in the document:**\n\n{chapters_list}"
        else:
            return "❌ No chapters found in the document. The PDF may not have a clear table of contents."

    # Handle chapter summary
    if is_summary_request(question):
        chapter_name = extract_chapter_name_from_query(question)
        if not chapter_name:
            return "❌ Please specify which chapter to summarize (e.g., 'summarize gravitation chapter')"
        
        chapter_chunks = retrieve_chapter_content(chapter_name)
        if not chapter_chunks:
            return f"❌ Could not find chapter: {chapter_name}"
        
        context = "\n\n".join(c.page_content for c in chapter_chunks)
        prompt = f"""Summarize this chapter content in a clear, structured way:

{context}

Provide:
1. Main topics covered
2. Key concepts and definitions
3. Important formulas or principles (if any)
4. Summary (3-4 sentences)"""
        
        summary = llm_text(answer_llm.invoke(prompt))
        st.session_state.answer_cache[qkey] = summary
        return summary

    is_sci_q = is_science_question(question)
    is_multi_hop = is_multi_hop_question(question)
    
    # Get vector RAG context
    docs = hybrid_retrieve(question)
    
    # ✅ ENHANCED DEBUG
    with st.sidebar.expander("🔍 Debug: Retrieved Chunks"):
        st.write(f"**Question:** {question}")
        st.write(f"**Question Type:**")
        st.write(f"  - Science Q: {is_sci_q}")
        st.write(f"  - Multi-hop: {is_multi_hop}")
        st.write(f"**Chunks found:** {len(docs)}")
        st.write(f"**KG Enabled:** {st.session_state.kg_enabled}")
        st.write(f"**KG Exists:** {st.session_state.knowledge_graph is not None}")
        if docs:
            st.write("**First chunk preview:**")
            st.text(docs[0].page_content[:400])
    
    # ✅ FIXED: Real-time KG decision
    kg_context = ""
    kg_was_used = False
    
    if (st.session_state.kg_enabled and 
        st.session_state.knowledge_graph and 
        st.session_state.knowledge_graph.get_stats()['entities'] > 0):
        
        # Decide if this question needs KG
        should_use_kg_for_question = (
            is_multi_hop or 
            (is_sci_q and is_definition_question(question)) or
            len(docs) < 3
        )
        
        if should_use_kg_for_question:
            try:
                kg_context = get_kg_context_enhanced(
                    question, 
                    st.session_state.knowledge_graph, 
                    answer_llm
                )
                if kg_context:
                    kg_was_used = True
                    with st.sidebar.expander("🕸️ KG Context Used"):
                        st.success("✅ Knowledge Graph ACTIVE")
                        st.text(kg_context[:500])
            except Exception as e:
                st.sidebar.error(f"KG extraction error: {e}")
        else:
            with st.sidebar.expander("🕸️ KG Status"):
                st.info(f"ℹ️ KG enabled but not needed\n(Not multi-hop, {len(docs)} chunks found)")
    else:
        with st.sidebar.expander("🕸️ KG Status"):
            if not st.session_state.kg_enabled:
                st.warning("⚠️ Knowledge Graph DISABLED by user")
            elif not st.session_state.knowledge_graph:
                st.warning("⚠️ KG not built")
            else:
                st.warning("⚠️ KG has no entities")
    
    # If no results at all
    if not docs and not kg_context:
        if is_sci_q and st.session_state.web_search_enabled:
            with st.spinner("🌐 Searching the web..."):
                web_results = web_search_science(question)
                if web_results:
                    web_answer = llm_text(answer_llm.invoke(
                        f"""Answer this science question clearly:

Question: {question}
Web Results: {web_results}

Provide a clear answer suitable for a 10th grade student."""
                    ))
                    return f"🌐 **[Web Search]** _(not found in document)_\n\n{web_answer}"
        return "❌ Not found in the document."
    
    # Build context
    vector_context = "\n\n".join(d.page_content for d in docs) if docs else ""
    
    # ✅ IMPROVED: Smart relevance check (only for obvious mismatches)
    if vector_context and not kg_context:
        question_lower = question.lower()
        
        # Only reject if clearly unrelated to science textbook content
        unrelated_patterns = [
            r'\bpython\s+programming\b',
            r'\bwho\s+(developed|created|invented)\s+python\b',
            r'\bjavascript\b',
            r'\bhtml\b',
            r'\bcoding\b',
            r'\bsoftware\s+development\b',
            r'\balgorithm\s+design\b',
            r'\bdatabase\b',
            r'\bweb\s+development\b'
        ]
        
        is_unrelated = any(re.search(pattern, question_lower) for pattern in unrelated_patterns)
        
        if is_unrelated:
            return f"""❌ This information is not found in the uploaded document.

**Your question:** {question}
**Document content:** Science textbook (10th grade)

This appears to be a programming/technology question. The document is a science textbook covering topics like physics, chemistry, and biology.

💡 Please ask questions related to science topics, or enable web search for general knowledge questions."""

    # Visual question handling
    if is_visual_question(question):
        visual_context = "\n\n".join(
            d.page_content for d in docs
            if d.metadata.get("type") in ["image", "table"]
        )
        if visual_context.strip():
            return llm_text(answer_llm.invoke(
                f"Answer using this visual content:\n{visual_context}\n\nQuestion: {question}"
            ))

    # ✅ IMPROVED: Build prompt based on what's ACTUALLY available
    if kg_was_used and vector_context:
        prompt = f"""You are a helpful science teacher. Answer using BOTH sources.

KNOWLEDGE GRAPH RELATIONSHIPS:
{kg_context}

TEXTBOOK CONTENT:
{vector_context}

STUDENT'S QUESTION: {question}

INSTRUCTIONS:
- Combine information from both the knowledge graph and textbook
- Use the graph to understand relationships and connections
- Use the textbook for detailed explanations
- Provide a clear, comprehensive answer

Answer:"""
        
    elif kg_was_used and not vector_context:
        prompt = f"""You are a helpful science teacher. Answer using the knowledge graph relationships.

KNOWLEDGE GRAPH:
{kg_context}

QUESTION: {question}

Answer based on the graph relationships:"""
        
    elif vector_context and not kg_was_used:
        prompt = f"""You are a helpful science teacher explaining from a textbook.

TEXTBOOK CONTENT:
{vector_context}

STUDENT'S QUESTION: {question}

Provide a clear explanation from the textbook:"""
    
    else:
        return "❌ Insufficient information to answer."
    
    # Get answer
    answer = llm_text(answer_llm.invoke(prompt))
    
    # Check if insufficient
    if is_insufficient_answer(answer):
        if is_sci_q and st.session_state.web_search_enabled:
            with st.spinner("🌐 Searching web for additional info..."):
                web_results = web_search_science(question)
                if web_results:
                    web_answer = llm_text(answer_llm.invoke(
                        f"""Document had incomplete info. Use web results:

Question: {question}
Web Results: {web_results}

Provide comprehensive answer:"""
                    ))
                    return f"🌐 **[Web Search]** _(document incomplete)_\n\n{web_answer}"
    
    # ✅ FIXED: Add accurate source label based on what was ACTUALLY used
    source_label = ""
    
    if kg_was_used and vector_context:
        source_label = "🕸️📄 **[Knowledge Graph + Vector RAG]**"
    elif kg_was_used and not vector_context:
        source_label = "🕸️ **[Knowledge Graph Only]**"
    elif vector_context and not kg_was_used:
        source_label = "📄 **[Vector RAG Only]**"
        if st.session_state.kg_enabled:
            source_label += " _(KG available but not needed)_"
        else:
            source_label += " _(KG disabled)_"
    
    answer = f"{source_label}\n\n{answer}"
    
    # Cache and return
    st.session_state.answer_cache[qkey] = answer
    return answer

# -------------------------------------------------
# Vision Mode Answer
# -------------------------------------------------
def vision_mode_answer(question):
    """Handle vision-only mode (when OCR failed)"""
    if not st.session_state.uploaded_image_base64:
        return "❌ No image available for analysis"
    
    prompt_content = [
        {"type": "text", "text": question},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{st.session_state.uploaded_image_base64}"
            }
        }
    ]
    
    response = answer_llm.invoke([HumanMessage(content=prompt_content)])
    return llm_text(response)

# -------------------------------------------------
# Chat Interface
# -------------------------------------------------
st.markdown("### 💬 Chat with Your Document")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask a question about your document..."):
    # Check if document is ready
    if not st.session_state.pdf_indexed and not st.session_state.vision_only:
        st.error("⚠️ Please upload a PDF or image first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Vision-only mode
                if st.session_state.vision_only:
                    response = vision_mode_answer(user_input)
                # Image-specific question with uploaded image
                elif (st.session_state.uploaded_image_base64 and 
                      is_image_specific_question(user_input)):
                    response = vision_mode_answer(user_input)
                # Regular RAG
                else:
                    response = rag_answer(user_input)
                
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

# -------------------------------------------------
# Additional Features in Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Document Stats")
    
    if st.session_state.pdf_indexed:
        st.write(f"📄 **Total Chunks:** {len(st.session_state.all_chunks)}")
        st.write(f"📚 **Chapters:** {len(st.session_state.chapter_titles)}")
        
        if st.session_state.knowledge_graph:
            stats = st.session_state.knowledge_graph.get_stats()
            st.write(f"🕸️ **KG Entities:** {stats['entities']}")
            st.write(f"🔗 **KG Relations:** {stats['relationships']}")
    
    st.markdown("---")
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    **1. Upload** a PDF or image
    
    **2. Ask questions:**
    - "What is gravity?"
    - "Who is Newton?"
    - "Summarize gravitation chapter"
    - "List all chapters"
    - "Who discovered gravity and what else did they discover?" (multi-hop)
    
    **3. Toggle Features:**
    - 🕸️ Enable/disable Knowledge Graph
    - 🌐 Enable/disable web search
    
    **4. Features:**
    - 📄 Vector RAG for precise answers
    - 🕸️ Knowledge Graph for complex reasoning
    - 🌐 Web search for additional info
    - 📊 Table and image extraction
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Settings")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.answer_cache = {}
        st.rerun()
    
    if st.button("💾 Export Chat"):
        chat_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in st.session_state.messages
        ])
        st.download_button(
            "📥 Download Chat",
            chat_text,
            file_name="chat_export.txt",
            mime="text/plain"
        )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>🚀 Powered by GPT-4o-mini | 🕸️ Knowledge Graph RAG | 🔍 Hybrid Search</p>
    <p>Features: Vector RAG + BM25 + Knowledge Graph + Web Search + Multi-modal</p>
</div>
""", unsafe_allow_html=True)