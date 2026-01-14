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
# 🔹 APPENDED: vision + table imports
from pdf2image import convert_from_path
import pytesseract
import camelot
import base64

from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ NEW: Web search imports
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot")

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
st.session_state.setdefault("web_search_enabled", True)  # ✅ NEW


# 🔹 APPENDED: multimodal storage
st.session_state.setdefault("image_chunks", [])
st.session_state.setdefault("table_chunks", [])

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
    
    # ✅ NEW: Web search toggle
    st.markdown("### 🌐 Web Search")
    st.session_state.web_search_enabled = st.checkbox(
        "Enable web search for science topics", 
        value=st.session_state.web_search_enabled
    )
    
    # ✅ DEBUG INFO
    st.markdown("### 🔍 Debug Info")
    st.write(f"Vision Only: {st.session_state.vision_only}")
    st.write(f"PDF Indexed: {st.session_state.pdf_indexed}")
    st.write(f"Current File Type: {st.session_state.current_file_type}")
    st.write(f"Chunks Count: {len(st.session_state.all_chunks)}")
    st.write(f"Web Search Enabled: {st.session_state.web_search_enabled}")
    st.write(f"Chapters Found: {len(st.session_state.chapter_titles)}")

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

# ✅ NEW: Web search initialization
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
web_search = load_web_search()  # ✅ NEW

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def reset_rag_state():
    """✅ FIXED: Properly clear all state including chapters"""
    st.session_state.vectorstore = None
    st.session_state.bm25 = None
    st.session_state.all_chunks = []
    st.session_state.chapter_titles = []  # ✅ Clear chapters
    st.session_state.retrieval_cache = {}
    st.session_state.answer_cache = {}
    st.session_state.messages = []
    st.session_state.uploaded_image_base64 = None
    st.session_state.pdf_indexed = False
    st.session_state.vision_only = False
    st.session_state.current_file_type = None


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
    return any(p in q.lower() for p in [
        "chapter titles",
        "titles of all chapters",
        "list of chapters",
        "chapters in the document",
        "table of contents",
        "index of chapters"
    ])

def is_definition_question(q):
    return q.lower().startswith(("who", "what", "where", "when", "how"))

def is_emotion_inference_question(q):
    return ("how might" in q.lower() or "how would" in q.lower()) and "felt" in q.lower()

def is_study_plan_request(q):
    return "study plan" in q.lower() or "revision plan" in q.lower()

# 🔹 APPENDED: visual question detection
def is_visual_question(q):
    return any(k in q.lower() for k in [
        "table", "diagram", "figure", "graph", "chart", "image", "illustration"
    ])

# ✅ NEW: Check if question references the image
def is_image_specific_question(q):
    """Check if the question is specifically asking about the uploaded image"""
    image_indicators = [
        "image", "picture", "photo", "diagram", "this", "what's in", "what is in",
        "describe", "show", "see", "visible", "displayed", "shown", "appears",
        "what does", "can you see", "in the image", "in this", "from the image",
        "analyze", "explain this", "what's this", "what is this"
    ]
    return any(indicator in q.lower() for indicator in image_indicators)

# ---------------- SCIENCE ROUTING (ENHANCED) ----------------
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
    "gravitation", "gravitational"  # ✅ Added for your example
}

def is_science_question(q):
    q_lower = q.lower()
    is_science = any(k in q_lower for k in SCIENCE_KEYWORDS)
    return is_science

def is_concept_identification_question(q):
    return (
        is_science_question(q)
        and any(p in q.lower() for p in [
            "what type of force",
            "what type of motion",
            "this is called",
            "this force is called"
        ])
    )

# ✅ NEW: Check if answer is insufficient
def is_insufficient_answer(answer):
    """Detect if the LLM couldn't find the answer in the document"""
    insufficient_phrases = [
        "does not contain",
        "not mentioned",
        "not found",
        "cannot explain",
        "no information",
        "not provided",
        "not available",
        "i cannot",
        "unable to",
        "insufficient information",
        "cannot provide"
    ]
    answer_lower = answer.lower()
    is_insufficient = any(phrase in answer_lower for phrase in insufficient_phrases)
    
    return is_insufficient or len(answer.strip()) < 80

# ✅ NEW: Web search function
def web_search_science(query):
    """Search the web for science topics"""
    if not web_search:
        return None
    try:
        search_query = f"{query} science explanation"
        results = web_search.run(search_query)
        return results
    except Exception as e:
        return None

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

def normalize_chapter_name(name):
    return re.sub(r"(chapter\s*\d+[:\-]?)", "", name.lower()).strip()

# -------------------------------------------------
# 🔹 APPENDED: standalone image OCR
# -------------------------------------------------
def extract_uploaded_image_text(image_file):
    try:
        image_file.seek(0)
        img = Image.open(image_file).convert("RGB")
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""


# -------------------------------------------------
# FIXED (MINIMAL): Chapter retrieval
# -------------------------------------------------
def retrieve_chapter_content(chapter_name):
    chapter_key = normalize_chapter_name(chapter_name)

    chunks = [
        c for c in st.session_state.all_chunks
        if c.metadata.get("page", 0) >= 5
    ]

    start_idx = None
    for i, c in enumerate(chunks):
        text = c.page_content.lower()
        if (
            re.search(rf"\b\d+\.\s*{re.escape(chapter_key)}", text)
            or text.strip().startswith(chapter_key)
        ):
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

# -------------------------------------------------
# TOC extraction (UNCHANGED)
# -------------------------------------------------
def extract_toc_at_ingestion(pages):
    raw_text = "\n\n".join(p.page_content for p in pages[:8])[:12000]
    prompt = f"""
Extract ONLY the chapter titles from the Table of Contents or Index.
Do NOT invent.
Ignore page numbers.
Return as a numbered list.
If none exist, return EXACTLY: NONE.

Text:
{raw_text}
"""
    result = llm_text(answer_llm.invoke(prompt)).strip()
    if result.upper() == "NONE":
        return []
    return [
        line.strip()
        for line in result.split("\n")
        if re.match(r"^\d+[\.\)]\s+", line.strip())
    ]

# -------------------------------------------------
# 🔹 OPTIMIZED: image & table extraction - LIMITED PAGES
# -------------------------------------------------
def extract_images_as_text(pdf_path, max_pages=5):
    image_docs = []
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=min(max_pages, 5))
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            if len(text.strip()) > 40:
                image_docs.append({
                    "text": text,
                    "page": i + 1
                })
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
                table_docs.append({
                    "text": text,
                    "page": table.page
                })
    except Exception as e:
        st.warning(f"Table extraction skipped: {e}")
    return table_docs

# -------------------------------------------------
# ✅ OPTIMIZED: File Upload Handler with Progress Bar
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF or Image",
    type=["pdf", "png", "jpg", "jpeg"],
    key="file_uploader"
)

# ✅ Process uploaded file
if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.type}"
    
    if st.session_state.last_uploaded_file != file_id or not st.session_state.pdf_indexed:
        
        reset_rag_state()  # ✅ This now properly clears chapters
        st.session_state.last_uploaded_file = file_id
        
        # -------------------------------------------------
        # 🔹 IMAGE PROCESSING (FIXED)
        # -------------------------------------------------
        if uploaded_file.type.startswith("image"):
            st.session_state.current_file_type = "image"
            st.session_state.upload_status = "⏳ Processing image..."
            
            # Always convert to base64 for vision
            st.session_state.uploaded_image_base64 = image_to_base64(uploaded_file)
            
            # Try OCR
            image_text = extract_uploaded_image_text(uploaded_file)
            
            if image_text.strip() and len(image_text.strip()) > 20:
                # OCR successful - create both vision and text index
                image_doc = Document(
                    page_content=f"[UPLOADED IMAGE OCR]\n{image_text}",
                    metadata={"page": 1, "type": "image"}
                )
                
                st.session_state.vectorstore = FAISS.from_documents([image_doc], embeddings)
                st.session_state.bm25 = BM25Okapi([tokenize(image_doc.page_content)])
                st.session_state.all_chunks = [image_doc]
                st.session_state.pdf_indexed = True
                st.session_state.vision_only = False  # Can use both OCR and vision
                st.session_state.upload_status = "✅ Image ready (OCR + Vision mode)"
                
            else:
                # OCR failed or insufficient text - use vision only
                st.session_state.pdf_indexed = True
                st.session_state.vision_only = True
                st.session_state.upload_status = "✅ Image ready (Vision mode only)"
        
        # -------------------------------------------------
        # 🔹 OPTIMIZED PDF PROCESSING WITH PROGRESS BAR
        # -------------------------------------------------
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
            
            status_text.text("📚 Extracting table of contents...")
            progress_bar.progress(20)
            titles = extract_toc_at_ingestion(pages)
            if not titles:
                titles = fallback_extract_chapters_regex(pages)
            st.session_state.chapter_titles = titles
            
            status_text.text("✂️ Splitting text into chunks...")
            progress_bar.progress(40)
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=200)
            chunks = splitter.split_documents(pages)
            
            status_text.text("🖼️ Extracting images (first 5 pages)...")
            progress_bar.progress(60)
            for img in extract_images_as_text(pdf_path, max_pages=5):
                chunks.append(
                    Document(
                        page_content=f"[IMAGE CONTENT]\n{img['text']}",
                        metadata={"page": img["page"], "type": "image"}
                    )
                )
            
            status_text.text("📊 Extracting tables (first 10 pages)...")
            progress_bar.progress(70)
            for tbl in extract_tables_as_text(pdf_path, max_pages=10):
                chunks.append(
                    Document(
                        page_content=f"[TABLE CONTENT]\n{tbl['text']}",
                        metadata={"page": tbl["page"], "type": "table"}
                    )
                )
            
            os.remove(pdf_path)
            
            status_text.text("🔍 Creating vector index...")
            progress_bar.progress(85)
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            
            status_text.text("📇 Creating keyword index...")
            progress_bar.progress(95)
            st.session_state.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])
            st.session_state.all_chunks = chunks
            
            st.session_state.pdf_indexed = True
            st.session_state.vision_only = False
            
            progress_bar.progress(100)
            status_text.text("✅ PDF indexed successfully!")
            st.session_state.upload_status = "✅ PDF indexed and ready"
            
            import time
            time.sleep(2)
            progress_bar.empty()
            status_text.empty()

# -------------------------------------------------
# Hybrid Retrieval (UNCHANGED)
# -------------------------------------------------
def hybrid_retrieve(query):
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
# ✅ ENHANCED: RAG Answer Logic with Web Search
# -------------------------------------------------
def rag_answer(question):

    qkey = question.lower().strip()

    if (
        qkey in st.session_state.answer_cache
        and not is_chapter_titles_request(question)
        and not is_summary_request(question)
    ):
        return st.session_state.answer_cache[qkey]

    if is_chapter_titles_request(question):
        if st.session_state.chapter_titles:
            return "Chapters in the document:\n" + "\n".join(st.session_state.chapter_titles)
        else:
            return "❌ No chapters found in the document."

    # ✅ Check if it's a science question
    is_sci_q = is_science_question(question)
    
    docs = hybrid_retrieve(question)
    
    # ✅ NEW: If no docs found and it's a science question, try web search
    if not docs:
        if is_sci_q and st.session_state.web_search_enabled:
            web_results = web_search_science(question)
            if web_results:
                web_answer = llm_text(answer_llm.invoke(
                    f"""Answer this science question using web search results:

Web Results:
{web_results}

Question: {question}

Provide a clear, educational answer."""
                ))
                return f"🌐 **Web Search Result** (not found in document):\n\n{web_answer}"
        
        return "❌ Not found in the document."

    context = "\n\n".join(d.page_content for d in docs)

    # 🔹 APPENDED: visual-aware answering
    if is_visual_question(question):
        visual_context = "\n\n".join(
            d.page_content for d in docs
            if d.metadata.get("type") in ["image", "table"]
        )
        if visual_context.strip():
            return llm_text(answer_llm.invoke(
                f"Answer ONLY using this visual content:\n{visual_context}\nQuestion:{question}"
            ))

    # Try to answer from document
    answer = llm_text(answer_llm.invoke(
        f"""Answer the question using ONLY the provided context.
If the context does not contain enough information, respond with: "The provided context does not contain sufficient information about [topic]."

Context:
{context}

Question: {question}"""
    ))

    # ✅ NEW: If answer is insufficient and it's a science question, try web search
    if is_sci_q and st.session_state.web_search_enabled and is_insufficient_answer(answer):
        web_results = web_search_science(question)
        if web_results:
            web_answer = llm_text(answer_llm.invoke(
                f"""The document has insufficient information. Use web search results to answer:

Web Results:
{web_results}

Question: {question}

Provide a clear, comprehensive answer."""
            ))
            return f"🌐 **Web Search Result** (insufficient info in document):\n\n{web_answer}"

    st.session_state.answer_cache[qkey] = answer
    return answer

# -------------------------------------------------
# ✅ FIXED: UI with Anti-Hallucination Protection
# -------------------------------------------------
if st.session_state.pdf_indexed:
    for role, msg in st.session_state.messages:
        with st.chat_message(role):
            st.write(msg)
    
    user_q = st.chat_input("Ask a question")

    if user_q:
        st.session_state.messages.append(("user", user_q))

        # ✅ FIXED: Proper routing based on file type AND question type
        if st.session_state.current_file_type == "image" and st.session_state.uploaded_image_base64:
            # For images: Check type of question
            
            # 1. Check if asking for chapters/summaries
            if is_chapter_titles_request(user_q) or (is_summary_request(user_q) and extract_chapter_name_from_query(user_q)):
                answer = "❌ You uploaded an image, not a PDF. Chapter summaries and document structure queries are only available for PDF documents. Please upload a PDF if you want to query chapters."
            
            # 2. Check if question is about the image itself
            elif is_image_specific_question(user_q):
                # Use vision model for image analysis questions
                try:
                    vision_answer = answer_llm.invoke(
                        [
                            HumanMessage(
                                content=[
                                    {"type": "text", "text": f"Answer this question STRICTLY based on what you see in the image. If the information is not visible in the image, say 'I cannot answer this from the image alone.' Question: {user_q}"},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{st.session_state.uploaded_image_base64}"
                                        }
                                    }
                                ]
                            )
                        ]
                    )
                    answer = llm_text(vision_answer)
                except Exception as e:
                    answer = f"❌ Error processing image: {str(e)}"
            
            # 3. General knowledge question - check if science and web search enabled
            else:
                # This is a general knowledge question, not about the image
                if is_science_question(user_q):
                    if st.session_state.web_search_enabled:
                        # Use web search for science questions
                        web_results = web_search_science(user_q)
                        if web_results:
                            web_answer = llm_text(answer_llm.invoke(
                                f"""Answer this science question using web search results:

Web Results:
{web_results}

Question: {user_q}

Provide a clear, educational answer."""
                            ))
                            answer = f"🌐 **Web Search Result**:\n\n{web_answer}"
                        else:
                            answer = "❌ Web search failed. Please try again or rephrase your question."
                    else:
                        answer = f"❌ Your question '{user_q}' appears to be a general science question, not about the uploaded image.\n\n**Options:**\n1. Enable 'Web Search' in the sidebar to get answers from the internet\n2. Ask a question about what's shown in the image (e.g., 'What is shown in this image?')\n3. Upload a relevant PDF document"
                else:
                    # Not a science question and not about image
                    answer = f"❌ I can only:\n1. Answer questions about the uploaded image (try: 'Describe this image')\n2. Answer science questions if web search is enabled\n\nYour question doesn't appear to reference the image. Please rephrase or enable web search."
        
        elif st.session_state.current_file_type == "pdf":
            # Use RAG for PDFs
            answer = rag_answer(user_q)
        
        else:
            answer = "❌ Please upload a PDF or image first."

        st.session_state.messages.append(("assistant", answer))
        st.rerun()

else:
    st.info("⬆️ Upload a PDF or Image to begin")