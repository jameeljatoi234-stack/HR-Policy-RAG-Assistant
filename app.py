import os
import hashlib
from typing import List, Dict, Tuple

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

TOP_K = 5
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        .source-box {
            padding: 12px;
            border-radius: 8px;
            background-color: #f5f5f5;
            margin-top: 10px;
            font-size: 0.9rem;
        }

        .answer-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f8f9fa;
            border-left: 5px solid #4f46e5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    """
    Load the Sentence Transformer model once and cache it.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():
    """
    Create Groq client using Streamlit secrets.

    For Streamlit Cloud:
        GROQ_API_KEY = "your-key"

    Environment variable fallback is included for flexibility.
    """

    api_key = None

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_bytes: bytes) -> List[Dict]:
    """
    Extract text from each PDF page.

    Returns:
        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]
    """

    pages = []

    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_number, page in enumerate(pdf_document, start=1):
        text = page.get_text("text")

        if text and text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    pdf_document.close()

    return pages


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(
    pages: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Split extracted PDF text into overlapping chunks.

    Each chunk keeps its original page number.
    """

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        # Normalize whitespace
        text = " ".join(text.split())

        start = 0

        while start < len(text):

            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(
                    {
                        "text": chunk_text.strip(),
                        "page": page_number,
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(
    chunks: List[Dict],
    embedding_model,
):
    """
    Generate embeddings and create a FAISS similarity index.

    We normalize embeddings and use inner product similarity,
    which is equivalent to cosine similarity for normalized vectors.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# SEARCH DOCUMENT
# ============================================================

def search_document(
    query: str,
    index,
    chunks: List[Dict],
    embedding_model,
    top_k: int = TOP_K,
) -> List[Dict]:
    """
    Retrieve the most relevant chunks from FAISS.
    """

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = query_embedding.astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        result = chunks[idx].copy()
        result["score"] = float(score)

        results.append(result)

    return results


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(
    question: str,
    search_results: List[Dict],
    groq_client: Groq,
) -> str:
    """
    Generate an answer using only retrieved HR policy context.
    """

    context_parts = []

    for i, result in enumerate(search_results, start=1):

        context_parts.append(
            f"""
SOURCE {i}
PAGE: {result['page']}
RELEVANCE SCORE: {result['score']:.4f}

CONTENT:
{result['text']}
"""
        )

    context = "\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer questions ONLY using the HR policy
information provided in the context.

Rules:

1. Do not invent HR policies.
2. Do not make assumptions that are not supported by the document.
3. If the answer cannot be found in the provided context,
   clearly say:

   "I could not find this information in the uploaded HR policy."

4. Cite the relevant page number whenever possible.
5. If multiple policy sections are relevant, mention them.
6. Keep the answer professional, clear, and easy to understand.
7. Do not provide legal advice.
8. If the policy appears ambiguous, explain the ambiguity rather
   than inventing an interpretation.

Answer format:

- Give the direct answer first.
- Add a short explanation if necessary.
- End with "Source: Page X" or "Sources: Pages X, Y".
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

QUESTION:

{question}

Answer the question using only the HR policy context above.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=1000,
        include_reasoning=False,
    )

    return response.choices[0].message.content


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_document(pdf_bytes: bytes):

    with st.spinner("Reading and processing the HR policy..."):

        pages = extract_pdf_text(pdf_bytes)

        if not pages:
            raise ValueError(
                "No readable text was found in this PDF. "
                "If this is a scanned PDF, OCR may be required."
            )

        chunks = create_chunks(pages)

        if not chunks:
            raise ValueError(
                "No text chunks could be created from the PDF."
            )

        embedding_model = load_embedding_model()

        index = create_faiss_index(
            chunks,
            embedding_model,
        )

    return pages, chunks, index


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📘 HR Policy Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Upload an HR Policy PDF and ask questions about company policies,
    leave, attendance, benefits, workplace rules, and more.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        """
        **RAG Pipeline**

        📄 PDF  
        ↓  
        🔎 PyMuPDF  
        ↓  
        ✂️ Text Chunking  
        ↓  
        🧠 Sentence Transformers  
        ↓  
        📊 FAISS  
        ↓  
        🤖 Groq GPT-OSS 20B
        """
    )

    st.divider()

    st.info(
        "Upload a PDF using the uploader in the main area."
    )


# ============================================================
# API KEY CHECK
# ============================================================

groq_client = get_groq_client()

if not groq_client:

    st.warning(
        "⚠️ Groq API key is not configured."
    )

    st.info(
        """
        For Streamlit Cloud, add this secret:

        GROQ_API_KEY = "your_groq_api_key"
        """
    )

    st.stop()


# ============================================================
# PDF UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload your HR Policy PDF",
    type=["pdf"],
    help="Upload the HR policy document you want to ask questions about.",
)


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    file_hash = hashlib.md5(pdf_bytes).hexdigest()

    # Process only if a new PDF has been uploaded
    if st.session_state.get("file_hash") != file_hash:

        try:

            pages, chunks, index = process_document(
                pdf_bytes
            )

            st.session_state["file_hash"] = file_hash
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["pages"] = pages
            st.session_state["chunks"] = chunks
            st.session_state["index"] = index

            st.session_state["messages"] = []

            st.success(
                f"✅ {uploaded_file.name} processed successfully."
            )

        except Exception as e:

            st.error(
                f"Could not process the PDF: {str(e)}"
            )

            st.stop()

    else:

        st.success(
            f"✅ {uploaded_file.name} is ready."
        )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if "chunks" in st.session_state:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "PDF Pages",
            len(st.session_state["pages"]),
        )

    with col2:
        st.metric(
            "Text Chunks",
            len(st.session_state["chunks"]),
        )

    with col3:
        st.metric(
            "Retrieval Top-K",
            TOP_K,
        )

    st.divider()


# ============================================================
# CHAT INTERFACE
# ============================================================

if "chunks" not in st.session_state:

    st.info(
        "👆 Upload an HR Policy PDF to start asking questions."
    )

else:

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous messages
    for message in st.session_state["messages"]:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander("📚 Retrieved policy sections"):

                    for source in message["sources"]:

                        st.markdown(
                            f"""
                            **Page {source['page']}**
                            
                            Relevance: `{source['score']:.3f}`
                            
                            {source['text']}
                            """
                        )

    question = st.chat_input(
        "Ask something about the HR policy..."
    )

    if question:

        # User message
        st.session_state["messages"].append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve relevant chunks
        embedding_model = load_embedding_model()

        search_results = search_document(
            question,
            st.session_state["index"],
            st.session_state["chunks"],
            embedding_model,
            TOP_K,
        )

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Searching the HR policy..."):

                try:

                    answer = generate_answer(
                        question,
                        search_results,
                        groq_client,
                    )

                    st.markdown(answer)

                    with st.expander(
                        "📚 Retrieved policy sections"
                    ):

                        for source in search_results:

                            st.markdown(
                                f"""
                                **Page {source['page']}**
                                
                                Relevance: `{source['score']:.3f}`
                                
                                {source['text']}
                                """
                            )

                    # Save assistant message
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": search_results,
                        }
                    )

                except Exception as e:

                    error_message = (
                        "Sorry, I couldn't generate an answer. "
                        f"Error: {str(e)}"
                    )

                    st.error(error_message)

                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                        }
                    )
