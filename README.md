# 📘 HR Policy Assistant

An AI-powered HR Policy Assistant built with Retrieval-Augmented Generation (RAG).

Users can upload an HR Policy PDF and ask questions about company policies.

The application retrieves relevant sections from the uploaded document using
FAISS and Sentence Transformers, then uses Groq's
`openai/gpt-oss-20b` model to generate an answer.

---

## 🚀 Features

- Upload HR Policy PDF
- Extract PDF text using PyMuPDF
- Split policy into overlapping text chunks
- Generate embeddings using Sentence Transformers
- Store embeddings in FAISS
- Retrieve the most relevant policy sections
- Generate answers using Groq GPT-OSS 20B
- Page-based source references
- Streamlit chat interface
- No database required
- No PDF needs to be stored in GitHub

---

## 🧠 RAG Architecture

```text
HR Policy PDF
      ↓
   PyMuPDF
      ↓
 Text Extraction
      ↓
 Text Chunking
      ↓
Sentence Transformers
      ↓
 Vector Embeddings
      ↓
    FAISS
      ↓
Top-K Relevant Chunks
      ↓
   Groq API
      ↓
GPT-OSS 20B
      ↓
 HR Policy Answer
