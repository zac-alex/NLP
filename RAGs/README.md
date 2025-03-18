# 📄 PDF-based Chatbot using Retrieval Augmented Generation (RAG)

This project builds an intelligent chatbot capable of understanding and answering questions from PDF documents using advanced NLP techniques, vector embeddings, and large language models.

---

## Features

- PDF parsing and text extraction
- Overlapping text chunking for contextual continuity
- Embedding generation with HuggingFace models
- Vector storage and retrieval via **ChromaDB**
- Semantic similarity search using **Cosine Similarity**
- Context-aware response generation using **GPT-Neo 1.3B**
- Interactive Q&A interface

---

## Requirements

Install the required dependencies:

```bash
pip install langchain
pip install langchain_community
pip install chromadb
pip install transformers
pip install tqdm
Note: You may also need to install pypdf, sentence-transformers, or additional dependencies based on your environment.

Architecture Overview
PDF Parsing: PyPDFLoader
Text Splitting: RecursiveCharacterTextSplitter
Embeddings: HuggingFaceEmbeddings (all-MiniLM-L6-v2)
Vector Store: ChromaDB
Retrieval Logic: Cosine Similarity
Language Model: EleutherAI/gpt-neo-1.3B
Answer Generation: HuggingFace pipeline("text-generation")
Workflow
Load and parse PDF documents
Split content into overlapping chunks
Generate embeddings and store them in ChromaDB
Retrieve relevant chunks using vector similarity search
Generate context-aware answers using GPT-Neo
Project Structure
arduino
Copy
Edit
project/
├── pdf_chatbot.py
├── db/                   # ChromaDB vector storage
└── README.md
Example Usage
Ask any question from the pdf (or type 'exit'): What is the main topic of the document?
→ The model fetches relevant content and generates an intelligent response.
Notes
Ensure correct PDF path is provided in the pdf_path variable.
The vector database is persisted in the db/ directory for reuse.
You can modify the model or embeddings according to your project needs.


Author
Developed by Zachariah Alex