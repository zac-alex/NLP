# 📄 PDF-based Chatbot using NLP & Vector Embeddings

This project implements an end-to-end intelligent chatbot that can read, understand, and answer queries from a PDF document using NLP techniques and vector embeddings. It leverages LangChain, HuggingFace models, ChromaDB, and Transformers.

---

## 🚀 Features

- 📄 Load and parse PDF documents
- ✂️ Split text into meaningful overlapping chunks
- 🔎 Store and retrieve embeddings using **ChromaDB**
- 🤖 Semantic search using **Cosine Similarity**
- 💬 Answer generation using **GPT-Neo 1.3B** model from HuggingFace
- 📚 Context-aware query resolution
- 🔁 Interactive Q&A session from PDF content

---

## 📦 Requirements

Make sure to install the necessary packages before running the code:

```bash
pip install langchain
pip install langchain_community
pip install chromadb
pip install transformers
pip install tqdm
Note: You may also need to install other dependencies like pypdf or sentence-transformers based on your environment.

🧠 Model & Tools Used
PDF Parsing: PyPDFLoader
Text Splitting: RecursiveCharacterTextSplitter
Embeddings: HuggingFaceEmbeddings (all-MiniLM-L6-v2)
Vector Store: ChromaDB
Retriever Logic: Cosine Similarity
Language Model: EleutherAI/gpt-neo-1.3B
Text Generation: HuggingFace pipeline("text-generation")
🔍 How It Works
Load PDF: Extract pages as documents using LangChain’s PyPDFLoader.
Split Text: Break down text into overlapping chunks for better context preservation.
Vector Embedding: Convert text chunks into vectors using MiniLM and store in ChromaDB.
Query Retrieval: Use cosine similarity to retrieve the most relevant chunks for a given user query.
Answer Generation: Use GPT-Neo to generate context-aware responses based on retrieved data.
📂 File Structure
bash
Copy
Edit
project/
├── pdf_chatbot.py
├── db/                    # Vector store (ChromaDB)
└── README.md
🧪 Example Use Case
bash
Copy
Edit
Ask any question from the pdf (or type 'exit'): What is the main topic of the document?
→ The model retrieves the most relevant context and provides an intelligent answer from the document.

📌 Important Notes
Ensure your PDF path is correctly set in pdf_path.
Vector DB is persisted locally in a folder (db/) to avoid re-computation.
You can modify the GPT model or embedding model as needed.
🙌 Contribution
Feel free to fork, improve or raise issues in this repo. Contributions are always welcome!

📜 License
This project is licensed under the MIT License.

👤 Author
Developed by [Your Name Here]