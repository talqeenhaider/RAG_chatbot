# 📚 PDF RAG Chatbot

An AI-powered **PDF Question Answering Chatbot** built using **Python, Flask, LangChain, HuggingFace Embeddings, ChromaDB, and OpenRouter**.

The system allows users to upload a PDF and ask questions. It retrieves relevant information from the document and uses **GPT-4o-mini** to generate an answer based on the retrieved context.

## 🚀 Features

* 📄 Upload PDF documents
* ✂️ Split documents into chunks
* 🧠 Generate HuggingFace embeddings
* 🗄️ Store embeddings in ChromaDB
* 🔍 Retrieve relevant information using MultiQueryRetriever
* 🤖 Generate answers using GPT-4o-mini
* 📑 Show source page numbers
* 🌐 Flask-based web API

## 🔄 How It Works

```text
PDF Upload
    ↓
PDF Loader
    ↓
Text Chunking
    ↓
HuggingFace Embeddings
    ↓
ChromaDB
    ↓
MultiQueryRetriever
    ↓
Relevant Context
    ↓
GPT-4o-mini
    ↓
Answer + Source Pages
```

## 📂 Project Structure

```text
PDF-RAG-Chatbot/
│
├── app.py
├── rag.py
├── templates/
│   └── index.html
├── uploads/
├── docs/
│   └── chroma/
├── .env
└── README.md
```

## 🛠️ Technologies

* Python
* Flask
* LangChain
* HuggingFace
* Sentence Transformers
* ChromaDB
* OpenRouter
* GPT-4o-mini

## ⚙️ Setup

```bash
git clone YOUR_REPOSITORY_URL
cd PDF-RAG-Chatbot

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
SECRET_KEY=your_openrouter_api_key
```

Run the application:

```bash
python app.py
```

Then open the local Flask URL in your browser.

## 🎯 Purpose

This project demonstrates how **RAG (Retrieval-Augmented Generation)** can be used to build a document-based AI assistant that answers questions from uploaded PDF documents.
