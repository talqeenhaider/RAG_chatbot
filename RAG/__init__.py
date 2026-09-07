from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil



def process_pdf(pdf_path):

    # ==========================================
    # 1. Load uploaded PDF
    # ==========================================

    loader = PyPDFLoader(pdf_path)

    pages = loader.load()

    print(f"Loaded pages: {len(pages)}")


   
    # ==========================================
    # 2. Split PDF into chunks
    # ==========================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    docs = text_splitter.split_documents(pages)

    print(f"Created chunks: {len(docs)}")


    # ==========================================
    # 3. Create embeddings
    # ==========================================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )



    # ==========================================
    # 4. Chroma database location
    # ==========================================

    persist_directory = "docs/chroma"


    # ==========================================
    # 5. Delete old database
    # ==========================================

    if os.path.exists(persist_directory):

        shutil.rmtree(persist_directory)

        print("Old ChromaDB deleted.")

    # ==========================================
    # 6. Create new Chroma database
    # ==========================================

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print("New ChromaDB created successfully.")
    print("Database location:", persist_directory)


   
    # ==========================================
    # 7. Return vector database
    # ==========================================

    return vectordb



