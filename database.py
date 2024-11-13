import os
import chromadb
from langchain.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_store import add_documents_to_store
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def initialize_vector_store():
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    client = chromadb.Client()
    return client

def load_documents(uploaded_files):
    all_chunks = []

    for uploaded_file in uploaded_files:
        pdf_loader = PyMuPDFLoader(uploaded_file.name)
        documents = pdf_loader.load()

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        chunks = text_splitter.split_documents(documents)

        # Flatten any nested lists in chunks
        for chunk in chunks:
            if isinstance(chunk, list):
                all_chunks.extend(chunk)
            else:
                all_chunks.append(chunk)

    # Ensure all_chunks contains only individual document objects
    add_documents_to_store(all_chunks)
    print("All documents have been processed and added to the vector store.")