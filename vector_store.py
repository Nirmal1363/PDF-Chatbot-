from langchain_community.vectorstores import Chroma
from embeddings import get_embedding_function
import uuid 

# Set the path for ChromaDB persistence
CHROMA_PATH = "chroma"

def initialize_vector_store():
    # Initialize Chroma vector store with persistence
    db = Chroma(
        collection_name="document_embeddings",
        embedding_function=get_embedding_function(),
        persist_directory=CHROMA_PATH
    )
    return db

def add_documents_to_store(documents):
    db = initialize_vector_store()
    embeddings_model = get_embedding_function()

    # Validate and flatten `documents` if needed
    valid_documents = []
    for doc in documents:
        if hasattr(doc, 'page_content'):
            valid_documents.append(doc)
        elif isinstance(doc, list):  # Handle any nested lists
            for sub_doc in doc:
                if hasattr(sub_doc, 'page_content'):
                    valid_documents.append(sub_doc)

    # Generate embeddings for each document's content
    embeddings = []
    metadatas = []
    ids = []
    
    for doc in valid_documents:
        embedding = embeddings_model.embed_documents([doc.page_content])[0]
        embeddings.append(embedding)
        metadatas.append({"source": doc.metadata.get("source")})
        ids.append(doc.metadata.get("id", str(uuid.uuid4())))

    # Add documents to the vector store
    db.add_documents(documents=embeddings, metadatas=metadatas, ids=ids)
    db.persist()
    print(f"Added {len(embeddings)} documents to the vector store.")