from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    # Use a Hugging Face embedding model that doesn’t require a local server
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
