import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoTokenizer, pipeline
import torch
import os
import uuid
import numpy as np
import chromadb

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = [("Hello, I am a Chatbot, how may I help you?", "bot")]

# Set up the ChromaDB collection path
CHROMA_PATH = "chroma_db"
if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)

# Load model for generating embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize ChromaDB client and collection
client = chromadb.Client()
collection = client.get_or_create_collection(name="document_embeddings")

# Function to generate embeddings
def generate_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Define chat layout styling
st.markdown(
    """
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 20px;
    }
    .message {
        display: flex;
        align-items: center;
        padding: 10px;
        border-radius: 10px;
        max-width: 80%;
    }
    .bot-message {
        background-color: #f1f1f1;
        color: black;
        justify-content: flex-start;
    }
    .user-message {
        background-color: #d1e7dd;
        color: black;
        justify-content: flex-end;
        align-self: flex-end;
    }
    .avatar {
        width: 30px;
        height: 30px;
        margin-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display chat history
st.write('<div class="chat-container">', unsafe_allow_html=True)
for msg, sender in st.session_state.history:
    if sender == "bot":
        st.markdown(
            f"""
            <div class="message bot-message">
                <img src="https://via.placeholder.com/30?text=🤖" class="avatar" />
                <div>{msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="message user-message">
                <div>{msg}</div>
                <img src="https://via.placeholder.com/30?text=🙂" class="avatar" />
            </div>
            """,
            unsafe_allow_html=True,
        )
st.write('</div>', unsafe_allow_html=True)

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Load and process PDFs into ChromaDB if uploaded
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load the PDF using PyMuPDFLoader
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)

        # Remove the temporary file after loading
        os.remove(temp_path)

    st.write(f"Loaded {len(documents)} document(s) from PDF(s).")

    # Split documents into chunks and generate embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = text_splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = generate_embeddings(texts)
    metadatas = [{"id": str(uuid.uuid4())} for _ in chunks]

    # Add documents to ChromaDB
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=[str(uuid.uuid4()) for _ in texts]
    )
    st.write("Documents have been embedded and added to the vector store.")

# Question-answering pipeline setup
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Input box for user query with a temporary variable to avoid direct modification
if "input_query_temp" not in st.session_state:
    st.session_state.input_query_temp = ""  # Initialize temporary input query variable

user_input = st.text_input("Message:", value=st.session_state.input_query_temp, key="input_query")

# Process the user input
if st.button("Send"):
    if user_input:
        # Append user question to chat history
        st.session_state.history.append((user_input, "user"))

        # Retrieve relevant context from ChromaDB
        query_embedding = generate_embeddings([user_input])[0]
        results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=5)
        context = "\n\n".join([doc for sublist in results["documents"] for doc in sublist])

        # Get answer from the question-answering model
        answer = qa_pipeline({
            'question': user_input,
            'context': context
        })["answer"]

        # Append bot answer to chat history
        st.session_state.history.append((answer, "bot"))

        # Reset the temporary variable to clear the input field indirectly
        st.session_state.input_query_temp = ""
