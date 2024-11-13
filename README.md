# PDF Chatbot

This project is a PDF-based chatbot that allows users to upload PDF documents and ask questions about the content. The chatbot retrieves relevant information from the uploaded PDFs and answers questions interactively using a conversational interface built with Streamlit.

## Features

- **PDF Upload**: Users can upload multiple PDF files, which are then processed and stored.
- **Document Embedding with ChromaDB**: The chatbot uses ChromaDB to store and retrieve document embeddings for efficient similarity search.
- **Question-Answering with Transformer Models**: Uses a question-answering model from Hugging Face's Transformers library to answer questions based on context retrieved from the PDF content.
- **Interactive Chat Interface**: Provides a user-friendly chat interface for interactive Q&A, with styled message bubbles and avatars for both the user and chatbot.

## Demo

![Demo Screenshot](demo_screenshot.png) <!-- Replace with actual path to your screenshot -->

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/PDF-Chatbot.git
   cd PDF-Chatbot
