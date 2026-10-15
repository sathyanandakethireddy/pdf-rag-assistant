
cat > README.md <<'EOF'
# PDF RAG Assistant

A PDF question-answering application built using Python and Streamlit.
It uses Retrieval-Augmented Generation (RAG) to answer questions
based on uploaded PDF documents.

## Features

- Upload multiple PDF files
- Extract text from PDFs using PyPDF
- Split text into chunks using LangChain
- Generate embeddings using Sentence Transformers
- Store and retrieve chunks using ChromaDB
- Generate answers using the Google Gemini API
- Display retrieved document sources and page numbers

## Tech Stack

- Python
- Streamlit
- PyPDF
- LangChain Text Splitters
- Sentence Transformers
- ChromaDB
- Google Gemini API

## Setup

### 1. Clone the repository

Clone this repository from GitHub and enter its directory.

### 2. Create a virtual environment

    python3 -m venv .venv
    source .venv/bin/activate

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Configure the API key

Create a `.env` file in the project root:

    GEMINI_API_KEY=your_gemini_api_key

Replace the placeholder with your own Gemini API key.

### 5. Run the application

    streamlit run file.py

## Security

Do not commit API keys, private PDFs, virtual environments,
or local database files.
EOF

