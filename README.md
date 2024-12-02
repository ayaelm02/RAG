# Streamlit RAG Application

This repository contains a Streamlit-based Retrieval-Augmented Generation (RAG) application that integrates Ollama's Llama 3.1 model with open-source models such as Flan-T5, MiniLM, and Falcon for answering user queries based on uploaded documents.

## Features
- **Dynamic Model Selection**: Choose between Ollama Llama 3.1, Flan-T5, MiniLM, and Falcon models.
- **Document Uploads**: Add PDF files to create a searchable knowledge base.
- **Retrieval-Augmented Generation**: Combine document retrieval with powerful language models to answer questions accurately.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/streamlit-rag-application.git
   cd streamlit-rag-application
   
2. Install dependencies:
pip install -r requirements.txt

3.Create a .env file with the following content:
OLLAMA_BASE_URL=http://127.0.0.1:11434

4.Start the application:
streamlit run app.py

## Usage
1. Select a model from the sidebar.
1. Upload PDFs to build a document knowledge base.
1. Enter your question in the input box, and click Get Answer.
## Supported Models
1. Ollama Llama 3.1
2. Flan-T5 (Small)
3. MiniLM (Embeddings)
