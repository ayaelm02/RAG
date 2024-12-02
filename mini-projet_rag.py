import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Persistent directory for Chroma DB
persist_directory = "./rag_env/pdf_chroma_db"

# Streamlit UI
st.title("Streamlit RAG Application")
st.sidebar.title("Settings")

# LLM Options
llm_options = {
    "Ollama Llama 3.1": {
        "class": Ollama,
        "model": "llama3.1",
        "base_url": "http://127.0.0.1:11434",
        "embedding_class": OllamaEmbeddings
    },
    "OpenAI GPT-3.5-Turbo": {
        "class": OpenAI,
        "model": "gpt-3.5-turbo",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "embedding_class": OpenAIEmbeddings
    }
}

# Select LLM
selected_llm = st.sidebar.selectbox("Select LLM", options=llm_options.keys())
llm_config = llm_options[selected_llm]

# Initialize LLM and embeddings
llm = llm_config["class"](**{k: v for k, v in llm_config.items() if k not in ["class", "embedding_class"]})
embed_model = llm_config["embedding_class"](**{k: v for k, v in llm_config.items() if k not in ["class", "embedding_class"]})

# Initialize or load Chroma vector store
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embed_model)

# Define retriever and retrieval chain
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
retrieval_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Document upload
st.sidebar.subheader("Add Your Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        vector_store.add_documents(chunks)
        vector_store.persist()
    st.sidebar.success("Documents added successfully!")

# Chat interface
st.subheader("Chat with RAG")
question = st.text_input("Enter your question here")

if st.button("Get Answer"):
    if question:
        with st.spinner("Generating response..."):
            response = retrieval_chain.run(question)
            st.write("**Response:**", response)
    else:
        st.error("Please enter a question.")
