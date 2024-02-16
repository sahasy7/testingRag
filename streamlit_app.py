import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
import os 
os.environ['OPENAI_API_KEY'] =st.secrets.openai_key

def load_data():
        loader = TextLoader("data/GSM Mall Update Q&A.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        print("data is loaded")
        return db

vectore_store = load_data()
dr = vectore_store.as_retriever()
if dr :
        st.title("Sheraton-Bot")
else:
        print("something went wrong")
