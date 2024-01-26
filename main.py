import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


st.title("Betterzilla Assignment")
st.header("by Kaustubh Kamble")
user_question = st.text_input("Ask a question about the document and then click on Process")

with st.sidebar:
    st.write("1. ChromaDB for vector store")
    st.write("2. Embedding from Hugging Face Hub")
    st.write("3. LLM from Hugging Face Hub (Google's FLAN T5)")


if st.button("Process"):
    with st.spinner("Processing..."):
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory='chroma', embedding_function=embedding_function)

        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl")
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        response = conversation_chain.invoke(
            {'question' : user_question}
        )
        st.write(response['answer'])