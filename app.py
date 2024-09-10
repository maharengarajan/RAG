import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# App layout and title
st.set_page_config(page_title="RAG App", layout="centered")
st.title("Retrieval-Augmented Generation (RAG) Application")

# Define directory to save uploaded PDFs
upload_dir = "uploaded_pdfs"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Sidebar for API Key input
st.sidebar.header("API Key")
api_key = st.sidebar.text_input("Enter your API Key", type="password")

# LLM Initialization after API key input
if api_key:
    try:
        llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192")
        st.sidebar.success("API Key entered successfully!")
    except Exception as e:
        st.sidebar.error(f"Error initializing LLM: {e}")
else:
    st.sidebar.warning("Please enter your API Key to proceed.")

# Chat Interface and Session Management
session_id = st.text_input("Session ID", value="default_session")

if 'store' not in st.session_state:
    st.session_state.store = {}

# File Upload Section
st.header("Upload PDF Files")
uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files and api_key:
    documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Process the PDF file using PyPDFLoader
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                st.success(f"File '{uploaded_file.name}' processed successfully!")
            except Exception as e:
                st.error(f"Error processing file '{uploaded_file.name}': {e}")

    # Split the documents and create embeddings
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Contextualize Questions based on Chat History
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer question
        system_prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User input for questions
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            try:
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
                st.write("Assistant:", response['answer'])
                st.write("Chat History:", session_history.messages)
            except Exception as e:
                st.error(f"Error during Q&A processing: {e}")

# Display footer for additional info
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Renga Rajan K")
st.sidebar.markdown("Powered by Streamlit")
