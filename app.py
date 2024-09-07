import os
import sqlite3
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import asyncio

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Database setup
def create_users_table():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)"""
    )
    conn.commit()
    conn.close()


def add_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)", (username, password)
    )
    conn.commit()
    conn.close()


def validate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?", (username, password)
    )
    user = c.fetchone()
    conn.close()
    return user


# Function to get text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create a vector store from text chunks
def create_vector_store(text_chunks):
    embeddings = FastEmbedEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("Faiss")


# Function to load and configure the conversational chain
def get_conversational_chain():
    prompt_template = """
        You are TaxWise AI, a highly experienced accountant providing tax advice based on Indian Tax laws.
        You will respond to the user's queries by leveraging your accounting and tax expertise and the Context Provided.
        Context: {context}
        Question: {question}
        Answer:
    """

    # Ensure there is an event loop in the current thread
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        system_instruction="You are TaxWise AI, a highly experienced assistant providing tax advice based on Indian laws and Tax Regulations.",
    )

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = FastEmbedEmbeddings()

    if os.path.exists("Faiss"):
        new_db = FAISS.load_local(
            "Faiss", embeddings, allow_dangerous_deserialization=True
        )
    else:
        pdf_files = [
            os.path.join("dataset", file)
            for file in os.listdir("dataset")
            if file.endswith(".pdf")
        ]
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        create_vector_store(text_chunks)
        new_db = FAISS.load_local(
            "Faiss", embeddings, allow_dangerous_deserialization=True
        )

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    return response["output_text"]


# Streamlit app interface
def main():
    st.set_page_config("TaxWise AI", page_icon=":scales:", layout="centered")
    st.header("TaxWise AI: AI Tax Advisor :scales:")

    # Create users table if not exists
    create_users_table()

    # Login and Signup functionality
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

        with login_tab:
            st.subheader("Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                user = validate_user(username, password)
                if user:
                    st.session_state.authenticated = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with signup_tab:
            st.subheader("Sign Up")
            new_username = st.text_input("New Username", key="signup_username")
            new_password = st.text_input(
                "New Password", type="password", key="signup_password"
            )
            if st.button("Sign Up"):
                if new_username and new_password:
                    try:
                        add_user(new_username, new_password)
                        st.success("Signup successful! Please login.")
                    except sqlite3.IntegrityError:
                        st.error("Username already exists. Try a different one.")
                else:
                    st.error("Please fill in both fields.")

    if st.session_state.authenticated:

        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hi I'm TaxWise AI, an AI Tax Saving Assistant.",
                }
            ]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        prompt = st.chat_input("Type your question here...")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = user_input(prompt)
                        st.write(response)

                if response is not None:
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)


def prepare_data():
    if not os.path.exists("Faiss"):
        pdf_files = [
            os.path.join("dataset", file)
            for file in os.listdir("dataset")
            if file.endswith(".pdf")
        ]
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        create_vector_store(text_chunks)


if __name__ == "__main__":
    prepare_data()
    main()
