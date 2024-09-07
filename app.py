import os
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

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
    You are Taxy, a highly experienced accountant providing tax advice based on Indian Tax laws.
    You will respond to the user's queries by leveraging your accounting and tax expertise and the Context Provided.
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, system_instruction="You are Lawy, a highly experienced attorney providing legal advice based on Indian laws.")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and provide a response
def user_input(user_question):
    embeddings = FastEmbedEmbeddings()
    new_db = FAISS.load_local("Faiss", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Streamlit app interface
def main():
    st.set_page_config("Taxy", page_icon=":scales:")
    st.header("Taxy: AI Tax Advisor :scales:")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi I'm Taxy, an AI Tax Advisor."}]

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

# Process PDF files and create vector store
def prepare_data():
    pdf_files = []
    for file in os.listdir("dataset"):
        if file.endswith(".pdf"):
            pdf_files.append(os.path.join("dataset", file))

    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    create_vector_store(text_chunks)

if __name__ == "__main__":
    prepare_data()  # Prepare the data before starting the app
    main()
