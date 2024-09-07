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
import asyncio
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and configure API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Supported languages
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn"
}

# Function to get text from PDF files
@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into manageable chunks
@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
@st.cache_resource
def create_vector_store(text_chunks):
    embeddings = FastEmbedEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("Faiss")

# Function to load and configure the conversational chain
@lru_cache(maxsize=None)
def get_conversational_chain():
    prompt_template = """
    You are Taxy, a highly experienced accountant providing tax advice based on Indian Tax laws.
    You will respond to the user's queries by leveraging your accounting and tax expertise and the Context Provided.
    Always strive to give accurate, up-to-date information based on the latest Indian tax laws.
    If you're unsure about any information, please state that clearly.

    Context: {context}
    Question: {question}
    Language: {language}

    Please provide your answer in the specified language.

    Example of a good response:
    "Based on the current Indian tax laws, [specific tax advice]. However, please note that tax laws can change, and it's always best to consult with a certified tax professional for personalized advice."

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, system_instruction="You are Taxy, a highly experienced tax advisor providing advice based on Indian tax laws.")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "language"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and provide a response
async def user_input(user_question, language):
    try:
        embeddings = FastEmbedEmbeddings()
        new_db = FAISS.load_local("Faiss", embeddings)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = await asyncio.to_thread(chain, {"input_documents": docs, "question": user_question, "language": language})
        return response["output_text"]
    except Exception as e:
        logger.error(f"Error processing user input: {e}")
        return "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."

# Streamlit app interface
def main():
    st.set_page_config("Taxy", page_icon=":scales:", layout="wide")
    st.title("Taxy: AI Tax Advisor :scales:")

    # Sidebar for language selection
    with st.sidebar:
        st.header("Settings")
        selected_language = st.selectbox("Select Language", list(LANGUAGES.keys()))

    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi I'm Taxy, an AI Tax Advisor. How can I help you today?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    prompt = st.chat_input("Ask me about Indian taxes...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = asyncio.run(user_input(prompt, LANGUAGES[selected_language]))
                st.write(response)

        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

    # FAQ Section
    with st.expander("Frequently Asked Questions"):
        st.subheader("Common Tax-Related Questions")
        st.write("1. What are the current tax slabs in India?")
        st.write("2. How can I claim deductions under Section 80C?")
        st.write("3. What is the difference between old and new tax regimes?")
        st.write("4. How do I file my ITR online?")
        st.write("5. What are the tax implications of investing in mutual funds?")

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
    try:
        prepare_data()  # Prepare the data before starting the app
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Application error: {e}")
