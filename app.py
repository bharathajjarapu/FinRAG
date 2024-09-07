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
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import plotly.graph_objects as go
from googletrans import Translator

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Async function to get text from PDF files
async def get_pdf_text(pdf_docs):
    text = ""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        for pdf in pdf_docs:
            if isinstance(pdf, (str, io.IOBase)):
                pdf_reader = await loop.run_in_executor(pool, PdfReader, pdf)
                for page in pdf_reader.pages:
                    text += await loop.run_in_executor(pool, page.extract_text)
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
@st.cache_resource
def create_vector_store(text_chunks):
    embeddings = FastEmbedEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Function to load and configure the conversational chain
def get_conversational_chain():
    prompt_template = """
    You are Taxy, a highly experienced accountant providing tax advice based on Indian Tax laws.
    You will respond to the user's queries by leveraging your accounting and tax expertise and the Context Provided.
    
    Important Instructions:
    1. Always provide accurate information based on the latest Indian tax laws.
    2. If you're unsure about any information, clearly state that and suggest consulting a professional tax advisor.
    3. For numerical calculations, show the step-by-step process.
    4. If the query is not related to Indian taxes, politely redirect the user to ask about Indian tax matters.

    Examples:
    User: What are the tax slabs for FY 2024-25?
    Taxy: For FY 2024-25, India offers two tax regimes: the old regime and the new regime. Here are the tax slabs for the new regime:
    - Up to ₹3 lakh: No tax
    - ₹3-6 lakh: 5%
    - ₹6-9 lakh: 10%
    - ₹9-12 lakh: 15%
    - ₹12-15 lakh: 20%
    - Above ₹15 lakh: 30%
    Remember, these slabs are subject to change, so always verify with the latest official information.

    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3,
                                   system_instruction="You are Taxy, a highly experienced accountant providing tax advice based on Indian laws.")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and provide a response
@st.cache_data(show_spinner=False)
def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to translate text
def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Streamlit app interface
def main():
    st.set_page_config("Taxy", page_icon=":scales:", layout="wide")
    st.title("Taxy: AI Tax Advisor :scales:")

    # Sidebar for language selection and file upload
    with st.sidebar:
        st.header("Settings")
        language = st.selectbox("Select Language", ["English", "Hindi", "Bengali", "Tamil"])
        uploaded_file = st.file_uploader("Upload your tax document", type="pdf")
        if uploaded_file:
            with st.spinner("Processing your document..."):
                text = asyncio.run(get_pdf_text([uploaded_file]))
                chunks = get_text_chunks(text)
                st.session_state.personal_vector_store = create_vector_store(chunks)
            st.success("Document processed successfully!")

    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi I'm Taxy, an AI Tax Advisor. How can I help you today?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask me about Indian taxes...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if hasattr(st.session_state, 'personal_vector_store'):
                    vector_store = st.session_state.personal_vector_store
                else:
                    vector_store = create_vector_store(get_text_chunks(asyncio.run(get_pdf_text(["dataset/indian_tax_laws.pdf"]))))
                
                response = user_input(prompt, vector_store)
                
                if language != "English":
                    response = translate_text(response, language.lower())
                
                st.write(response)

        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

    # Display tax brackets visualization
    st.subheader("Indian Tax Brackets (New Regime) FY 2024-25")
    fig = go.Figure(data=[go.Bar(
        x=['0-3L', '3L-6L', '6L-9L', '9L-12L', '12L-15L', '15L+'],
        y=[0, 5, 10, 15, 20, 30],
        text=['0%', '5%', '10%', '15%', '20%', '30%'],
        textposition='auto',
    )])
    fig.update_layout(title_text='Tax Rates', xaxis_title='Income Bracket', yaxis_title='Tax Rate (%)')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
