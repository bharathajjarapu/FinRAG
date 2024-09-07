# FinTAX AI 💼
AI-Driven Tax Saving Assistant for Indian Finance 

**FinTAX** is an AI-powered tax advisor for Indian finance and tax laws. Using advanced LLMs, it helps users navigate tax regulations, optimize savings, and receive personalized financial guidance via a chatbot interface.

## Features 🔥

- **AI Tax Advisor**: Tailored tax advice based on Indian laws.
- **Natural Language Understanding**: Understands and responds to complex tax queries.
- **PDF Processing**: Analyzes financial documents for tax-saving insights.
- **Efficient Search**: Uses FAISS for quick access to relevant info from large documents.
- **User Authentication**: Secure login/signup with SQLite3.
- **Modern UI**: Built with Streamlit for a sleek, intuitive experience.

## Use Cases 🚀

1. **Tax Savings**: Get personalized tax-saving tips.
2. **Document Analysis**: Upload tax forms for detailed advice.
3. **Financial Planning**: Optimize finances for tax efficiency.
4. **Tax Law Queries**: Clarify doubts about Indian tax laws.

## Technical Stack 🛠️

- **Python**: Core programming language.
- **Streamlit**: Web-based chatbot UI.
- **Langchain**: Manages conversational AI and document processing.
- **FAISS**: Fast similarity search for relevant data.
- **SQLite3**: Handles user authentication.
- **PyPDF2**: Parses PDF documents.

## Architecture Overview 🧠

1. **Frontend**: Streamlit-powered chatbot UI.
2. **Backend**: Langchain manages AI workflows, Google LLM for responses, FAISS for efficient info retrieval.
3. **PDF Processing**: Extracts and indexes text for quick search.
4. **Authentication**: SQLite3 ensures secure user sessions.

## Installation Guide 🛠️

1. **Clone the repository**:
    ```bash
    git clone https://github.com/bharathajjarapu/FinTax.git
    cd FinTAX
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Set up environment variables**:
    ```bash
    GOOGLE_API_KEY=your_google_api_key
    ```
4. **Prepare the dataset**: Place PDFs in the `dataset` folder.
5. **Run the application**:
    ```bash
    streamlit run app.py
    ```

## Example Usage 👨‍💻

1. **Login/Sign Up**: Create an account or log in.
2. **Chat with Taxy**: Ask tax-related questions.
3. **Upload PDFs**: Receive document-based advice.

## Why This Project? 💡

As a college student passionate about finance and AI, I created FinTAX to simplify tax planning and help users make better financial decisions.

## Future Improvements 🔮

- **Financial Planning**: Add investment and retirement advice.
- **Tax Law Updates**: Keep AI updated with latest tax laws.
- **Multi-Language Support**: Enable conversations in regional Indian languages.
- **Mobile App**: Build a mobile version for on-the-go advice.

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
