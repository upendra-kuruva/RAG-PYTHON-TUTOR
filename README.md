# 🐍 PyMentor

AI-powered Python learning from your own study material.

PyMentor is a document-grounded AI Python tutor that uses **Retrieval-Augmented Generation (RAG)** to answer questions from uploaded Python study materials.

## Features

* 📚 Upload and learn from Python study materials
* 🔎 Semantic search over uploaded documents
* 🤖 AI-generated answers grounded in retrieved content
* 💬 Conversational follow-up questions
* 📄 Document and page source references
* 🛡️ Responses grounded in uploaded study material
* ⚡ Streaming AI responses

## How It Works

PyMentor follows a Retrieval-Augmented Generation (RAG) pipeline:

```text
PDF Upload
    ↓
PDF Processing
    ↓
Text Chunking
    ↓
Hugging Face Embeddings
    ↓
FAISS Vector Store
    ↓
Relevant Context Retrieval
    ↓
Follow-up Query Rewriting
    ↓
Groq Llama
    ↓
Grounded Answer + Sources
```

The system retrieves relevant information from the uploaded documents before generating an answer. The LLM is instructed to use only the retrieved document context and avoid adding unsupported information.

## Tech Stack

* **Python**
* **Streamlit** — Web application interface
* **LangChain** — Document processing and RAG components
* **FAISS** — Vector similarity search
* **Hugging Face** — Text embeddings
* **Groq** — LLM inference
* **PyPDF** — PDF document loading

## Project Structure

```text
PyMentor/
├── app.py                 # Streamlit application and UI
├── rag.py                 # RAG pipeline and AI logic
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignored files and folders
└── README.md              # Project documentation
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/umertahir7/PyMentor.git
cd PyMentor
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**

```powershell
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Groq API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

### 6. Run the application

```bash
streamlit run app.py
```

## Usage

1. Start PyMentor.
2. Upload your Python study material as PDF files.
3. Ask a question about the uploaded material.
4. Ask follow-up questions naturally.
5. Review the sources used for the response.

## Grounding

PyMentor is designed to keep responses grounded in the user's uploaded study material.

When the retrieved documents do not contain enough information to answer a question, the system is instructed to state that the information is unavailable rather than relying on outside knowledge.

## Future Improvements

* Deploy PyMentor as a public web application
* Improve retrieval evaluation
* Add additional learning-focused features
* Improve production-level error handling and monitoring
