import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        f"GROQ_API_KEY was not found.\n"
        f"Please create a .env file here:\n{ENV_FILE}"
    )

client = Groq(api_key=api_key)


# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLD = 1.20


# ============================================================
# CASUAL MESSAGES
# ============================================================

CASUAL_MESSAGES = {
    "hello",
    "hi",
    "hey",
    "hello there",
    "hi there",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
}


# ============================================================
# FOLLOW-UP WORDS
# ============================================================

FOLLOWUP_WORDS = {
    "it",
    "they",
    "them",
    "this",
    "that",
    "these",
    "those",
    "again",
    "another",
    "more",
    "example",
    "explain",
    "why",
    "how",
}


# ============================================================
# PYTHON TOPICS
# ============================================================

PYTHON_TOPICS = {
    "list",
    "lists",
    "tuple",
    "tuples",
    "dictionary",
    "dictionaries",
    "loop",
    "loops",
    "function",
    "functions",
    "class",
    "classes",
    "object",
    "objects",
    "encapsulation",
    "inheritance",
    "polymorphism",
    "abstraction",
    "variable",
    "variables",
}


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# CHECK CASUAL MESSAGE
# ============================================================

def is_casual_message(question):

    cleaned = question.lower().strip()

    return cleaned in CASUAL_MESSAGES


# ============================================================
# ASK AI USING DOCUMENT CONTEXT
# ============================================================

def ask_ai(question, context):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a document-grounded Python tutor. "
                "Your answers must be based ONLY on the provided context. "
                "Do not use your own general knowledge to add information. "
                "Do not infer benefits, reasons, examples, or details "
                "that are not explicitly supported by the context. "
                "If the context does not contain enough information to "
                "answer the question, say that the information is not "
                "available in the uploaded documents."
            ),
        }
    ]

    prompt = f"""
Answer the question using ONLY the information explicitly stated
in the provided context.

STRICT GROUNDING RULES:

- Do not use outside knowledge.
- Do not add facts that are not present in the context.
- Do not expand a short statement using your own knowledge.
- Do not invent examples.
- If the context gives only a short answer, give only a short answer.
- You may rephrase information from the context for clarity.
- You may combine information from multiple retrieved passages
  if they support the answer.
- If the context does not contain enough information, say:

"I don't know based on the provided documents."

For Python code:

- Only provide code that appears in or is directly supported by
  the context.
- Do not create new examples from your own knowledge.

Context:
{context}

Question:
{question}
"""

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.0,
        stream=True,
    )

    for chunk in stream:

        if chunk.choices and chunk.choices[0].delta.content:

            yield chunk.choices[0].delta.content


# ============================================================
# CHECK FOLLOW-UP QUESTION
# ============================================================

def is_follow_up(question):

    words = (
        question.lower()
        .replace("?", "")
        .replace(",", "")
        .replace(".", "")
        .split()
    )

    # If the user explicitly mentions a Python concept,
    # treat it as a new question.
    if any(word in PYTHON_TOPICS for word in words):

        return False

    return any(
        word in FOLLOWUP_WORDS
        for word in words
    )


# ============================================================
# REWRITE FOLLOW-UP QUESTION
# ============================================================

def rewrite_query(question):

    chat_history = st.session_state.get(
        "chat_history",
        []
    )

    # Remove the current user question
    previous_messages = chat_history[:-1]

    conversation = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in previous_messages[-6:]
    )

    rewrite_prompt = f"""
You are a query-rewriting assistant for a Python tutor.

Rewrite the latest user question into a standalone question.

IMPORTANT RULES:

- Use the previous conversation to identify the exact topic.
- The most recent topic has the highest priority.
- Preserve the exact Python concept being discussed.
- Never replace a specific concept with a general topic.
- Resolve words such as "it", "that", "this", "example",
  "another", "more", "why", and "how".
- If the user asks for an example, give an example of the
  exact concept from the previous conversation.
- Do not introduce a new topic.
- Do not answer the question.
- Return ONLY the standalone rewritten question.

Examples:

Conversation:
User: What is a list?
Assistant: A list is an ordered collection that can store multiple values.

Latest Question:
Any example?

Standalone Question:
What is an example of a Python list?

---

Conversation:
User: What is encapsulation?
Assistant: Encapsulation protects data and controls access to it.

Latest Question:
Why is it useful?

Standalone Question:
Why is encapsulation useful in Python?

---

Conversation:
User: What is a tuple?
Assistant: A tuple is an ordered, immutable collection.

Latest Question:
Can you give me an example?

Standalone Question:
Can you give me an example of a Python tuple?

---

Conversation:
{conversation}

Latest Question:
{question}

Standalone Question:
"""

    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": rewrite_prompt,
            }
        ],
        temperature=0,
    )

    standalone_question = (
        chat_completion
        .choices[0]
        .message
        .content
        .strip()
    )

    return standalone_question


# ============================================================
# CREATE FAISS VECTORSTORE
# ============================================================

def create_vectorstore(uploaded_files):

    embeddings = get_embeddings()

    print("Creating new FAISS index...")

    documents = []

    for uploaded_file in uploaded_files:

        print(
            f"Loading {uploaded_file.name}..."
        )

        temp_path = BASE_DIR / (
            f"temp_{uploaded_file.name}"
        )

        try:

            with open(temp_path, "wb") as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            loader = PyPDFLoader(
                str(temp_path)
            )

            loaded_documents = loader.load()

            for document in loaded_documents:

                document.metadata[
                    "source"
                ] = uploaded_file.name

            documents.extend(
                loaded_documents
            )

        finally:

            if temp_path.exists():

                temp_path.unlink()

    if not documents:

        raise ValueError(
            "No documents could be loaded."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    print("FAISS index created successfully!")

    return vectorstore


# ============================================================
# MAIN CHAT FUNCTION
# ============================================================

def chat(user_prompt):

    # --------------------------------------------------------
    # CASUAL MESSAGE
    # --------------------------------------------------------

    if is_casual_message(user_prompt):

        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly Python tutor. "
                        "Respond naturally to greetings and "
                        "casual conversation. "
                        "Keep responses short and friendly."
                    ),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.5,
            stream=True,
        )

        for chunk in stream:

            if chunk.choices and chunk.choices[0].delta.content:

                yield chunk.choices[0].delta.content

        return

    # --------------------------------------------------------
    # CHECK VECTORSTORE
    # --------------------------------------------------------

    if "vectorstore" not in st.session_state:

        yield (
            "Please upload a Python document first."
        )

        return

    # --------------------------------------------------------
    # FOLLOW-UP QUESTION
    # --------------------------------------------------------

    if is_follow_up(user_prompt):

        standalone_question = rewrite_query(
            user_prompt
        )

    else:

        standalone_question = user_prompt

    # --------------------------------------------------------
    # SIMILARITY SEARCH
    # --------------------------------------------------------

    results = (
        st.session_state.vectorstore
        .similarity_search_with_score(
            standalone_question,
            k=3,
        )
    )

    # --------------------------------------------------------
    # FILTER RESULTS
    # --------------------------------------------------------

    docs = []

    for doc, score in results:

        if score < THRESHOLD:

            docs.append(doc)

    # --------------------------------------------------------
    # NO RELEVANT DOCUMENT
    # --------------------------------------------------------

    if not docs:

        yield (
            "I couldn't find that information "
            "in the uploaded document(s)."
        )

        return

    # --------------------------------------------------------
    # CREATE CONTEXT
    # --------------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # --------------------------------------------------------
    # GENERATE RESPONSE
    # --------------------------------------------------------

    response = ""

    for chunk in ask_ai(
        standalone_question,
        context,
    ):

        response += chunk

        yield chunk

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    sources = set()

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown",
        )

        page = doc.metadata.get(
            "page_label",
            doc.metadata.get(
                "page",
                "Unknown",
            ),
        )

        sources.add(
            f"{os.path.basename(source)} | Page {page}"
        )

    yield "\n\n**Sources:**\n"

    for source in sorted(sources):

        yield f"- {source}\n"