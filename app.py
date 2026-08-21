import streamlit as st
import os
from rag import chat, create_vectorstore
st.set_page_config(
    page_title="PyMentor",
    page_icon="🐍",
    layout="wide"
)
st.markdown(
    """
    <style>

    /* Clear Chat button */
    div.stButton > button {
        border-radius: 8px;
        border: 1px solid #3a3f4b;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        padding: 6px 10px;
        border-radius: 12px;
        margin-bottom: 6px;
        border: 1px solid #2b303b;
    }

    /* User message */
    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-user"]
    ) {
        background-color: #151922;
    }

    /* Assistant message */
    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-assistant"]
    ) {
        background-color: #111827;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        max-width: 900px;
        margin: auto;
    }

    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <h1 style="margin-bottom: 0;">
        🐍 PyMentor
    </h1>
    <p style="color: #9ca3af; font-size: 16px; margin-top: 5px;">
        AI-powered Python learning from your own study material.
    </p>
    """,
    unsafe_allow_html=True
)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_upload" not in st.session_state:
    st.session_state.last_upload = ()
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

with st.sidebar:

    st.markdown(
        "### 📚 Knowledge Base"
    )
    
    st.markdown("#### Upload study material")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type="pdf",
        accept_multiple_files=True,
        help="Upload the Python study material you want PyMentor to use."
    )
    st.divider()
    st.markdown("#### 📄 Indexed PDFs")

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown(
                f"""
                <div style="
                    padding: 8px 10px;
                    margin-bottom: 6px;
                    border-radius: 8px;
                    background-color: #1e222d;
                    border: 1px solid #2b303b;
                    font-size: 14px;
                ">
                    📄 {uploaded_file.name}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No PDFs uploaded in this session.")
        st.divider()

    st.markdown("#### Conversation")

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
if not st.session_state.chat_history:
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 80px 20px;
            color: #9ca3af;
        ">
            <h2 style="color: #f3f4f6;">
                Welcome to PyMentor
            </h2>
            <p style="font-size: 16px;">
                Ask a Python question based on your uploaded study material.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.write(message["content"])
user_prompt = st.chat_input(
    "Ask anything about your Python material..."
)
if uploaded_files:

    current_upload = tuple(
        sorted(file.name for file in uploaded_files)
    )

    if current_upload != st.session_state.last_upload:

        st.success("PDF(s) uploaded successfully!")

        st.session_state.vectorstore = create_vectorstore(
        uploaded_files
        )

        st.session_state.last_upload = current_upload

        st.success("Knowledge base updated successfully!")

if user_prompt:

    if st.session_state.vectorstore is None:

        with st.chat_message("assistant"):
            st.write(
                "Please upload at least one PDF before asking a question."
            )

    else:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_prompt
            }
        )

        with st.chat_message("user"):
            st.write(user_prompt)

        response = ""

        with st.chat_message("assistant"):

            message_placeholder = st.empty()

            for chunk in chat(user_prompt):

                response += chunk

                message_placeholder.write(response)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )