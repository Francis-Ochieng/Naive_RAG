# app/pdf_handler.py

import streamlit as st
import tempfile
import os

from app.vectorstore import save_pdf_to_vectorstore
from app.chain import get_answer

st.set_page_config(page_title="Naive RAG Chatbot", page_icon="🤖", layout="centered")

st.title("📄 Naive RAG Chatbot")
st.write("Upload a PDF, then ask questions about its content.")

# Step 1: File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file to a temp directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    st.success(f"Uploaded: {uploaded_file.name}")

    # Step 2: Process and store PDF in Chroma
    with st.spinner("Processing document..."):
        save_pdf_to_vectorstore(temp_pdf_path)
    st.success("Document processed and added to vector store!")

    # Step 3: Question input
    user_question = st.text_input("Ask a question about the document:")

    if user_question:
        with st.spinner("Generating answer..."):
            try:
                answer = get_answer(user_question)
                st.markdown("### 💡 Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.info("Please upload a PDF to start.")
