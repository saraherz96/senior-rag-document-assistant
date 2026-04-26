import sys
import os
sys.path.append(os.path.abspath("."))

import time
import streamlit as st

from app.utils.pdf_loader import read_pdf
from app.rag.chunking import chunk_text
from app.rag.retriever import save_chunks, retrieve_context, build_context
from app.rag.generator import generate_answer

st.set_page_config(
    page_title="Senior RAG Document Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("Senior RAG Document Assistant")
st.write("Upload PDF documents and ask questions grounded in real context.")

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process documents"):
        with st.spinner("Processing documents..."):
            total_chunks = 0

            for file in uploaded_files:
                text = read_pdf(file)

                if not text.strip():
                    st.warning(f"No text extracted from {file.name}. It may be a scanned PDF.")
                    continue

                chunks = chunk_text(text)
                save_chunks(chunks, file.name)
                total_chunks += len(chunks)

            st.success(f"Documents processed successfully. Total chunks: {total_chunks}")

question = st.text_input("Ask a question about your documents")

if question:
    start_time = time.time()

    with st.spinner("Retrieving context and generating answer..."):
        retrieved_chunks = retrieve_context(question)
        context = build_context(retrieved_chunks)
        answer = generate_answer(question, context)

    end_time = time.time()

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for item in retrieved_chunks:
        st.write(
            f"Source: {item['source']} | Chunk: {item['chunk_index']} | Distance: {round(item['distance'], 4)}"
        )

    with st.expander("View retrieved context"):
        st.write(context)

    st.info(f"Response time: {round(end_time - start_time, 2)} seconds")