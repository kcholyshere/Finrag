import streamlit as st

from src.generation.answer import stream_answer
from src.retrieval.retriever import retrieve

st.set_page_config(page_title="IFC Annual Report 2024 - RAG", page_icon="📊")
st.title("IFC Annual Report 2024 (Financials) - RAG")
st.caption("Naive text-based RAG over the IFC Annual Report 2024 (Phase 1)")

with st.sidebar:
    backend = st.selectbox("Vector store backend", ["faiss", "qdrant"])
    k = st.slider("Chunks to retrieve", min_value=2, max_value=10, value=4)

query = st.text_input(
    "Ask a question about the IFC Annual Report 2024",
    placeholder="What is IFC's mission and how many member countries does it have?",
)

if query:
    with st.spinner("Retrieving relevant context..."):
        docs = retrieve(query, backend=backend, k=k)

    st.subheader("Answer")
    answer_placeholder = st.empty()
    answer_text = ""
    for token in stream_answer(query, docs):
        answer_text += token
        answer_placeholder.markdown(answer_text)

    st.subheader("Retrieved source snippets")
    for i, doc in enumerate(docs, start=1):
        section = doc.metadata.get("section")
        page = doc.metadata.get("start_page")
        with st.expander(f"[{i}] {section} (page {page})"):
            st.write(doc.page_content)
