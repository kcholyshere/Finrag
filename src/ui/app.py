import pandas as pd
import streamlit as st

from src import config
from src.generation.answer import answer_query


def _markdown_table_to_df(markdown: str) -> pd.DataFrame | None:
    """Parse a GFM pipe table (header, separator, body rows) into a DataFrame.

    Table chunks are always Docling's own export_to_markdown output, so the
    shape is predictable; falls back to None on anything unexpected rather
    than crashing the page on a malformed table.
    """
    lines = [line.strip() for line in markdown.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return None
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    header, _separator, *body = rows
    try:
        return pd.DataFrame(body, columns=header)
    except ValueError:
        return None

st.set_page_config(page_title="IFC Annual Report 2024 - RAG", page_icon="📊", layout="wide")
st.title("IFC Annual Report 2024 (Financials) - RAG")
st.caption("Naive text-based RAG over the IFC Annual Report 2024 (Phase 1)")

# Wide financial tables (content_type: "table") otherwise get clipped by the page
# edge instead of scrolling - Streamlit doesn't add horizontal scroll to markdown
# tables by default.
st.markdown(
    """
    <style>
    div[data-testid="stMarkdownContainer"] table {
        display: block;
        overflow-x: auto;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    backend = st.selectbox("Vector store backend", ["faiss", "qdrant"])
    k = st.slider("Chunks to retrieve", min_value=2, max_value=10, value=4)

query = st.text_input(
    "Ask a question about the IFC Annual Report 2024",
    placeholder="What is IFC's mission and how many member countries does it have?",
)

if query:
    with st.spinner("Retrieving relevant context..."):
        docs, tokens = answer_query(query, backend=backend, k=k)

    st.subheader("Answer")
    answer_placeholder = st.empty()
    answer_text = ""
    for token in tokens:
        answer_text += token
        answer_placeholder.markdown(answer_text)

    st.subheader("Retrieved source snippets")
    for i, doc in enumerate(docs, start=1):
        section = doc.metadata.get("section")
        page = config.display_page(doc.metadata.get("start_page"))
        with st.expander(f"[{i}] {section} (page {page})"):
            df = None
            if doc.metadata.get("content_type") == "table":
                df = _markdown_table_to_df(doc.page_content)
            if df is not None:
                st.dataframe(df, use_container_width=True)
            else:
                st.write(doc.page_content)
