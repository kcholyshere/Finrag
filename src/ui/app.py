import pandas as pd
import streamlit as st
from google.genai import errors as genai_errors

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
    if len(set(header)) != len(header):
        # Spanned headers (e.g. "June 30, 2024" repeated across six columns) come
        # out of Docling as duplicate cells; pandas accepts them but Streamlit's
        # Arrow conversion crashes. The markdown fallback renders them faithfully.
        return None
    try:
        return pd.DataFrame(body, columns=header)
    except ValueError:
        return None

st.set_page_config(page_title="IFC Annual Report 2024 - RAG", page_icon="📊", layout="wide")
st.title("IFC Annual Report 2024 (Financials) - RAG")
st.caption(
    "Multimodal RAG over the IFC Annual Report 2024 - text-chunk retrieval "
    "(hybrid + reranking) or ColPali-style page-image retrieval"
)

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

PIPELINE_LABELS = {
    "text": "Text chunks (Phases 1-5)",
    "colpali": "Page images (ColPali, Phase 6)",
}

with st.sidebar:
    pipeline = st.radio(
        "Retrieval pipeline", PIPELINE_LABELS, format_func=PIPELINE_LABELS.get
    )
    if pipeline == "colpali":
        # The page-image collection lives in Qdrant only - no backend to choose.
        backend = "qdrant"
        st.caption("Page-image retrieval runs on Qdrant (native MaxSim).")
        k = st.slider("Pages to retrieve", min_value=1, max_value=6, value=4)
        metadata_filter = None
    else:
        backend = st.selectbox("Vector store backend", ["faiss", "qdrant"])
        k = st.slider("Chunks to retrieve", min_value=2, max_value=10, value=4)
        content_type = st.selectbox("Content type filter", ["All", "text", "table", "image"])
        metadata_filter = None if content_type == "All" else {"content_type": content_type}

query = st.text_input(
    "Ask a question about the IFC Annual Report 2024",
    placeholder="What is IFC's mission and how many member countries does it have?",
)

if query:
    # Vertex AI occasionally returns transient 5xx errors; without this guard
    # Streamlit renders the raw traceback, which reads as an app crash.
    try:
        with st.spinner("Retrieving relevant context..."):
            docs, tokens = answer_query(
                query, backend=backend, k=k, pipeline=pipeline, metadata_filter=metadata_filter
            )

        st.subheader("Answer")
        answer_placeholder = st.empty()
        answer_text = ""
        for token in tokens:
            answer_text += token
            # Escape dollar signs or Streamlit's markdown treats a pair of "$"
            # on one line as LaTeX math delimiters - financial answers with two
            # amounts ("$3.8 billion vs $2.6 billion") render garbled otherwise.
            answer_placeholder.markdown(answer_text.replace("$", r"\$"))
    except genai_errors.APIError as exc:
        st.error(
            f"The model endpoint returned an error (HTTP {exc.code}). "
            "This is usually transient - please try the question again."
        )
        st.stop()
    except Exception as exc:
        # Catches everything the APIError guard above misses: a missing FAISS
        # index, Qdrant down, a missing ColPali collection, or a missing page
        # PNG. The retriever/generation layers raise these with a message
        # that's already written to be shown directly to the user.
        st.error(str(exc))
        st.stop()

    st.subheader("Retrieved source snippets")
    for i, doc in enumerate(docs, start=1):
        is_page_image = doc.metadata.get("content_type") == "page_image"
        try:
            page = config.display_page(doc.metadata.get("start_page"))
            if is_page_image:
                # ColPali pipeline source attribution: the retrieved unit is
                # the whole report page, so show the page itself - the
                # literal pixels the model answered from.
                score = doc.metadata.get("score")
                title = f"[{i}] Report page {page} (MaxSim score {score:.2f})"
            else:
                title = f"[{i}] {doc.metadata.get('section')} (page {page})"
        except Exception:
            # Malformed metadata (e.g. a missing score) shouldn't crash the
            # whole loop - fall through with a bare title and let the render
            # try/except below report the real error.
            title = f"[{i}] Source"

        with st.expander(title):
            # One bad source (e.g. a missing page image) should not take down
            # the rest of the list - degrade that entry to an error and keep
            # rendering the remaining sources.
            try:
                if is_page_image:
                    st.image(str(config.PROJECT_ROOT / doc.metadata["image_path"]))
                    continue

                df = None
                if doc.metadata.get("content_type") == "table":
                    df = _markdown_table_to_df(doc.page_content)
                if df is not None:
                    # Table chunks carry a "Table: <caption>" heading and LLM summary
                    # above the markdown table (retrieval enrichment) - show that
                    # preamble as a caption rather than losing it to the dataframe.
                    preamble = "\n".join(
                        line
                        for line in doc.page_content.splitlines()
                        if not line.strip().startswith("|")
                    ).strip()
                    if preamble:
                        # Same "$...$" markdown-math escape as the answer text above.
                        st.caption(preamble.replace("$", r"\$"))
                    st.dataframe(df, use_container_width=True)
                else:
                    # Same "$...$" markdown-math escape as the answer text above.
                    st.markdown(doc.page_content.replace("$", r"\$"))
            except Exception as exc:
                st.error(str(exc))
