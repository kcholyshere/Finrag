from collections.abc import Iterator

from langchain_core.documents import Document
from langfuse import observe

from src import config
from src.retrieval.retriever import Backend, retrieve
from src.services.genai_client import get_client

SYSTEM_PROMPT = (
    "You are a financial analyst assistant answering questions about the IFC "
    "Annual Report 2024 (Financials). Answer only using the provided context. "
    "Some context is markdown tables extracted from the report - read the header "
    "row carefully to match the right column (e.g. fiscal year) to the right row "
    "before quoting or calculating a figure. If the context does not contain the "
    "answer, say so plainly. Cite the page number(s) you used."
)


def _build_prompt(query: str, context_docs: list[Document]) -> str:
    context = "\n\n".join(
        f"[Source: page {config.display_page(d.metadata.get('start_page'))}, "
        f"section '{d.metadata.get('section')}', type: {d.metadata.get('content_type')}]\n"
        f"{d.page_content}"
        for d in context_docs
    )
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"


@observe(as_type="generation")
def stream_answer(query: str, context_docs: list[Document]) -> Iterator[str]:
    client = get_client()
    prompt = _build_prompt(query, context_docs)
    for chunk in client.models.generate_content_stream(model=config.GEMINI_MODEL, contents=prompt):
        if chunk.text:
            yield chunk.text


def generate_answer(query: str, context_docs: list[Document]) -> str:
    return "".join(stream_answer(query, context_docs))


@observe(name="rag_query")
def answer_query(
    query: str, backend: Backend = "faiss", k: int = 4
) -> tuple[list[Document], Iterator[str]]:
    """Single traced entry point: retrieval and generation nest under one Langfuse trace.

    Returns the retrieved docs (for display) alongside the streaming answer.
    """
    context_docs = retrieve(query, backend=backend, k=k)
    return context_docs, stream_answer(query, context_docs)
