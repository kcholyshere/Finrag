from collections.abc import Iterator

from langchain_core.documents import Document

from src import config
from src.services.genai_client import get_client

SYSTEM_PROMPT = (
    "You are a financial analyst assistant answering questions about the IFC "
    "Annual Report 2024 (Financials). Answer only using the provided context. "
    "If the context does not contain the answer, say so plainly. Cite the page "
    "number(s) you used."
)


def _build_prompt(query: str, context_docs: list[Document]) -> str:
    context = "\n\n".join(
        f"[Source: page {d.metadata.get('start_page')}, section '{d.metadata.get('section')}']\n"
        f"{d.page_content}"
        for d in context_docs
    )
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def stream_answer(query: str, context_docs: list[Document]) -> Iterator[str]:
    client = get_client()
    prompt = _build_prompt(query, context_docs)
    for chunk in client.models.generate_content_stream(model=config.GEMINI_MODEL, contents=prompt):
        if chunk.text:
            yield chunk.text


def generate_answer(query: str, context_docs: list[Document]) -> str:
    return "".join(stream_answer(query, context_docs))
