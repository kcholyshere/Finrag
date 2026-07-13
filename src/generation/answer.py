from collections.abc import Iterator

from google.genai import errors, types
from langchain_core.documents import Document
from langfuse import observe

from src import config
from src.generation.calculator import calculate
from src.retrieval.retriever import Backend, retrieve_reranked
from src.services.genai_client import get_client

SYSTEM_PROMPT = (
    "You are a financial analyst assistant answering questions about the IFC "
    "Annual Report 2024 (Financials). Answer only using the provided context. "
    "Some context is markdown tables extracted from the report - read the header "
    "row carefully to match the right column (e.g. fiscal year) to the right row "
    "before quoting or calculating a figure. When a question needs arithmetic "
    "(differences, percentage changes, ratios), call the calculate tool with the "
    "figures from the context rather than computing mentally. If the context does "
    "not contain the answer, say so plainly. Cite the page number(s) you used."
)

# The SDK auto-generates the function declaration from the callable, but its
# automatic function calling is disabled here because it silently returns an
# empty stream with generate_content_stream (google-genai 2.10.0) - so
# stream_answer runs the execute-and-feed-back loop itself instead.
GENERATION_CONFIG = types.GenerateContentConfig(
    tools=[calculate],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

# One round covers a multi-step expression; a few spares let the model recover
# from a malformed expression without ever looping forever.
MAX_TOOL_TURNS = 4


def _run_calculate(args: dict) -> dict:
    try:
        return {"result": calculate(**args)}
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as exc:
        return {"error": str(exc)}


def _build_prompt(query: str, context_docs: list[Document]) -> str:
    context = "\n\n".join(
        f"[Source: page {config.display_page(d.metadata.get('start_page'))}, "
        f"section '{d.metadata.get('section')}', type: {d.metadata.get('content_type')}]\n"
        f"{d.page_content}"
        for d in context_docs
    )
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def _stream_turn(client, contents: list) -> Iterator:
    """Yield chunks for one model turn, retrying once if the endpoint fails with
    a 5xx before any chunk arrives. Mid-stream failures re-raise instead -
    retrying those would replay tokens the user has already seen.
    """
    for attempts_left in (1, 0):
        received = False
        try:
            for chunk in client.models.generate_content_stream(
                model=config.GEMINI_MODEL, contents=contents, config=GENERATION_CONFIG
            ):
                received = True
                yield chunk
            return
        except errors.ServerError:
            if received or not attempts_left:
                raise


@observe(as_type="generation")
def stream_answer(query: str, context_docs: list[Document]) -> Iterator[str]:
    client = get_client()
    contents: list = [_build_prompt(query, context_docs)]

    for _ in range(MAX_TOOL_TURNS):
        function_calls = []
        model_parts = []
        for chunk in _stream_turn(client, contents):
            if chunk.function_calls:
                function_calls.extend(chunk.function_calls)
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    model_parts.append(part)
                    # Thought parts are internal reasoning - keep them in the
                    # history (their signatures are required) but never show them.
                    if part.text and not part.thought:
                        yield part.text

        if not function_calls:
            return

        # Echo the model's own parts back verbatim - Gemini 3.5 rejects the next
        # turn if the function_call parts lose their thought_signature.
        contents.append(types.Content(role="model", parts=model_parts))
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=fc.name, response=_run_calculate(dict(fc.args))
                    )
                    for fc in function_calls
                ],
            )
        )


def generate_answer(query: str, context_docs: list[Document]) -> str:
    return "".join(stream_answer(query, context_docs))


@observe(name="rag_query")
def answer_query(
    query: str, backend: Backend = "faiss", k: int = 4
) -> tuple[list[Document], Iterator[str]]:
    """Single traced entry point: retrieval and generation nest under one Langfuse trace.

    Returns the retrieved docs (for display) alongside the streaming answer.
    Uses the strongest retrieval pipeline (hybrid + cross-encoder reranking) -
    plain dense retrieve() remains available for comparisons and evaluation.
    """
    context_docs = retrieve_reranked(query, backend=backend, k=k)
    return context_docs, stream_answer(query, context_docs)
