"""Wiring to run RAGAS against Vertex AI Gemini instead of RAGAS's OpenAI default.

Must be imported before anything else touches `ragas`: ragas 0.4.3 unconditionally
imports `langchain_community.chat_models.vertexai.ChatVertexAI` at module load time
purely for an isinstance() dispatch table, but that submodule was removed from
langchain-community 0.4.x in favour of the standalone `langchain-google-vertexai`
package, so the bare `import ragas` raises ModuleNotFoundError. The shim below
registers a stand-in module before ragas is imported so that dead import resolves;
we never use langchain-community's Vertex integration ourselves.
"""

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _RemovedChatVertexAI:
        pass

    _stub.ChatVertexAI = _RemovedChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from functools import lru_cache

from langchain_google_vertexai import ChatVertexAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from src import config
from src.embedding.embedder import GeminiEmbeddings


@lru_cache(maxsize=1)
def get_ragas_llm() -> LangchainLLMWrapper:
    chat_model = ChatVertexAI(
        model_name=config.GEMINI_MODEL,
        project=config.GCP_PROJECT,
        location=config.GCP_LOCATION,
    )
    return LangchainLLMWrapper(chat_model)


@lru_cache(maxsize=1)
def get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    return LangchainEmbeddingsWrapper(GeminiEmbeddings())
