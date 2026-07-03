from functools import lru_cache

from google import genai

from src import config


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=config.GCP_PROJECT,
        location=config.GCP_LOCATION,
    )
