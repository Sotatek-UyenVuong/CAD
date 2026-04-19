"""embeddings.py — Generate text embeddings via Cohere API.

Model: embed-multilingual-v3.0 (1024-dim)
  - Supports Japanese + English natively
  - Two input types:
      "search_document" → when indexing pages (upload)
      "search_query"    → when embedding a user query (search/Q&A)

Cohere docs: https://docs.cohere.com/reference/embed
"""

from __future__ import annotations

import time
from typing import Literal, Sequence

from cad_pipeline.config import COHERE_API_KEY, COHERE_INPUT_TYPE, EMBEDDING_MODEL


def _client():
    import cohere  # type: ignore
    return cohere.Client(api_key=COHERE_API_KEY)


def embed_text(
    text: str,
    input_type: str = "search_document",
) -> list[float]:
    """Embed a single text string. Returns a 1024-dim float list."""
    return embed_batch([text], input_type=input_type)[0]


def embed_query(text: str) -> list[float]:
    """Embed a user search/Q&A query (uses search_query input type)."""
    return embed_text(text, input_type="search_query")


def embed_batch(
    texts: Sequence[str],
    input_type: str = "search_document",
    batch_size: int = 96,
    retry_delay: float = 2.0,
) -> list[list[float]]:
    """Embed a list of texts in batches using Cohere.

    Args:
        texts: List of text strings to embed.
        input_type: "search_document" (indexing) or "search_query" (retrieval).
        batch_size: Cohere allows up to 96 texts per call.
        retry_delay: Seconds to wait on rate-limit error.

    Returns:
        List of 1024-dim embedding vectors, same order as input.
    """
    client = _client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        # Cohere rejects empty strings
        batch = [t if t.strip() else "." for t in batch]

        while True:
            try:
                response = client.embed(
                    texts=batch,
                    model=EMBEDDING_MODEL,
                    input_type=input_type,
                    embedding_types=["float"],
                )
                vectors = response.embeddings.float
                all_embeddings.extend(vectors)
                break
            except Exception as exc:
                err = str(exc).lower()
                if "rate" in err or "429" in err or "too many" in err:
                    time.sleep(retry_delay)
                else:
                    raise

    return all_embeddings


def embed_image_as_text(description: str) -> list[float]:
    """Embed a textual description of an image (document side)."""
    return embed_text(description, input_type="search_document")
