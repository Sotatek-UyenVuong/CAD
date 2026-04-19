"""search_pipeline.py — Global semantic search pipeline.

Flow:
  User Query
  → Cohere embed (search_query)
  → Qdrant search top-K (filter score >= SIMILARITY_CUTOFF_VECTORSEARCH)
  → Fetch Mongo context
  → Return top-N results sorted by vector score
"""

from __future__ import annotations

from cad_pipeline.config import (
    SIMILARITY_CUTOFF_VECTORSEARCH,
    TOP_K,
    TOP_N,
)
from cad_pipeline.core.embeddings import embed_query
from cad_pipeline.storage import mongo, qdrant_store


def run_search(
    query: str,
    top_k: int = TOP_K,
    top_n: int = TOP_N,
    folder_id: str | None = None,
    file_id: str | None = None,
) -> list[dict]:
    """Semantic search across all indexed pages.

    Args:
        query: User's search query.
        top_k: Candidates to retrieve from Qdrant.
        top_n: Final results to return.
        folder_id: Optional filter by folder.
        file_id: Optional filter by file.

    Returns:
        List of {file_name, page_number, image_url, vector_score}
        sorted by score descending, limited to top_n.
    """
    # Step 1: Embed query
    query_vector = embed_query(query)

    # Step 2: Qdrant vector search
    candidates = qdrant_store.search_pages(
        query_vector=query_vector,
        top_k=top_k,
        folder_id=folder_id,
        file_id=file_id,
    )

    # Filter by similarity cutoff
    candidates = [c for c in candidates if c["score"] >= SIMILARITY_CUTOFF_VECTORSEARCH]

    if not candidates:
        return []

    # Step 3: Fetch Mongo context
    page_ids = [c["page_id"] for c in candidates]
    pages_data = {p["_id"]: p for p in mongo.get_pages_by_ids(page_ids)}

    results: list[dict] = []
    for c in candidates:
        page_doc = pages_data.get(c["page_id"], {})
        file_doc = mongo.get_file(c["file_id"]) or {}
        results.append({
            "page_id": c["page_id"],
            "file_id": c["file_id"],
            "file_name": file_doc.get("file_name", c["file_id"]),
            "page_number": c["page_number"],
            "image_url": page_doc.get("image_url", ""),
            "short_summary": c.get("short_summary", ""),
            "vector_score": round(c["score"], 4),
        })

    return results[:top_n]
