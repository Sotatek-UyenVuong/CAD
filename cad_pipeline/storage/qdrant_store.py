"""qdrant_store.py — Qdrant vector store for CAD page embeddings.

Collection schema:
  id      → page_id (string → mapped to uint64 hash)
  vector  → text embedding (1536 dim)
  payload → {file_id, folder_id, page_number, short_summary}
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Sequence

from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import (  # type: ignore
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from cad_pipeline.config import (
    EMBEDDING_DIM,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)


_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30,
        )
        _ensure_collection(_client)
    return _client


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def _page_id_to_int(page_id: str) -> int:
    digest = hashlib.md5(page_id.encode()).digest()
    return struct.unpack("<Q", digest[:8])[0]


def _resolve_query_points_response(raw: Any) -> list[Any]:
    """Normalize qdrant query response across client versions."""
    if isinstance(raw, list):
        return raw
    points = getattr(raw, "points", None)
    if isinstance(points, list):
        return points
    result = getattr(raw, "result", None)
    if isinstance(result, list):
        return result
    return []


def _extract_payload_value(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def upsert_page_vector(
    page_id: str,
    vector: list[float],
    file_id: str,
    folder_id: str,
    page_number: int,
    short_summary: str = "",
) -> None:
    client = get_client()
    point = PointStruct(
        id=_page_id_to_int(page_id),
        vector=vector,
        payload={
            "page_id": page_id,
            "file_id": file_id,
            "folder_id": folder_id,
            "page_number": page_number,
            "short_summary": short_summary,
        },
    )
    client.upsert(collection_name=QDRANT_COLLECTION, points=[point])


def upsert_page_vectors_batch(records: list[dict]) -> None:
    client = get_client()
    points = [
        PointStruct(
            id=_page_id_to_int(r["page_id"]),
            vector=r["vector"],
            payload={
                "page_id": r["page_id"],
                "file_id": r["file_id"],
                "folder_id": r["folder_id"],
                "page_number": r["page_number"],
                "short_summary": r.get("short_summary", ""),
            },
        )
        for r in records
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def search_pages(
    query_vector: list[float],
    top_k: int = 20,
    folder_id: str | None = None,
    file_id: str | None = None,
) -> list[dict]:
    client = get_client()

    filter_conditions: list[FieldCondition] = []
    if folder_id:
        filter_conditions.append(FieldCondition(key="folder_id", match=MatchValue(value=folder_id)))
    if file_id:
        filter_conditions.append(FieldCondition(key="file_id", match=MatchValue(value=file_id)))

    query_filter = Filter(must=filter_conditions) if filter_conditions else None

    if hasattr(client, "search"):
        results: list[Any] = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
    elif hasattr(client, "query_points"):
        raw = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        results = _resolve_query_points_response(raw)
    else:
        raise AttributeError("QdrantClient does not support search/query_points")

    return [
        {
            "page_id": _extract_payload_value(getattr(r, "payload", {}), "page_id", ""),
            "file_id": _extract_payload_value(getattr(r, "payload", {}), "file_id", ""),
            "folder_id": _extract_payload_value(getattr(r, "payload", {}), "folder_id", ""),
            "page_number": _extract_payload_value(getattr(r, "payload", {}), "page_number", 0),
            "short_summary": _extract_payload_value(getattr(r, "payload", {}), "short_summary", ""),
            "score": getattr(r, "score", 0.0),
        }
        for r in results
    ]


def delete_file_vectors(file_id: str) -> None:
    client = get_client()
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
        ),
    )
