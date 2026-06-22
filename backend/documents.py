"""
Document registry for the knowledge base.

The source of truth is the Chroma collection itself: every chunk carries a
`source` (document name) in its metadata, so the list of documents and their
chunk counts are derived directly from there. This avoids a second store that
could drift out of sync with what is actually searchable.

Uploaded originals live under data/uploads/<source>/ and are removed when a
document is deleted.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import chromadb

from config import VECTOR_DB_PATH, COLLECTION_NAME, UPLOADS_DIR


def _collection():
    """Open the Chroma collection directly (no embedding model needed for reads)."""
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


def list_documents() -> list[dict]:
    """
    Return one row per source document:
        {source, chunks, images, tables, has_upload}
    sorted by source name.
    """
    col = _collection()
    data = col.get(include=["metadatas"])
    metas = data.get("metadatas", []) or []

    summary: dict[str, dict] = {}
    for meta in metas:
        src = meta.get("source", "Unknown")
        row = summary.setdefault(src, {"source": src, "chunks": 0, "images": 0, "tables": 0})
        row["chunks"] += 1
        if meta.get("category") == "contains_image":
            row["images"] += 1
        elif meta.get("category") == "contains_table":
            row["tables"] += 1

    uploads = Path(UPLOADS_DIR)
    for row in summary.values():
        row["has_upload"] = (uploads / row["source"]).is_dir()

    return sorted(summary.values(), key=lambda r: r["source"].lower())


def count_documents() -> int:
    return len(list_documents())


def total_chunks() -> int:
    return _collection().count()


def delete_document(source: str) -> int:
    """Remove every chunk for `source` and its uploaded files. Returns chunks removed."""
    col = _collection()
    existing = col.get(where={"source": source})
    ids = existing.get("ids", []) if existing else []
    if ids:
        col.delete(ids=ids)

    upload_dir = Path(UPLOADS_DIR) / source
    if upload_dir.is_dir():
        shutil.rmtree(upload_dir, ignore_errors=True)

    return len(ids)
