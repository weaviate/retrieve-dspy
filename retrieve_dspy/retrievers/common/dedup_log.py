"""Simple file logger for tracking unique document counts after deduplication/fusion."""

import os
from pathlib import Path
from typing import List

from retrieve_dspy.models import ObjectFromDB

# Log file lives at the repo root so it's easy to find and delete
LOG_PATH = Path(os.environ.get("DEDUP_LOG_PATH", "dedup_log.jsonl"))


def log_dedup(
    retriever: str,
    question: str,
    bm25_results: List[ObjectFromDB],
    vector_results: List[ObjectFromDB],
    fused_count: int,
    retrieved_k: int,
):
    """Append one JSON-lines entry tracking per-query deduplication stats."""
    import json
    from datetime import datetime, timezone

    bm25_ids = {obj.object_id for obj in bm25_results}
    vector_ids = {obj.object_id for obj in vector_results}
    overlap = len(bm25_ids & vector_ids)
    unique_total = len(bm25_ids | vector_ids)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retriever": retriever,
        "retrieved_k": retrieved_k,
        "bm25_docs": len(bm25_results),
        "vector_docs": len(vector_results),
        "unique_total": unique_total,
        "fused_returned": fused_count,
        "overlap": overlap,
        "question": question[:200],
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
