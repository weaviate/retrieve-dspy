"""Relative Score Fusion (RSF) for combining BM25 and vector search results.

Normalizes scores from each pathway to [0,1] via min-max scaling, then
combines with a weighted sum: score = alpha * bm25_norm + (1-alpha) * vector_norm.

This matches Weaviate's native hybrid RSF, allowing fair comparison between
Condition A (hybrid search) and Conditions B/C (split-query retrieval).
"""

from typing import List, Optional

from retrieve_dspy.models import ObjectFromDB


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize a dict of {id: score} to [0, 1] via min-max scaling."""
    if not scores:
        return {}
    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return {k: 1.0 for k in scores}
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


def relative_score_fusion(
    bm25_results: List[ObjectFromDB],
    vector_results: List[ObjectFromDB],
    alpha: float = 0.5,
    top_k: Optional[int] = None,
) -> List[ObjectFromDB]:
    """Fuse BM25 and vector results using Relative Score Fusion.

    Args:
        bm25_results: Results from BM25 search with relevance_score populated.
        vector_results: Results from vector search with relevance_score populated
                       (already converted from distance to similarity).
        alpha: Weight for BM25 scores. Vector weight is (1 - alpha).
        top_k: Maximum number of results to return.

    Returns:
        Fused and ranked list of ObjectFromDB.
    """
    # Build score dicts
    bm25_scores = {obj.object_id: obj.relevance_score for obj in bm25_results
                   if obj.relevance_score is not None}
    vector_scores = {obj.object_id: obj.relevance_score for obj in vector_results
                     if obj.relevance_score is not None}

    # Min-max normalize each pathway
    bm25_norm = _min_max_normalize(bm25_scores)
    vector_norm = _min_max_normalize(vector_scores)

    # Build doc map (keep first occurrence of each doc for content/vector)
    doc_map: dict[str, ObjectFromDB] = {}
    for obj in bm25_results:
        if obj.object_id not in doc_map:
            doc_map[obj.object_id] = obj
    for obj in vector_results:
        if obj.object_id not in doc_map:
            doc_map[obj.object_id] = obj

    # Compute fused scores for all unique docs
    all_ids = set(bm25_norm.keys()) | set(vector_norm.keys())
    fused_scores: dict[str, float] = {}
    for doc_id in all_ids:
        b = bm25_norm.get(doc_id, 0.0)
        v = vector_norm.get(doc_id, 0.0)
        fused_scores[doc_id] = alpha * b + (1 - alpha) * v

    # Sort by fused score descending
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # Build output
    results = []
    for new_rank, (doc_id, score) in enumerate(sorted_docs[:top_k], start=1):
        obj = doc_map[doc_id]
        results.append(ObjectFromDB(
            object_id=obj.object_id,
            content=obj.content,
            relevance_rank=new_rank,
            relevance_score=score,
            vector=obj.vector,
            source_query=obj.source_query,
        ))

    return results
