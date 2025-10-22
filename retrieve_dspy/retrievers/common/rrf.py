from typing import List, Dict, Optional
from collections import defaultdict
from retrieve_dspy.models import ObjectFromDB

from typing import Dict, List, Optional
from retrieve_dspy.models import RerankItem


def fuse_rankings_with_rrf(
    rankings: Dict[str, List[RerankItem]],
    top_k: int,
    rrf_k: int = 60,
    weights: Optional[Dict[str, float]] = None,
) -> List[RerankItem]:
    """Fuse multiple rankings using Reciprocal Rank Fusion."""
    weights = weights or {}
    
    # Handle edge cases
    cohere_items = rankings.get("cohere", [])
    voyage_items = rankings.get("voyage", [])
    if not cohere_items:
        return voyage_items[:top_k]
    if not voyage_items:
        return cohere_items[:top_k]
    
    # Calculate RRF scores
    scores: Dict[int, float] = {}
    for name, items in rankings.items():
        if not items:
            continue
        w = weights.get(name, 0.5)
        for rank, item in enumerate(items):
            scores[item.index] = scores.get(item.index, 0.0) + w / (rrf_k + rank + 1)
    
    # Sort and return
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [RerankItem(index=idx, relevance_score=score) for idx, score in fused]

def reciprocal_rank_fusion(
    result_sets: List[List[ObjectFromDB]], 
    k: int = 60,  # Standard RRF constant
    top_k: Optional[int] = None
) -> List[ObjectFromDB]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    Args:
        result_sets: List of lists, each containing ObjectFromDB results from different queries
        k: RRF constant (typically 60)
        top_k: Number of top results to return
    """
    # Track RRF scores and document details
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, ObjectFromDB] = {}
    
    for result_set in result_sets:
        for rank, obj in enumerate(result_set, start=1):
            doc_id = obj.object_id
            
            # Calculate RRF score: 1/(rank + k)
            rrf_scores[doc_id] += 1.0 / (rank + k)
            
            # Store document if not seen before (keeps first occurrence)
            if doc_id not in doc_map:
                doc_map[doc_id] = obj
    
    # Sort by RRF score and create final ranking
    sorted_docs = sorted(
        rrf_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Create output list with updated ranks and scores
    results = []
    for new_rank, (doc_id, rrf_score) in enumerate(sorted_docs[:top_k], start=1):
        obj = doc_map[doc_id]
        # Create new object with updated rank and score
        results.append(ObjectFromDB(
            object_id=obj.object_id,
            content=obj.content,
            relevance_rank=new_rank,
            relevance_score=rrf_score,
            vector=obj.vector,
            source_query=obj.source_query
        ))
    
    return results