from typing import List, Dict, Optional
from collections import defaultdict
from retrieve_dspy.models import ObjectFromDB

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