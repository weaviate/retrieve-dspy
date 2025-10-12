from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal

import dspy

class ObjectFromDB(BaseModel):
    object_id: str
    content: str
    relevance_rank: Optional[int] = None
    relevance_score: Optional[float] = None
    vector: Optional[list[float]] = None
    source_query: Optional[str] = None

class SearchQueryWithFilter(BaseModel):
    search_query: str
    filter: Optional[str]

class Cluster(BaseModel):
    cluster_name: str
    doc_ids: list[str]
    vectors: list[list[float]]

class DSPyAgentRAGResponse(dspy.Prediction):
    def __init__(self, final_answer: str = "", sources: List[ObjectFromDB] = None, 
                 searches: Optional[List[str]] = None, aggregations: Optional[List] = None,
                 usage: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        
        self.final_answer = final_answer
        self.sources = sources or []
        self.searches = searches
        self.usage = usage or {}

class RerankerClient(BaseModel):
    name: Literal["cohere", "voyage"]
    client: Any

class RerankItem(BaseModel):
    index: int
    relevance_score: float

class ListwiseRankedDocument(BaseModel):
    content: Any
    original_position: int
    current_position: Optional[int] = None

class MultiLMConfig(BaseModel):
    signature_name: str
    lm: Any