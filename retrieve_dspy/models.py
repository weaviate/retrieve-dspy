from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import dspy

class ObjectFromDB(BaseModel):
    object_id: str
    content: str
    relevance_rank: Optional[int]
    vector: Optional[list[float]]

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