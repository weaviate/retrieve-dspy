from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import dspy

class Source(BaseModel):
    object_id: str

class SourceWithContentAndVector(Source):
    content: str
    vector: list[float]

class SearchResult(BaseModel):
    id: int
    content: str
    dataset_id: Optional[str]

class SearchQueryWithFilter(BaseModel):
    search_query: str
    filter: Optional[str]

class Cluster(BaseModel):
    cluster_name: str
    doc_ids: list[str]
    vectors: list[list[float]]

class DSPyAgentRAGResponse(dspy.Prediction):
    def __init__(self, final_answer: str = "", sources: List[Source] = None, 
                 searches: Optional[List[str]] = None, aggregations: Optional[List] = None,
                 usage: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        
        self.final_answer = final_answer
        self.sources = sources or []
        self.searches = searches
        self.aggregations = aggregations
        self.usage = usage or {}