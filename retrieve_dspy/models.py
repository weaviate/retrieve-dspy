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

class AgentRAGResponse(BaseModel):
    final_answer: str
    sources: list[Source]
    searches: Optional[list[str]] = None
    aggregations: Optional[list] = None
    usage: Optional[Dict[str, Any]] = None

class DSPyAgentRAGResponse(dspy.Prediction):
    """
    DSPy-compatible version of AgentRAGResponse that inherits from dspy.Prediction.
    
    This class provides the set_lm_usage method that DSPy expects while maintaining
    compatibility with the existing benchmark infrastructure that expects AgentRAGResponse interface.
    """
    
    def __init__(self, final_answer: str = "", sources: List[Source] = None, 
                 searches: Optional[List[str]] = None, aggregations: Optional[List] = None,
                 usage: Optional[Dict[str, Any]] = None, **kwargs):
        # Initialize the parent dspy.Prediction class
        super().__init__(**kwargs)
        
        # Set our custom attributes
        self.final_answer = final_answer
        self.sources = sources or []
        self.searches = searches
        self.aggregations = aggregations
        self.usage = usage or {}
    
    def to_agent_rag_response(self) -> AgentRAGResponse:
        """Convert to the original AgentRAGResponse for compatibility with benchmark infrastructure."""
        return AgentRAGResponse(
            final_answer=self.final_answer,
            sources=self.sources,
            searches=self.searches,
            aggregations=self.aggregations,
            usage=self.usage
        )
    
    @classmethod
    def from_agent_rag_response(cls, response: AgentRAGResponse) -> 'DSPyAgentRAGResponse':
        """Create DSPyAgentRAGResponse from AgentRAGResponse."""
        return cls(
            final_answer=response.final_answer,
            sources=response.sources,
            searches=response.searches,
            aggregations=response.aggregations,
            usage=response.usage
        )