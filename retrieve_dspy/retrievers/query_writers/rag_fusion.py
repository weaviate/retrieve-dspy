import asyncio
from typing import Optional

import dspy

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB
from retrieve_dspy.signatures import WriteSearchQueries, VerboseWriteSearchQueries

class RAGFusion(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        verbose: Optional[bool] = False,
        verbose_signature: Optional[bool] = True
    ):
        super().__init__(collection_name, target_property_name, verbose)
        if self.verbose_signature:
            self.decompose_query = dspy.Predict(VerboseWriteSearchQueries)
        else:
            self.decompose_query = dspy.Predict(WriteSearchQueries)
    
    def forward(self, question: str) -> DSPyAgentRAGResponse:
        search_queries = self.decompose_query(question=question)
        search_queries = search_queries.search_queries
        if self.verbose:
            print(f"Search queries: {search_queries}")
        sources: list[ObjectFromDB] = []
        for q in search_queries:
            _, src = weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k
            )
            sources.extend(src)
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=[],
            searches=search_queries,
            aggregations=None,
            usage={},
        )