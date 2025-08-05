from typing import Optional

import dspy

from retrieve_dspy.tools.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import WriteFollowUpQueries

class LoopingQueryWriter(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 20
    ):
        super().__init__(collection_name, target_property_name, search_only=search_only, verbose=verbose, retrieved_k=retrieved_k)
        self.looping_query_writer = dspy.Predict(WriteFollowUpQueries)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        contexts, sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        follow_up_queries_needed, follow_up_queries = self.looping_query_writer(
            question=question,
            contexts=contexts,
        )

        if follow_up_queries_needed:
            for follow_up_query in follow_up_queries:
                contexts, sources = weaviate_search_tool(
                    query=follow_up_query,
                    collection_name=self.collection_name,
                    target_property_name=self.target_property_name,
                    retrieved_k=self.retrieved_k,
                )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        pass