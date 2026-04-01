import asyncio
from typing import Optional

import dspy

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)   
from retrieve_dspy.retrievers.base_retriever import BaseRetriever
from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import WriteSearchQuery, VerboseWriteSearchQuery

class SearchQueryWriter(BaseRetriever):
    def __init__(
        self,
        collection_name: str,
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = True,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 20,
        search_type: str = "hybrid",
    ):
        super().__init__(collection_name, target_property_name, search_only=search_only, verbose=verbose, retrieved_k=retrieved_k, search_type=search_type)
        signature = VerboseWriteSearchQuery if verbose else WriteSearchQuery
        self.write_search_query = dspy.Predict(signature)

    def forward(self, question: str, weaviate_client=None) -> DSPyAgentRAGResponse:
        written_search_query = self.write_search_query(question=question).search_query

        if self.verbose:
            print(f"\033[95mWritten search query from:\n{question}\nto:\n{written_search_query}\033[0m")

        sources = weaviate_search_tool(
            query=written_search_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_client=weaviate_client,
            search_type=self.search_type,
        )

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[written_search_query],
            aggregations=None,
            usage={},
        )

    async def aforward(self, question: str, weaviate_async_client=None) -> DSPyAgentRAGResponse:
        written_search_query_pred = await self.write_search_query.acall(question=question)
        written_search_query = written_search_query_pred.search_query

        if self.verbose:
            print(f"\033[95mWritten search query from:\n{question}\nto:\n{written_search_query}\033[0m")

        sources = await async_weaviate_search_tool(
            query=written_search_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_async_client=weaviate_async_client,
            search_type=self.search_type,
        )

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[written_search_query],
            aggregations=None,
            usage={},
        )
    
async def main():
    test_pipeline = SearchQueryWriter(
        collection_name="BrightBiology_Default",
        target_property_name="content",
        retrieved_k=5
    )
    test_q = "How do I integrate Weaviate and Langchain?"
    response = test_pipeline.forward(test_q)
    print(response)
    async_response = await test_pipeline.aforward(test_q)
    print(async_response)

if __name__ == "__main__":
    asyncio.run(main())