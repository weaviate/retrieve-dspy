import asyncio
from typing import Optional

import dspy

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)
from retrieve_dspy.retrievers.base_retriever import BaseRetriever
from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import ExpandQueryWithHint

class PRF_QueryExpander(BaseRetriever):
    def __init__(
        self,
        collection_name: str,
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 20
    ):
        super().__init__(collection_name, target_property_name, search_only=search_only, verbose=verbose, retrieved_k=retrieved_k)
        self.expand_query = dspy.Predict(ExpandQueryWithHint)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        initial_search_results = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=5
        )
        expanded_query = self.expand_query(question=question, initial_search_results=initial_search_results).expanded_query

        if self.verbose:
            print(f"\033[95mExpanded query from:\n{question}\nto:\n{expanded_query}\033[0m")

        sources = weaviate_search_tool(
            query=expanded_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[expanded_query],
            aggregations=None,
            usage={},
        )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        initial_search_results = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=5
        )
        expanded_query_pred = await self.expand_query.acall(question=question, initial_search_results=initial_search_results)
        expanded_query = expanded_query_pred.expanded_query

        if self.verbose:
            print(f"\033[95mExpanded query from:\n{question}\nto:\n{expanded_query}\033[0m")

        sources = await async_weaviate_search_tool(
            query=expanded_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[expanded_query],
            aggregations=None,
            usage={},
        )
    
async def main():
    test_pipeline = PRF_QueryExpander(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=5
    )
    test_q = "How do I integrate Weaviate and Langchain?"
    response = test_pipeline.forward(test_q)
    print(response)
    async_response = await test_pipeline.aforward(test_q)
    print(async_response)

if __name__ == "__main__":
    asyncio.run(main())