import asyncio
from typing import Optional

import dspy

from retrieve_dspy.tools.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import ExpandQuery

class QueryExpander(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 20
    ):
        super().__init__(collection_name, target_property_name, search_only=search_only, verbose=verbose, retrieved_k=retrieved_k)
        self.expand_query = dspy.Predict(ExpandQuery)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        expanded_query = self.expand_query(question=question)

        if self.verbose:
            print(f"\033[95mExpanded query from:\n{question}\nto:\n{expanded_query}\033[0m")

        contexts, sources = weaviate_search_tool(
            query=expanded_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=[],
            searches=[expanded_query],
            aggregations=None,
            usage={},
        )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        expanded_query = await self.expand_query.acall(question=question)

        if self.verbose:
            print(f"\033[95mExpanded query from:\n{question}\nto:\n{expanded_query}\033[0m")

        contexts, sources = await async_weaviate_search_tool(
            query=expanded_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=[],
            searches=[expanded_query],
            aggregations=None,
            usage={},
        )
    
async def main():
    test_pipeline = QueryExpander(
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