import asyncio
from typing import Optional

import dspy
import weaviate

from retrieve_dspy.tools.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import QuerySummarizer

class VanillaRAG(BaseRAG):
    def __init__(
        self, 
        collection_name: str, 
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 20,
    ):
        super().__init__(collection_name, target_property_name, search_only=search_only, verbose=verbose, retrieved_k=retrieved_k)
        
    def forward(self, weaviate_client: weaviate.Client, question: str) -> DSPyAgentRAGResponse:
        sources = weaviate_search_tool(
            weaviate_client=weaviate_client,
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        if not self.search_only:
            print("")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[question],
            aggregations=None,
            usage={},
        )
    
    async def aforward(self, weaviate_async_client: weaviate.AsyncClient, question: str) -> DSPyAgentRAGResponse:
        sources = await async_weaviate_search_tool(
            weaviate_async_client=weaviate_async_client,
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        if not self.search_only:
            print("")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[question],
            aggregations=None,
            usage={},
        )

async def main():
    test_pipeline = VanillaRAG(
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