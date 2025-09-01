import asyncio
from typing import Optional

import dspy
import weaviate

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse

class VanillaRAG(BaseRAG):
    def __init__(
        self, 
        weaviate_client: weaviate.WeaviateClient,
        collection_name: str, 
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 20,
    ):
        super().__init__(weaviate_client, collection_name, target_property_name, search_only=search_only, verbose=verbose, retrieved_k=retrieved_k)
        
    def forward(self, question: str, weaviate_client: Optional[weaviate.WeaviateClient] = None) -> DSPyAgentRAGResponse:
        if weaviate_client is None:
            weaviate_client = self.weaviate_client

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
            usage={},
        )
    
    async def aforward(self, weaviate_async_client: weaviate.WeaviateAsyncClient, question: str) -> DSPyAgentRAGResponse:
        pass # Need to resolve async client injection
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
            usage={},
        )

async def main():
    import os
    test_pipeline = VanillaRAG(
        collection_name="EnronEmails",
        target_property_name="email_body",
        retrieved_k=5
    )
    test_q = "What are the implications of SBX12?"
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    response = test_pipeline.forward(weaviate_client, test_q)
    print(response)
    weaviate_async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    await weaviate_async_client.connect()
    async_response = await test_pipeline.aforward(weaviate_async_client, test_q)
    print(async_response)

if __name__ == "__main__":
    asyncio.run(main())