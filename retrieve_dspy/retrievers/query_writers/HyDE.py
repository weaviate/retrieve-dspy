import asyncio
import os
from typing import Optional

import dspy
import weaviate

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.retrievers.base_retriever import BaseRetriever

from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import HyDE, VerboseHyDE

class HyDE_QueryExpander(BaseRetriever):
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient | weaviate.WeaviateAsyncClient] = None,
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 20,
        search_type: str = "hybrid",
    ):
        super().__init__(
            collection_name=collection_name,
            target_property_name=target_property_name,
            search_only=search_only,
            verbose=verbose,
            retrieved_k=retrieved_k,
            search_type=search_type,
        )
        self.weaviate_client = weaviate_client
        if self.verbose:
            self.hyde = dspy.Predict(VerboseHyDE)
        else:
            self.hyde = dspy.Predict(HyDE)

    def forward(self, question: str, weaviate_client: Optional[weaviate.WeaviateClient] = None) -> DSPyAgentRAGResponse:
        if weaviate_client is None:
            if isinstance(self.weaviate_client, weaviate.WeaviateClient):
                weaviate_client = self.weaviate_client

        hypothetical_passage = self.hyde(question=question).passage

        if self.verbose:
            print(f"\033[95mHypothetical passage:\n{hypothetical_passage}\033[0m")

        sources = weaviate_search_tool(
            query=hypothetical_passage,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            weaviate_client=weaviate_client,
            retrieved_k=self.retrieved_k,
            search_type=self.search_type,
        )

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[hypothetical_passage],
            aggregations=None,
            usage={},
        )
    
    async def aforward(self, question: str, weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None) -> DSPyAgentRAGResponse:
        if weaviate_async_client is None:
            if isinstance(self.weaviate_async_client, weaviate.WeaviateAsyncClient):
                weaviate_async_client = self.weaviate_async_client

        hypothetical_passage_pred = await self.hyde.acall(question=question)
        hypothetical_passage = hypothetical_passage_pred.passage

        if self.verbose:
            print(f"\033[95mHypothetical passage:\n{hypothetical_passage}\033[0m")

        sources = await async_weaviate_search_tool(
            query=hypothetical_passage,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            weaviate_async_client=weaviate_async_client,
            retrieved_k=self.retrieved_k,
            search_type=self.search_type,
        )

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[hypothetical_passage],
            aggregations=None,
            usage={},
        )

async def main():
    test_pipeline = HyDE_QueryExpander(
        collection_name="BrightBiology",
        target_property_name="content",
        retrieved_k=5
    )
    test_q = "How many cells are in the human body?"
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    weaviate_async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    await weaviate_async_client.connect()
    test_sync_response = test_pipeline.forward(test_q, weaviate_client=weaviate_client)
    print(test_sync_response)
    test_async_response = await test_pipeline.aforward(test_q, weaviate_async_client=weaviate_async_client)
    print(test_async_response)

if __name__ == "__main__":
    asyncio.run(main())