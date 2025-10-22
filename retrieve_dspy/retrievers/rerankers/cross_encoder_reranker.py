from __future__ import annotations

from typing import Optional, List, Dict

import weaviate

from retrieve_dspy.database.weaviate_database import weaviate_search_tool, async_weaviate_search_tool
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB, RerankerClient
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.retrievers.common.call_ce_ranker import ce_rank, async_ce_rank, reorder, Provider


class CrossEncoderReranker(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
        return_property_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 50,
        reranked_k: Optional[int] = 20,
        model_name_overrides: Optional[Dict[str, str]] = None,
        provider: Optional[Provider] = None,
        rrf_k: Optional[int] = 60,
        hybrid_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            weaviate_client=weaviate_client,
            collection_name=collection_name,
            target_property_name=target_property_name,
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k,
        )
        self.return_property_name = return_property_name
        self.reranker_clients = reranker_clients
        self.reranked_k = int(reranked_k or 20)
        self.model_name_overrides = model_name_overrides or {}
        self.provider = provider
        self.rrf_k = int(rrf_k or 60)
        self.hybrid_weights = hybrid_weights
        self.verbose = bool(verbose)

    def forward(
        self,
        question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
    ) -> DSPyAgentRAGResponse:
        weaviate_client = weaviate_client or self.weaviate_client
        reranker_clients = reranker_clients or self.reranker_clients

        # Retrieve
        sources = weaviate_search_tool(
            weaviate_client=weaviate_client,
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"Retrieved {len(sources)} documents")

        # Early return if no rerankers
        if not reranker_clients:
            if self.verbose:
                print("No rerankers provided, returning retrieval order")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=sources[: self.reranked_k],
                searches=[question],
                aggregations=None,
                usage={},
            )

        # Rerank
        docs = [s.content for s in sources]
        items = ce_rank(
            query=question,
            documents=docs,
            top_k=self.reranked_k,
            clients=reranker_clients,
            provider=self.provider,
            model_name_overrides=self.model_name_overrides,
            rrf_k=self.rrf_k,
            hybrid_weights=self.hybrid_weights,
            verbose=self.verbose,
        )

        reranked = reorder(items, sources)
        if self.verbose:
            print(f"Reranked: Returning {len(reranked)} documents")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked,
            searches=[question],
            aggregations=None,
            usage={},
        )

    async def aforward(
        self,
        question: str,
        weaviate_async_client: Optional[weaviate.AsyncWeaviateClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
    ) -> DSPyAgentRAGResponse:
        reranker_clients = reranker_clients or self.reranker_clients

        # Retrieve
        sources = await async_weaviate_search_tool(
            weaviate_async_client=weaviate_async_client,
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"Retrieved {len(sources)} documents (async)")

        # Early return if no rerankers
        if not reranker_clients:
            if self.verbose:
                print("No rerankers provided, returning retrieval order (async)")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=sources[: self.reranked_k],
                searches=[question],
                aggregations=None,
                usage={},
            )

        # Rerank
        docs = [s.content for s in sources]
        items = await async_ce_rank(
            query=question,
            documents=docs,
            top_k=self.reranked_k,
            clients=reranker_clients,
            provider=self.provider,
            model_name_overrides=self.model_name_overrides,
            rrf_k=self.rrf_k,
            hybrid_weights=self.hybrid_weights,
            verbose=self.verbose,
        )

        reranked = reorder(items, sources)
        if self.verbose:
            print(f"Reranked: Returning {len(reranked)} documents (async)")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked,
            searches=[question],
            aggregations=None,
            usage={},
        )

async def main():
    import os
    import cohere
    import weaviate
    
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    cross_encoder_reranker = CrossEncoderReranker(
        collection_name="BrightBiology",
        target_property_name="content",
        weaviate_client=weaviate_client,
        verbose=True,
        search_only=True,
        retrieved_k=50,
        reranked_k=20,
    )
    
    # Test forward() method
    print("Testing forward() method:")
    response = cross_encoder_reranker.forward(
        question="How many cells are in the human body?",
        weaviate_client=weaviate_client,
        reranker_clients=[RerankerClient(name="cohere", client=cohere_client)],
    )
    print(f"Sync successfully returned: {len(response.sources)} documents")

    weaviate_async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    await weaviate_async_client.connect()
    cohere_async_client = cohere.AsyncClientV2(api_key=os.getenv("COHERE_API_KEY"))
    
    # Test aforward() method
    print("\nTesting aforward() method:")
    async_response = await cross_encoder_reranker.aforward(
        question="How many cells are in the human body?",
        weaviate_async_client=weaviate_async_client,
        reranker_clients=[RerankerClient(name="cohere", client=cohere_async_client)],
    )
    print(f"Async successfully returned: {len(async_response.sources)} documents")
    
    await weaviate_async_client.close()
    weaviate_client.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())