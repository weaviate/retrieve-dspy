from __future__ import annotations

from typing import Optional, List, Dict

import weaviate

from retrieve_dspy.database.weaviate_database import weaviate_search_tool, async_weaviate_search_tool
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB, RerankerClient
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.retrievers.common.call_ce_ranker import (
    RerankItem,
    ce_rank,
    async_ce_rank,
    reorder,
)
from retrieve_dspy.signatures import CrossEncoderReranker


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
        rrf_k: Optional[int] = 60, # Used for Mixture of Cross Encoders
        hybrid_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize CrossEncoderReranker.
        
        Args:
            model_name_overrides: Optional dict mapping provider names to model names.
                Example: {"cohere": "rerank-v4.0", "voyage": "rerank-3.0"}
                If not provided, defaults from call_ce_ranker will be used.
        """
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
        self.rrf_k = int(rrf_k or 60)
        self.hybrid_weights = hybrid_weights
        self.verbose = bool(verbose)

    def forward(
        self,
        question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
    ) -> DSPyAgentRAGResponse:
        if weaviate_client is None:
            weaviate_client = self.weaviate_client
        
        if reranker_clients is None:
            reranker_clients = self.reranker_clients

        sources = weaviate_search_tool(
            weaviate_client=weaviate_client,
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(sources)} documents\033[0m")
            print(f"Query: '{question}'")

        docs: List[str] = [s.content for s in sources]

        if not reranker_clients:
            if self.verbose:
                print("\033[93mNo reranker_clients provided; returning retrieved order\033[0m")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=sources[: self.reranked_k],
                searches=[question],
                aggregations=None,
                usage={},
            )

        items: List[RerankItem] = ce_rank(
            query=question,
            documents=docs,
            top_k=self.reranked_k,
            clients=reranker_clients,
            model_name_overrides=self.model_name_overrides,
            rrf_k=self.rrf_k,
            hybrid_weights=self.hybrid_weights,
            verbose=self.verbose,
        )

        reranked: List[ObjectFromDB] = reorder(items, sources)
        if self.verbose:
            print(f"\n\033[96mReranked: Returning {len(reranked)} documents\033[0m")

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
        sources = await async_weaviate_search_tool(
            weaviate_async_client=weaviate_async_client,
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(sources)} documents\033[0m")
            print(f"Query: '{question}' (async)")

        docs: List[str] = [s.content for s in sources]

        if not reranker_clients:
            if self.verbose:
                print("\033[93mNo reranker_clients provided; returning retrieved order (async)\033[0m")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=sources[: self.reranked_k],
                searches=[question],
                aggregations=None,
                usage={},
            )

        items = await async_ce_rank(
            query=question,
            documents=docs,
            top_k=self.reranked_k,
            clients=reranker_clients,
            model_name_overrides=self.model_name_overrides,
            rrf_k=self.rrf_k,
            hybrid_weights=self.hybrid_weights,
            verbose=self.verbose,
        )

        reranked: List[ObjectFromDB] = reorder(items, sources)
        if self.verbose:
            print(f"\n\033[96mReranked: Returning {len(reranked)} documents\033[0m")

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
    import voyageai
    import weaviate
    
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    # Example with default models
    cross_encoder_reranker = CrossEncoderReranker(
        collection_name="EnronEmails",
        target_property_name="email_body",
        verbose=True,
        search_only=True,
        retrieved_k=50,
        reranked_k=20,
    )
    
    # Example with custom model overrides
    cross_encoder_with_overrides = CrossEncoderReranker(
        collection_name="EnronEmails",
        target_property_name="email_body",
        verbose=True,
        search_only=True,
        retrieved_k=50,
        reranked_k=20,
        model_name_overrides={
            "cohere": "rerank-v3.5",
            "voyage": "rerank-2.5"
        }
    )
    
    # Test forward() method
    print("Testing forward() method:")
    response = cross_encoder_reranker.forward(
        question="What are the implications of SBX12?",
        weaviate_client=weaviate_client,
        reranker_clients=[RerankerClient(name="cohere", client=cohere_client)],
    )
    print(f"\033[92mSync successfully returned: {len(response.sources)} documents\033[0m")

    weaviate_async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    await weaviate_async_client.connect()
    cohere_async_client = cohere.AsyncClientV2(api_key=os.getenv("COHERE_API_KEY"))
    voyage_async_client = voyageai.AsyncClient(api_key=os.getenv("VOYAGE_API_KEY"))
    
    # Test aforward() method
    print("\nTesting aforward() method:")
    async_response = await cross_encoder_reranker.aforward(
        question="What are the implications of SBX12?",
        weaviate_async_client=weaviate_async_client,
        reranker_clients=[
            RerankerClient(name="cohere", client=cohere_async_client),
            RerankerClient(name="voyage", client=voyage_async_client)
        ],
    )
    print(f"\033[92mAsync successfully returned: {len(async_response.sources)} documents\033[0m")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())