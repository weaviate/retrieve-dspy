# cross_encoder_ranker.py
from __future__ import annotations

from typing import Optional, List, Literal, Dict

import weaviate

from retrieve_dspy.database.weaviate_database import weaviate_search_tool
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB, RerankerClient

from retrieve_dspy.retrievers.common.call_ce_ranker import (
    RerankItem,
    ce_rank,
    async_ce_rank,
    reorder,
)

RerankProvider = Literal["cohere", "voyage", "hybrid"]


class CrossEncoderReranker(BaseRAG):
    def __init__(
        self,
        weaviate_client: weaviate.WeaviateClient,
        reranker_clients: List[RerankerClient],
        collection_name: str,
        target_property_name: str,
        return_property_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 50,
        reranked_k: Optional[int] = 20,
        reranker_provider: Optional[RerankProvider] = None,  # None => auto based on clients
        cohere_model: Optional[str] = "rerank-v3.5",
        voyage_model: Optional[str] = "rerank-2.5",
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
        self.reranker_provider = reranker_provider
        self.cohere_model = cohere_model or "rerank-v3.5"
        self.voyage_model = voyage_model or "rerank-2.5"
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
            provider=self.reranker_provider,  # None => auto
            cohere_model=self.cohere_model,
            voyage_model=self.voyage_model,
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
        weaviate_client: weaviate.WeaviateClient,
        reranker_clients: Optional[List[RerankerClient]] = None,
    ) -> DSPyAgentRAGResponse:
        pass
        try:
            sources = weaviate_search_tool(
                weaviate_client=weaviate_client,
                query=question,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                return_property_name=self.return_property_name,
                retrieved_k=self.retrieved_k,
            )
        except TypeError:
            sources = weaviate_search_tool(
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
            provider=self.reranker_provider,
            cohere_model=self.cohere_model,
            voyage_model=self.voyage_model,
            rrf_k=self.rrf_k,
            hybrid_weights=self.hybrid_weights,
            verbose=self.verbose,
        )

        reranked: List[ObjectFromDB] = self._reorder(items, sources, tag="[async]")
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

    cross_encoder_reranker = CrossEncoderReranker(
        collection_name="EnronEmails",
        target_property_name="email_body",
        verbose=True,
        search_only=True,
        retrieved_k=50,
        reranked_k=20,
    )
    response = cross_encoder_reranker.forward(
        question="What are the implications of SBX12?",
        weaviate_client=weaviate_client,
        reranker_clients=[RerankerClient(name="cohere", client=cohere_client)],
    )
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())