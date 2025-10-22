from typing import Optional, List, Dict

import weaviate
import dspy
            
from retrieve_dspy.database.weaviate_database import weaviate_search_tool, async_weaviate_search_tool
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB, RerankerClient
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.retrievers.common.call_ce_ranker import ce_rank, async_ce_rank, reorder, Provider
from retrieve_dspy.signatures import AssessRelevance

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
        use_dspy_reranker: Optional[bool] = False,
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
        
        # DSPy cross-encoder as internal module
        self.use_dspy_reranker = use_dspy_reranker
        if use_dspy_reranker:
            self.cross_encoder = dspy.Predict(AssessRelevance)
        else:
            self.cross_encoder = None

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
        if not reranker_clients and not self.cross_encoder:
            if self.verbose:
                print("No rerankers provided, returning retrieval order")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=sources[: self.reranked_k],
                searches=[question],
                aggregations=None,
                usage={},
            )

        # Build reranker clients list (include DSPy if enabled)
        all_clients = list(reranker_clients) if reranker_clients else []
        if self.cross_encoder:
            all_clients.append(RerankerClient(name="dspy", client=self.cross_encoder))

        # Rerank
        docs = [s.content for s in sources]
        items = ce_rank(
            query=question,
            documents=docs,
            top_k=self.reranked_k,
            clients=all_clients,
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
        weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
    ) -> DSPyAgentRAGResponse:
        weaviate_async_client = weaviate_async_client or self.weaviate_async_client
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
        if not reranker_clients and not self.cross_encoder:
            if self.verbose:
                print("No rerankers provided, returning retrieval order (async)")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=sources[: self.reranked_k],
                searches=[question],
                aggregations=None,
                usage={},
            )

        # Build reranker clients list (include DSPy if enabled)
        all_clients = list(reranker_clients) if reranker_clients else []
        if self.cross_encoder:
            all_clients.append(RerankerClient(name="dspy", client=self.cross_encoder))

        # Rerank
        docs = [s.content for s in sources]
        items = await async_ce_rank(
            query=question,
            documents=docs,
            top_k=self.reranked_k,
            clients=all_clients,
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
    import voyageai
    import weaviate
    import time

    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    test_query = "Why are fearful stimuli more powerful at night? For example, horror movies appear to be scarier when viewed at night than during broad day light. Does light have any role in this phenomenon? Are there changes in hormones at night versus during the day that makes fear stronger?"

    # Test 1: Cohere only
    print("=" * 80)
    print("Test 1: Cohere only")
    print("=" * 80)
    reranker = CrossEncoderReranker(
        collection_name="BrightBiology",
        target_property_name="content",
        weaviate_client=weaviate_client,
        verbose=True,
        retrieved_k=50,
        reranked_k=20,
    )
    start = time.time()
    response_cohere = reranker.forward(
        question=test_query,
        weaviate_client=weaviate_client,
        reranker_clients=[RerankerClient(name="cohere", client=cohere_client)],
    )
    elapsed = time.time() - start
    print(f"✓ Cohere returned: {len(response_cohere.sources)} documents")
    print(f"  Top score: {response_cohere.sources[0].relevance_score:.4f}")
    print(f"  Time taken: {elapsed:.2f} seconds")

    # Test 2: Voyage only
    print("\n" + "=" * 80)
    print("Test 2: Voyage only")
    print("=" * 80)
    start = time.time()
    response_voyage = reranker.forward(
        question=test_query,
        weaviate_client=weaviate_client,
        reranker_clients=[RerankerClient(name="voyage", client=voyage_client)],
    )
    elapsed = time.time() - start
    print(f"✓ Voyage returned: {len(response_voyage.sources)} documents")
    print(f"  Top score: {response_voyage.sources[0].relevance_score:.4f}")
    print(f"  Time taken: {elapsed:.2f} seconds")

    # Test 3: DSPy only (internal module)
    print("\n" + "=" * 80)
    print("Test 3: DSPy LLM-based cross encoder only (internal module)")
    print("=" * 80)
    reranker_with_dspy = CrossEncoderReranker(
        collection_name="BrightBiology",
        target_property_name="content",
        weaviate_client=weaviate_client,
        verbose=True,
        retrieved_k=50,
        reranked_k=20,
        use_dspy_reranker=True,  # Enable internal DSPy cross-encoder
    )
    start = time.time()
    response_dspy = reranker_with_dspy.forward(
        question=test_query,
        weaviate_client=weaviate_client,
    )
    elapsed = time.time() - start
    print(f"✓ DSPy returned: {len(response_dspy.sources)} documents")
    print(f"  Top score: {response_dspy.sources[0].relevance_score:.4f}")
    print(f"  Time taken: {elapsed:.2f} seconds")

    # Test 4: Hybrid mode (all three together)
    print("\n" + "=" * 80)
    print("Test 4: Hybrid mode (Cohere + Voyage + DSPy with RRF)")
    print("=" * 80)
    start = time.time()
    response_hybrid = reranker_with_dspy.forward(
        question=test_query,
        weaviate_client=weaviate_client,
        reranker_clients=[
            RerankerClient(name="cohere", client=cohere_client),
            RerankerClient(name="voyage", client=voyage_client),
        ],
        # DSPy is automatically included from self.cross_encoder
    )
    elapsed = time.time() - start
    print(f"✓ Hybrid returned: {len(response_hybrid.sources)} documents")
    print(f"  Top score: {response_hybrid.sources[0].relevance_score:.4f}")
    print(f"  Time taken: {elapsed:.2f} seconds")

    # Test 5: Async with all modes
    print("\n" + "=" * 80)
    print("Test 5: Async tests")
    print("=" * 80)

    weaviate_async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    await weaviate_async_client.connect()
    cohere_async_client = cohere.AsyncClientV2(api_key=os.getenv("COHERE_API_KEY"))
    voyage_async_client = voyageai.AsyncClient(api_key=os.getenv("VOYAGE_API_KEY"))

    # Async Cohere
    print("\n  Async Cohere:")
    start = time.time()
    async_response_cohere = await reranker.aforward(
        question=test_query,
        weaviate_async_client=weaviate_async_client,
        reranker_clients=[RerankerClient(name="cohere", client=cohere_async_client)],
    )
    elapsed = time.time() - start
    print(f"  ✓ Async Cohere returned: {len(async_response_cohere.sources)} documents")
    print(f"  Time taken: {elapsed:.2f} seconds")

    # Async Voyage
    print("\n  Async Voyage:")
    start = time.time()
    async_response_voyage = await reranker.aforward(
        question=test_query,
        weaviate_async_client=weaviate_async_client,
        reranker_clients=[RerankerClient(name="voyage", client=voyage_async_client)],
    )
    elapsed = time.time() - start
    print(f"  ✓ Async Voyage returned: {len(async_response_voyage.sources)} documents")
    print(f"  Time taken: {elapsed:.2f} seconds")

    # Async DSPy
    print("\n  Async DSPy:")
    start = time.time()
    async_response_dspy = await reranker_with_dspy.aforward(
        question=test_query,
        weaviate_async_client=weaviate_async_client,
    )
    elapsed = time.time() - start
    print(f"  ✓ Async DSPy returned: {len(async_response_dspy.sources)} documents")
    print(f"  Time taken: {elapsed:.2f} seconds")

    # Async Hybrid (all three)
    print("\n  Async Hybrid (all three):")
    start = time.time()
    async_response_hybrid = await reranker_with_dspy.aforward(
        question=test_query,
        weaviate_async_client=weaviate_async_client,
        reranker_clients=[
            RerankerClient(name="cohere", client=cohere_async_client),
            RerankerClient(name="voyage", client=voyage_async_client),
        ],
    )
    elapsed = time.time() - start
    print(f"  ✓ Async Hybrid returned: {len(async_response_hybrid.sources)} documents")
    print(f"  Time taken: {elapsed:.2f} seconds")

    await weaviate_async_client.close()
    weaviate_client.close()

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())