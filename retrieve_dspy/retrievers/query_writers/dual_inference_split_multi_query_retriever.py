import asyncio
from typing import Optional, List

import dspy
import weaviate

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool,
)
from retrieve_dspy.retrievers.base_retriever import BaseRetriever
from retrieve_dspy.retrievers.common.rrf import reciprocal_rank_fusion
from retrieve_dspy.retrievers.common.call_ce_ranker import ce_rank, async_ce_rank, reorder
from retrieve_dspy.retrievers.common.truncate_document import truncate_document
from retrieve_dspy.models import DSPyAgentRAGResponse, RerankerClient
from retrieve_dspy.signatures import WriteBM25SearchQueries, WriteVectorSearchQueries


class DualInferenceSplitMultiQueryRetriever(BaseRetriever):
    """Retriever that uses two separate LLM inferences to produce lists of
    BM25-optimized queries and vector-optimized queries, retrieves from each
    pathway independently, then fuses all results with RRF and optional
    cross-encoder reranking.

    Inference 1: WriteBM25SearchQueries → list of BM25 queries
    Inference 2: WriteVectorSearchQueries → list of vector queries

    Each query writer can be independently optimized with DSPy.
    """

    def __init__(
        self,
        collection_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        target_property_name: str = "content",
        number_of_queries: int = 3,
        retrieved_k: int = 20,
        rrf_k: int = 60,
        use_cross_encoder: bool = False,
        reranker_clients: Optional[List[RerankerClient]] = None,
        reranker_provider: Optional[str] = None,
        reranked_k: Optional[int] = None,
        verbose: bool = False,
        embedding_model: Optional[str] = None,
    ):
        super().__init__(
            collection_name=collection_name,
            weaviate_client=weaviate_client,
            target_property_name=target_property_name,
            verbose=verbose,
            retrieved_k=retrieved_k,
            embedding_model=embedding_model,
        )
        self.number_of_queries = number_of_queries
        self.rrf_k = rrf_k
        self.use_cross_encoder = use_cross_encoder
        self.reranker_clients = reranker_clients
        self.reranker_provider = reranker_provider
        self.reranked_k = reranked_k if reranked_k is not None else retrieved_k

        # Two separate predictors — independently optimizable
        self.write_bm25_queries = dspy.Predict(WriteBM25SearchQueries)
        self.write_vector_queries = dspy.Predict(WriteVectorSearchQueries)

    def forward(
        self,
        question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
    ) -> DSPyAgentRAGResponse:
        weaviate_client = weaviate_client or self.weaviate_client

        # Two separate LLM inferences
        bm25_queries = self.write_bm25_queries(
            question=question,
            number_of_queries=self.number_of_queries,
        ).search_queries

        vector_queries = self.write_vector_queries(
            question=question,
            number_of_queries=self.number_of_queries,
        ).search_queries

        if self.verbose:
            print(f"\033[95mDual-inference split multi-query from: {question}\033[0m")
            print(f"\033[95m  BM25 queries ({len(bm25_queries)}):   {bm25_queries}\033[0m")
            print(f"\033[95m  Vector queries ({len(vector_queries)}): {vector_queries}\033[0m")

        # Retrieve from each pathway
        result_sets = []
        all_searches = []

        for q in bm25_queries:
            results = weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k,
                weaviate_client=weaviate_client,
                search_type="bm25",
            )
            for obj in results:
                obj.source_query = q
            result_sets.append(results)
            all_searches.append(q)

        for q in vector_queries:
            results = weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k,
                weaviate_client=weaviate_client,
                search_type="vector",
            )
            for obj in results:
                obj.source_query = q
            result_sets.append(results)
            all_searches.append(q)

        if self.verbose:
            total = sum(len(rs) for rs in result_sets)
            print(f"\033[96m  Retrieved {total} total docs across {len(result_sets)} queries\033[0m")

        # Fuse with RRF
        fused = reciprocal_rank_fusion(
            result_sets=result_sets,
            k=self.rrf_k,
            top_k=self.reranked_k,
        )

        if self.verbose:
            print(f"\033[96m  Fused to {len(fused)} unique docs (RRF)\033[0m")

        # Optional cross-encoder reranking
        final_results = fused
        if self.use_cross_encoder and self.reranker_clients:
            docs = [truncate_document(s.content, 500) for s in fused]
            items = ce_rank(
                query=question,
                documents=docs,
                top_k=self.reranked_k,
                clients=self.reranker_clients,
                provider=self.reranker_provider,
                verbose=self.verbose,
            )
            final_results = reorder(items, fused)
            if self.verbose:
                print(f"\033[96m  Reranked to {len(final_results)} docs\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_results,
            searches=all_searches,
            aggregations=None,
            usage={},
        )

    async def aforward(
        self,
        question: str,
        weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None,
    ) -> DSPyAgentRAGResponse:
        weaviate_async_client = weaviate_async_client or self.weaviate_client

        # Two separate LLM inferences (concurrent)
        bm25_pred, vector_pred = await asyncio.gather(
            self.write_bm25_queries.acall(
                question=question,
                number_of_queries=self.number_of_queries,
            ),
            self.write_vector_queries.acall(
                question=question,
                number_of_queries=self.number_of_queries,
            ),
        )
        bm25_queries = bm25_pred.search_queries
        vector_queries = vector_pred.search_queries

        if self.verbose:
            print(f"\033[95mDual-inference split multi-query from: {question}\033[0m")
            print(f"\033[95m  BM25 queries ({len(bm25_queries)}):   {bm25_queries}\033[0m")
            print(f"\033[95m  Vector queries ({len(vector_queries)}): {vector_queries}\033[0m")

        # Retrieve from all pathways concurrently
        bm25_tasks = [
            async_weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k,
                weaviate_async_client=weaviate_async_client,
                search_type="bm25",
            )
            for q in bm25_queries
        ]
        vector_tasks = [
            async_weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k,
                weaviate_async_client=weaviate_async_client,
                search_type="vector",
            )
            for q in vector_queries
        ]

        all_results = await asyncio.gather(*bm25_tasks, *vector_tasks)
        all_queries = bm25_queries + vector_queries

        result_sets = []
        for query, results in zip(all_queries, all_results):
            for obj in results:
                obj.source_query = query
            result_sets.append(results)

        if self.verbose:
            total = sum(len(rs) for rs in result_sets)
            print(f"\033[96m  Retrieved {total} total docs across {len(result_sets)} queries\033[0m")

        # Fuse with RRF
        fused = reciprocal_rank_fusion(
            result_sets=result_sets,
            k=self.rrf_k,
            top_k=self.reranked_k,
        )

        if self.verbose:
            print(f"\033[96m  Fused to {len(fused)} unique docs (RRF)\033[0m")

        # Optional cross-encoder reranking
        final_results = fused
        if self.use_cross_encoder and self.reranker_clients:
            docs = [truncate_document(s.content, 500) for s in fused]
            items = await async_ce_rank(
                query=question,
                documents=docs,
                top_k=self.reranked_k,
                clients=self.reranker_clients,
                provider=self.reranker_provider,
                verbose=self.verbose,
            )
            final_results = reorder(items, fused)
            if self.verbose:
                print(f"\033[96m  Reranked to {len(final_results)} docs\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_results,
            searches=all_queries,
            aggregations=None,
            usage={},
        )
