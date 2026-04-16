import asyncio
from typing import Optional, List, Dict

import dspy
import weaviate

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool,
)
from retrieve_dspy.retrievers.base_retriever import BaseRetriever
from retrieve_dspy.retrievers.common.call_ce_ranker import ce_rank, async_ce_rank, reorder
from retrieve_dspy.retrievers.common.rrf import reciprocal_rank_fusion
from retrieve_dspy.retrievers.common.dedup_log import log_dedup
from retrieve_dspy.retrievers.common.truncate_document import truncate_document
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB, RerankerClient
from retrieve_dspy.signatures import WriteSearchQuery, WriteVectorSearchQuery


def _dedupe_pool(result_sets: List[List[ObjectFromDB]]) -> List[ObjectFromDB]:
    """Pool documents from multiple result sets, deduplicating by object_id."""
    doc_map: Dict[str, ObjectFromDB] = {}
    for result_set in result_sets:
        for obj in result_set:
            if obj.object_id not in doc_map:
                doc_map[obj.object_id] = obj
    return list(doc_map.values())


class DualInferenceSplitRetriever(BaseRetriever):
    """Retriever that uses two separate LLM inferences to produce one BM25-optimized
    query and one vector-optimized query, retrieves from each pathway independently,
    then pools results and sends them to the cross-encoder reranker.

    Unlike SplitQueryRetriever (single inference, two outputs), this uses two
    independent dspy.Predict calls — one with WriteSearchQuery (BM25) and one
    with WriteVectorSearchQuery (vector). This allows each query writer to be
    optimized independently.
    """

    def __init__(
        self,
        collection_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        target_property_name: str = "content",
        retrieved_k: int = 20,
        use_cross_encoder: bool = True,
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
        self.use_cross_encoder = use_cross_encoder
        self.reranker_clients = reranker_clients
        self.reranker_provider = reranker_provider
        self.reranked_k = reranked_k if reranked_k is not None else retrieved_k

        # Two separate predictors — independently optimizable
        self.write_bm25_query = dspy.Predict(WriteSearchQuery)
        self.write_vector_query = dspy.Predict(WriteVectorSearchQuery)

    def forward(
        self,
        question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
    ) -> DSPyAgentRAGResponse:
        weaviate_client = weaviate_client or self.weaviate_client

        # Two separate LLM inferences
        bm25_query = self.write_bm25_query(question=question).search_query
        vector_query = self.write_vector_query(question=question).search_query

        if self.verbose:
            print(f"\033[95mDual-inference split queries from: {question}\033[0m")
            print(f"\033[95m  BM25 (inference 1):   {bm25_query}\033[0m")
            print(f"\033[95m  Vector (inference 2): {vector_query}\033[0m")

        # Retrieve from each pathway
        bm25_results = weaviate_search_tool(
            query=bm25_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_client=weaviate_client,
            search_type="bm25",
        )
        for obj in bm25_results:
            obj.source_query = bm25_query

        vector_results = weaviate_search_tool(
            query=vector_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_client=weaviate_client,
            search_type="vector",
        )
        for obj in vector_results:
            obj.source_query = vector_query

        if self.verbose:
            print(f"\033[96m  BM25 returned {len(bm25_results)} docs, "
                  f"Vector returned {len(vector_results)} docs\033[0m")

        if self.use_cross_encoder and self.reranker_clients:
            # Pool + dedupe, then CE rerank
            pooled = _dedupe_pool([bm25_results, vector_results])
            if self.verbose:
                print(f"\033[96m  Pooled to {len(pooled)} unique docs\033[0m")
            docs = [truncate_document(s.content, 500) for s in pooled]
            items = ce_rank(
                query=question,
                documents=docs,
                top_k=self.reranked_k,
                clients=self.reranker_clients,
                provider=self.reranker_provider,
                verbose=self.verbose,
            )
            final_results = reorder(items, pooled)
            if self.verbose:
                print(f"\033[96m  Reranked to {len(final_results)} docs\033[0m")
        else:
            # RRF fusion only (no reranker)
            final_results = reciprocal_rank_fusion(
                [bm25_results, vector_results],
                top_k=self.reranked_k,
            )
            if self.verbose:
                print(f"\033[96m  RRF fused to {len(final_results)} docs\033[0m")

        log_dedup(
            retriever="DualInferenceSplitRetriever",
            question=question,
            bm25_results=bm25_results,
            vector_results=vector_results,
            fused_count=len(final_results),
            retrieved_k=self.retrieved_k,
        )

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_results,
            searches=[bm25_query, vector_query],
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
            self.write_bm25_query.acall(question=question),
            self.write_vector_query.acall(question=question),
        )
        bm25_query = bm25_pred.search_query
        vector_query = vector_pred.search_query

        if self.verbose:
            print(f"\033[95mDual-inference split queries from: {question}\033[0m")
            print(f"\033[95m  BM25 (inference 1):   {bm25_query}\033[0m")
            print(f"\033[95m  Vector (inference 2): {vector_query}\033[0m")

        # Retrieve from each pathway concurrently
        bm25_results, vector_results = await asyncio.gather(
            async_weaviate_search_tool(
                query=bm25_query,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k,
                weaviate_async_client=weaviate_async_client,
                search_type="bm25",
            ),
            async_weaviate_search_tool(
                query=vector_query,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k,
                weaviate_async_client=weaviate_async_client,
                search_type="vector",
            ),
        )

        for obj in bm25_results:
            obj.source_query = bm25_query
        for obj in vector_results:
            obj.source_query = vector_query

        if self.verbose:
            print(f"\033[96m  BM25 returned {len(bm25_results)} docs, "
                  f"Vector returned {len(vector_results)} docs\033[0m")

        if self.use_cross_encoder and self.reranker_clients:
            # Pool + dedupe, then CE rerank
            pooled = _dedupe_pool([bm25_results, vector_results])
            if self.verbose:
                print(f"\033[96m  Pooled to {len(pooled)} unique docs\033[0m")
            docs = [truncate_document(s.content, 500) for s in pooled]
            items = await async_ce_rank(
                query=question,
                documents=docs,
                top_k=self.reranked_k,
                clients=self.reranker_clients,
                provider=self.reranker_provider,
                verbose=self.verbose,
            )
            final_results = reorder(items, pooled)
            if self.verbose:
                print(f"\033[96m  Reranked to {len(final_results)} docs\033[0m")
        else:
            # RRF fusion only (no reranker)
            final_results = reciprocal_rank_fusion(
                [bm25_results, vector_results],
                top_k=self.reranked_k,
            )
            if self.verbose:
                print(f"\033[96m  RRF fused to {len(final_results)} docs\033[0m")

        log_dedup(
            retriever="DualInferenceSplitRetriever",
            question=question,
            bm25_results=bm25_results,
            vector_results=vector_results,
            fused_count=len(final_results),
            retrieved_k=self.retrieved_k,
        )

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_results,
            searches=[bm25_query, vector_query],
            aggregations=None,
            usage={},
        )
