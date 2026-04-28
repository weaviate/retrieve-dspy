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
from retrieve_dspy.signatures import WriteSplitSearchQueries


def _dedupe_pool(result_sets: List[List[ObjectFromDB]]) -> List[ObjectFromDB]:
    """Pool documents from multiple result sets, deduplicating by object_id.
    Keeps the first occurrence of each document."""
    doc_map: Dict[str, ObjectFromDB] = {}
    for result_set in result_sets:
        for obj in result_set:
            if obj.object_id not in doc_map:
                doc_map[obj.object_id] = obj
    return list(doc_map.values())


class SplitQueryRetriever(BaseRetriever):
    """Retriever that writes separate queries for BM25 and vector search in a
    single LLM inference, retrieves from each pathway independently, then pools
    the results and sends them to the cross-encoder reranker.

    This tests the hypothesis that BM25 and dense retrieval benefit from
    different query formulations.
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
        pathway_only: Optional[str] = None,
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
        self.pathway_only = pathway_only

        self.write_split_queries = dspy.Predict(WriteSplitSearchQueries)

    def forward(
        self,
        question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
    ) -> DSPyAgentRAGResponse:
        weaviate_client = weaviate_client or self.weaviate_client

        # Single LLM inference → two queries
        prediction = self.write_split_queries(question=question)
        bm25_query = prediction.bm25_search_query
        vector_query = prediction.vector_search_query

        if self.verbose:
            print(f"\033[95mSplit queries from: {question}\033[0m")
            print(f"\033[95m  BM25:   {bm25_query}\033[0m")
            print(f"\033[95m  Vector: {vector_query}\033[0m")

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

        # Per-pathway isolation: return only one pathway's results
        if self.pathway_only == "bm25":
            if self.verbose:
                print(f"\033[96m  pathway_only=bm25 → returning {len(bm25_results)} BM25 docs\033[0m")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=bm25_results,
                searches=[bm25_query],
                aggregations=None,
                usage={},
            )
        elif self.pathway_only == "vector":
            if self.verbose:
                print(f"\033[96m  pathway_only=vector → returning {len(vector_results)} vector docs\033[0m")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=vector_results,
                searches=[vector_query],
                aggregations=None,
                usage={},
            )

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
            retriever="SplitQueryRetriever",
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

        # Single LLM inference → two queries
        prediction = await self.write_split_queries.acall(question=question)
        bm25_query = prediction.bm25_search_query
        vector_query = prediction.vector_search_query

        if self.verbose:
            print(f"\033[95mSplit queries from: {question}\033[0m")
            print(f"\033[95m  BM25:   {bm25_query}\033[0m")
            print(f"\033[95m  Vector: {vector_query}\033[0m")

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

        # Per-pathway isolation: return only one pathway's results
        if self.pathway_only == "bm25":
            if self.verbose:
                print(f"\033[96m  pathway_only=bm25 → returning {len(bm25_results)} BM25 docs\033[0m")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=bm25_results,
                searches=[bm25_query],
                aggregations=None,
                usage={},
            )
        elif self.pathway_only == "vector":
            if self.verbose:
                print(f"\033[96m  pathway_only=vector → returning {len(vector_results)} vector docs\033[0m")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=vector_results,
                searches=[vector_query],
                aggregations=None,
                usage={},
            )

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
            retriever="SplitQueryRetriever",
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
