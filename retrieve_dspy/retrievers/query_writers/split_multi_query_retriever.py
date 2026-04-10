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
from retrieve_dspy.retrievers.common.truncate_document import truncate_document
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB, RerankerClient
from retrieve_dspy.signatures import WriteSplitSearchQueryLists


def _dedupe_pool(result_sets: List[List[ObjectFromDB]]) -> List[ObjectFromDB]:
    """Pool documents from multiple result sets, deduplicating by object_id."""
    doc_map: Dict[str, ObjectFromDB] = {}
    for result_set in result_sets:
        for obj in result_set:
            if obj.object_id not in doc_map:
                doc_map[obj.object_id] = obj
    return list(doc_map.values())


class SplitMultiQueryRetriever(BaseRetriever):
    """Retriever that writes separate *lists* of queries for BM25 and vector search
    in a single LLM inference, retrieves from each pathway for every query, then
    pools all results and sends them to the cross-encoder reranker.

    Single inference → `bm25_search_queries` (list) + `vector_search_queries` (list).
    Each BM25 query is sent through search_type="bm25", each vector query through
    search_type="vector". All results are pooled, deduped, and reranked.
    """

    def __init__(
        self,
        collection_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        target_property_name: str = "content",
        number_of_queries: int = 3,
        retrieved_k: int = 20,
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
        self.reranker_clients = reranker_clients
        self.reranker_provider = reranker_provider
        self.reranked_k = reranked_k if reranked_k is not None else retrieved_k

        self.write_split_query_lists = dspy.Predict(WriteSplitSearchQueryLists)

    def forward(
        self,
        question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
    ) -> DSPyAgentRAGResponse:
        weaviate_client = weaviate_client or self.weaviate_client

        # Single LLM inference → two lists
        prediction = self.write_split_query_lists(
            question=question,
            number_of_queries=self.number_of_queries,
        )
        bm25_queries = prediction.bm25_search_queries
        vector_queries = prediction.vector_search_queries

        if self.verbose:
            print(f"\033[95mSplit multi-query from: {question}\033[0m")
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

        # Pool + dedupe, then CE rerank
        pooled = _dedupe_pool(result_sets)
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

        # Single LLM inference → two lists
        prediction = await self.write_split_query_lists.acall(
            question=question,
            number_of_queries=self.number_of_queries,
        )
        bm25_queries = prediction.bm25_search_queries
        vector_queries = prediction.vector_search_queries

        if self.verbose:
            print(f"\033[95mSplit multi-query from: {question}\033[0m")
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

        # Pool + dedupe, then CE rerank
        pooled = _dedupe_pool(result_sets)
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

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_results,
            searches=all_queries,
            aggregations=None,
            usage={},
        )
