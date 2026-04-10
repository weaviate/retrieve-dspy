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
from retrieve_dspy.signatures import WriteSplitSearchQueries


class SplitQueryRetriever(BaseRetriever):
    """Retriever that writes separate queries for BM25 and vector search in a
    single LLM inference, retrieves from each pathway independently, then fuses
    the results with Reciprocal Rank Fusion and optional cross-encoder reranking.

    This tests the hypothesis that BM25 and dense retrieval benefit from
    different query formulations.
    """

    def __init__(
        self,
        collection_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        target_property_name: str = "content",
        retrieved_k: int = 20,
        rrf_k: int = 60,
        # Cross-encoder reranking (optional)
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
        self.rrf_k = rrf_k
        self.use_cross_encoder = use_cross_encoder
        self.reranker_clients = reranker_clients
        self.reranker_provider = reranker_provider
        self.reranked_k = reranked_k if reranked_k is not None else retrieved_k

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

        # Fuse with RRF
        fused = reciprocal_rank_fusion(
            result_sets=[bm25_results, vector_results],
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
        bm25_task = async_weaviate_search_tool(
            query=bm25_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_async_client=weaviate_async_client,
            search_type="bm25",
        )
        vector_task = async_weaviate_search_tool(
            query=vector_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_async_client=weaviate_async_client,
            search_type="vector",
        )
        bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)

        for obj in bm25_results:
            obj.source_query = bm25_query
        for obj in vector_results:
            obj.source_query = vector_query

        if self.verbose:
            print(f"\033[96m  BM25 returned {len(bm25_results)} docs, "
                  f"Vector returned {len(vector_results)} docs\033[0m")

        # Fuse with RRF
        fused = reciprocal_rank_fusion(
            result_sets=[bm25_results, vector_results],
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
            searches=[bm25_query, vector_query],
            aggregations=None,
            usage={},
        )
