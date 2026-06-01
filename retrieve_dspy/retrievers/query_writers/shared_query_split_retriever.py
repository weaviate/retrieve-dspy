"""Condition A with split retrieval: writes ONE hybrid query, sends it to
both BM25 and vector pathways separately, then fuses with RSF.

This is the fair-comparison version of SearchQueryWriter for the split-query
experiments. Unlike SearchQueryWriter (which uses Weaviate's native hybrid
search), this retriever does the BM25 and vector retrievals independently --
matching the architecture of SplitQueryRetriever (Condition B) and
DualInferenceSplitRetriever (Condition C). This eliminates the fusion-method
confound and enables overlap analysis (Table 3) for Condition A.

The key difference from SplitQueryRetriever: the SAME query goes to both
pathways, rather than writing a separate query per pathway.
"""

import asyncio
from typing import Optional, List

import dspy
import weaviate

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool,
)
from retrieve_dspy.retrievers.base_retriever import BaseRetriever
from retrieve_dspy.retrievers.common.rsf import relative_score_fusion
from retrieve_dspy.retrievers.common.dedup_log import log_dedup
from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import WriteHybridSearchQuery, VerboseWriteHybridSearchQuery


class SharedQuerySplitRetriever(BaseRetriever):
    """Writes one hybrid query and retrieves from BM25 + vector separately.

    Condition A implemented with the same split-retrieve-and-fuse architecture
    as Conditions B and C, for fair comparison.
    """

    def __init__(
        self,
        collection_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        target_property_name: str = "content",
        retrieved_k: int = 100,
        reranked_k: Optional[int] = None,
        verbose: bool = False,
        embedding_model: Optional[str] = None,
        pathway_only: Optional[str] = None,
        rsf_alpha: float = 0.5,
    ):
        super().__init__(
            collection_name=collection_name,
            weaviate_client=weaviate_client,
            target_property_name=target_property_name,
            verbose=verbose,
            retrieved_k=retrieved_k,
            embedding_model=embedding_model,
        )
        self.reranked_k = reranked_k if reranked_k is not None else retrieved_k
        self.pathway_only = pathway_only
        self.rsf_alpha = rsf_alpha
        signature = VerboseWriteHybridSearchQuery if self.verbose else WriteHybridSearchQuery
        self.write_search_query = dspy.Predict(signature)

        # Append-only log: each forward/aforward call adds an entry.
        # Safe under async concurrency (unlike overwriting instance attrs).
        self.pathway_results_log = []

    def forward(
        self,
        question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
    ) -> DSPyAgentRAGResponse:
        weaviate_client = weaviate_client or self.weaviate_client

        # One LLM inference -> one hybrid query
        query = self.write_search_query(question=question).search_query

        if self.verbose:
            print(f"\033[95mHybrid query from: {question}\033[0m")
            print(f"\033[95m  Query: {query}\033[0m")

        # Send the SAME query to both pathways
        bm25_results = weaviate_search_tool(
            query=query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_client=weaviate_client,
            search_type="bm25",
            return_score=True,
        )
        for obj in bm25_results:
            obj.source_query = query

        vector_results = weaviate_search_tool(
            query=query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_client=weaviate_client,
            search_type="vector",
            return_score=True,
        )
        for obj in vector_results:
            obj.source_query = query

        # Log for external per-pathway analysis (append-only, async-safe)
        self.pathway_results_log.append({
            "question": question,
            "bm25_ids": [s.object_id for s in bm25_results],
            "vector_ids": [s.object_id for s in vector_results],
        })

        if self.verbose:
            print(f"\033[96m  BM25 returned {len(bm25_results)} docs, "
                  f"Vector returned {len(vector_results)} docs\033[0m")

        # Per-pathway isolation
        if self.pathway_only == "bm25":
            if self.verbose:
                print(f"\033[96m  pathway_only=bm25 -> returning {len(bm25_results)} BM25 docs\033[0m")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=bm25_results,
                searches=[query],
                aggregations=None,
                usage={},
            )
        elif self.pathway_only == "vector":
            if self.verbose:
                print(f"\033[96m  pathway_only=vector -> returning {len(vector_results)} vector docs\033[0m")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=vector_results,
                searches=[query],
                aggregations=None,
                usage={},
            )

        # RSF fusion
        final_results = relative_score_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            alpha=self.rsf_alpha,
            top_k=self.reranked_k,
        )
        if self.verbose:
            print(f"\033[96m  RSF fused to {len(final_results)} docs (alpha={self.rsf_alpha})\033[0m")

        log_dedup(
            retriever="SharedQuerySplitRetriever",
            question=question,
            bm25_results=bm25_results,
            vector_results=vector_results,
            fused_count=len(final_results),
            retrieved_k=self.retrieved_k,
        )

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_results,
            searches=[query],
            aggregations=None,
            usage={},
        )

    async def aforward(
        self,
        question: str,
        weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None,
    ) -> DSPyAgentRAGResponse:
        weaviate_async_client = weaviate_async_client or self.weaviate_client

        # One LLM inference -> one hybrid query
        prediction = await self.write_search_query.acall(question=question)
        query = prediction.search_query

        if self.verbose:
            print(f"\033[95mHybrid query from: {question}\033[0m")
            print(f"\033[95m  Query: {query}\033[0m")

        # Send the SAME query to both pathways concurrently
        bm25_results, vector_results = await asyncio.gather(
            async_weaviate_search_tool(
                query=query,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k,
                weaviate_async_client=weaviate_async_client,
                search_type="bm25",
                return_score=True,
            ),
            async_weaviate_search_tool(
                query=query,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k,
                weaviate_async_client=weaviate_async_client,
                search_type="vector",
                return_score=True,
            ),
        )

        for obj in bm25_results:
            obj.source_query = query
        for obj in vector_results:
            obj.source_query = query

        # Log for external per-pathway analysis (append-only, async-safe)
        self.pathway_results_log.append({
            "question": question,
            "bm25_ids": [s.object_id for s in bm25_results],
            "vector_ids": [s.object_id for s in vector_results],
        })

        if self.verbose:
            print(f"\033[96m  BM25 returned {len(bm25_results)} docs, "
                  f"Vector returned {len(vector_results)} docs\033[0m")

        # Per-pathway isolation
        if self.pathway_only == "bm25":
            if self.verbose:
                print(f"\033[96m  pathway_only=bm25 -> returning {len(bm25_results)} BM25 docs\033[0m")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=bm25_results,
                searches=[query],
                aggregations=None,
                usage={},
            )
        elif self.pathway_only == "vector":
            if self.verbose:
                print(f"\033[96m  pathway_only=vector -> returning {len(vector_results)} vector docs\033[0m")
            return DSPyAgentRAGResponse(
                final_answer="",
                sources=vector_results,
                searches=[query],
                aggregations=None,
                usage={},
            )

        # RSF fusion
        final_results = relative_score_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            alpha=self.rsf_alpha,
            top_k=self.reranked_k,
        )
        if self.verbose:
            print(f"\033[96m  RSF fused to {len(final_results)} docs (alpha={self.rsf_alpha})\033[0m")

        log_dedup(
            retriever="SharedQuerySplitRetriever",
            question=question,
            bm25_results=bm25_results,
            vector_results=vector_results,
            fused_count=len(final_results),
            retrieved_k=self.retrieved_k,
        )

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_results,
            searches=[query],
            aggregations=None,
            usage={},
        )
