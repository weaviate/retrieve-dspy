import asyncio
import os
from typing import Optional, List, Literal, Dict, Sequence

import cohere
import voyageai
import dspy

from retrieve_dspy.database.weaviate_database import weaviate_search_tool
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB
from retrieve_dspy.signatures import QuerySummarizer

from retrieve_dspy.retrievers.common.call_ce_ranker import (
    RerankItem,
    rerank as ce_rerank,
    async_rerank as ce_async_rerank,
    make_cohere_reranker,
    make_voyage_reranker,
)

Provider = Literal["cohere", "voyage", "hybrid"]


class CrossEncoderReranker(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        return_property_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 50,
        reranked_k: Optional[int] = 20,
        reranker_provider: Provider = "cohere",
        cohere_model: Optional[str] = "rerank-v3.5",
        voyage_model: Optional[str] = "rerank-2.5",
        cohere_api_key: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
        summarize_query: Optional[bool] = False,
        rrf_k: Optional[int] = 60,
        hybrid_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the Cross Encoder Reranker (now powered by functional adapters from call_ce_ranker).
        """
        super().__init__(
            collection_name=collection_name,
            target_property_name=target_property_name,
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k,
        )
        self.return_property_name = return_property_name
        self.reranked_k = int(reranked_k or 20)
        self.reranker_provider = reranker_provider
        self.cohere_model = cohere_model or "rerank-v3.5"
        self.voyage_model = voyage_model or "rerank-2.5"
        self.summarize_query = summarize_query
        self.query_summarizer = dspy.Predict(QuerySummarizer)
        self.rrf_k = int(rrf_k or 60)
        self.hybrid_weights = hybrid_weights or {"cohere": 0.5, "voyage": 0.5}
        self.verbose = bool(verbose)

        # ---- Build reranker adapters based on provider selection ----
        self._rerankers: Dict[str, callable] = {}
        # (Optional) async adapters; we’ll fall back to to_thread if not provided
        self._async_rerankers: Dict[str, callable] = {}

        need_cohere = reranker_provider in ("cohere", "hybrid")
        need_voyage = reranker_provider in ("voyage", "hybrid")

        if need_cohere:
            co_key = cohere_api_key or os.getenv("COHERE_API_KEY")
            if not co_key:
                raise ValueError("COHERE_API_KEY must be provided or set as environment variable")
            co_client = cohere.ClientV2(co_key)
            self._rerankers["cohere"] = make_cohere_reranker(co_client, self.cohere_model)

        if need_voyage:
            vo_key = voyage_api_key or os.getenv("VOYAGE_API_KEY")
            if not vo_key:
                raise ValueError("VOYAGE_API_KEY must be provided or set as environment variable")
            vo_client = voyageai.Client(api_key=vo_key)
            self._rerankers["voyage"] = make_voyage_reranker(vo_client, self.voyage_model)

        if reranker_provider not in ("cohere", "voyage", "hybrid"):
            raise ValueError(f"Unsupported reranker provider: {reranker_provider}")

    # -------------------------- Internal helpers --------------------------

    def _run_rerank(
        self, provider: Provider, query: str, documents: Sequence[str]
    ) -> List[RerankItem]:
        """Sync rerank via functional API."""
        return ce_rerank(
            provider,
            query,
            documents,
            self.reranked_k,
            rerankers=self._rerankers,
            rrf_k=self.rrf_k,
            hybrid_weights=self.hybrid_weights,
            verbose=self.verbose,
        )

    async def _run_async_rerank(
        self, provider: Provider, query: str, documents: Sequence[str]
    ) -> List[RerankItem]:
        """Async rerank via functional API (falls back to threads if async adapters not set)."""
        return await ce_async_rerank(
            provider,
            query,
            documents,
            self.reranked_k,
            async_rerankers=self._async_rerankers,   # optional; may be empty
            rerankers=self._rerankers,               # fallback for to_thread
            rrf_k=self.rrf_k,
            hybrid_weights=self.hybrid_weights,
            verbose=self.verbose,
        )

    def _verbose_preview_docs(self, documents: Sequence[str], n: int = 3) -> None:
        if not self.verbose:
            return
        print(f"\n\033[93mPreparing {len(documents)} documents for reranking...\033[0m")
        for i, doc in enumerate(documents[:n]):
            preview = (doc[:100] + "...") if len(doc) > 100 else doc
            print(f"  Doc {i+1} preview: {preview}")

    def _reorder_sources_by_items(
        self, items: List[RerankItem], sources: List[ObjectFromDB]
    ) -> List[ObjectFromDB]:
        out: List[ObjectFromDB] = []
        for i, it in enumerate(items):
            if 0 <= it.index < len(sources):
                out.append(sources[it.index])
                if self.verbose and i < 5:
                    print(f"Rank {i+1}: Document {it.index + 1} (score: {it.relevance_score:.4f})")
        return out

    # ---------------------------- Public API -----------------------------

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        """
        Execute retrieval + CE reranking (sync).
        """
        # Initial retrieval
        sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(sources)} documents\033[0m")
            print(f"Query: '{question}'")
            print(f"Using {self.reranker_provider} for reranking")

        documents = [s.content for s in sources]
        self._verbose_preview_docs(documents)

        # Optional query summarization
        if self.summarize_query:
            question_pred = self.query_summarizer(question=question)
            question = question_pred.summary
            if self.verbose:
                print(f"\033[96mSummarized query: {question}\033[0m")

        # Rerank
        items = self._run_rerank(self.reranker_provider, question, documents)

        if self.verbose:
            print(f"\n\033[93m{self.reranker_provider.capitalize()} reranking complete.\033[0m")

        # Reorder sources
        reranked_sources = self._reorder_sources_by_items(items, sources)

        if self.verbose:
            print(f"\n\033[96mReranked: Returning {len(reranked_sources)} documents\033[0m")
            if items and self.reranker_provider != "hybrid" and items[0].relevance_score < 0.1:
                print(
                    f"\033[91mWarning: Low relevance scores detected! "
                    f"Top score: {items[0].relevance_score:.4f}\033[0m"
                )

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage={},
        )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        """
        Execute retrieval + CE reranking (async).
        """
        sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(sources)} documents\033[0m")
            print(f"Query: '{question}'")
            print(f"Using {self.reranker_provider} for reranking (async)")

        documents = [s.content for s in sources]
        self._verbose_preview_docs(documents)

        if self.summarize_query:
            # dspy Predict is sync; keep same behavior
            question_pred = self.query_summarizer(question=question)
            question = question_pred.summary
            if self.verbose:
                print(f"\033[96mSummarized query: {question}\033[0m")

        items = await self._run_async_rerank(self.reranker_provider, question, documents)

        if self.verbose:
            print(f"\n\033[93m{self.reranker_provider.capitalize()} async reranking complete.\033[0m")

        reranked_sources = self._reorder_sources_by_items(items, sources)

        if self.verbose:
            print(f"\n\033[96mReranked: Returning {len(reranked_sources)} documents\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage={},
        )
