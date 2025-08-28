from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Literal, Tuple

Provider = Literal["cohere", "voyage", "hybrid"]


@dataclass
class RerankItem:
    """Unified result for CE rerankers (and hybrid)."""
    index: int
    relevance_score: float  # in hybrid, this is the fused RRF score


class CERankerClient:
    """
    Thin wrapper over Cohere/Voyage cross-encoders with a unified API.

    Features:
    - Lazy client init (env or provided keys)
    - Consistent output (RerankItem)
    - Sync and async entrypoints
    - Built-in HYBRID mode via RRF with configurable weights
    """

    def __init__(
        self,
        cohere_model: str = "rerank-v3.5",
        voyage_model: str = "rerank-2.5",
        cohere_api_key: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
        # Hybrid settings
        rrf_k: int = 60,
        hybrid_weights: Optional[Dict[str, float]] = None,  # e.g. {"cohere": 0.6, "voyage": 0.4}
        verbose: bool = False,
    ) -> None:
        self.cohere_model = cohere_model
        self.voyage_model = voyage_model

        self._cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        self._voyage_api_key = voyage_api_key or os.getenv("VOYAGE_API_KEY")

        self.rrf_k = int(rrf_k)
        self.hybrid_weights = hybrid_weights or {"cohere": 0.5, "voyage": 0.5}
        self.verbose = verbose

        self._co_client = None  # type: ignore
        self._vo_client = None  # type: ignore

    # ---------- Lazy SDK clients ----------

    def _cohere(self):
        if self._co_client is None:
            if not self._cohere_api_key:
                raise ValueError("COHERE_API_KEY must be provided or set in env")
            import cohere  # lazy import
            self._co_client = cohere.ClientV2(self._cohere_api_key)
            if self.verbose:
                print("\033[90mInitialized Cohere ClientV2\033[0m")
        return self._co_client

    def _voyage(self):
        if self._vo_client is None:
            if not self._voyage_api_key:
                raise ValueError("VOYAGE_API_KEY must be provided or set in env")
            import voyageai  # lazy import
            self._vo_client = voyageai.Client(api_key=self._voyage_api_key)
            if self.verbose:
                print("\033[90mInitialized Voyage Client\033[0m")
        return self._vo_client

    # ---------- Public API ----------

    def rerank(
        self,
        provider: Provider,
        query: str,
        documents: List[str],
        top_k: int,
    ) -> List[RerankItem]:
        """Synchronous CE rerank call (single provider or hybrid)."""
        if provider == "cohere":
            return self._rerank_with_cohere(query, documents, top_k)
        if provider == "voyage":
            return self._rerank_with_voyage(query, documents, top_k)
        if provider == "hybrid":
            return self._rerank_hybrid(query, documents, top_k)
        raise ValueError(f"Unsupported provider: {provider}")

    async def async_rerank(
        self,
        provider: Provider,
        query: str,
        documents: List[str],
        top_k: int,
    ) -> List[RerankItem]:
        """Asynchronous CE rerank call. Hybrid runs both providers concurrently."""
        if provider in ("cohere", "voyage"):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.rerank, provider, query, documents, top_k)

        if provider == "hybrid":
            # run cohere + voyage concurrently, then fuse
            co_task = self._async_single("cohere", query, documents, top_k)
            vo_task = self._async_single("voyage", query, documents, top_k)
            co_res, vo_res = await asyncio.gather(co_task, vo_task, return_exceptions=True)

            co_items: List[RerankItem] = []
            vo_items: List[RerankItem] = []

            if isinstance(co_res, Exception):
                if self.verbose:
                    print(f"\033[91mCohere async rerank failed: {co_res}\033[0m")
            else:
                co_items = co_res

            if isinstance(vo_res, Exception):
                if self.verbose:
                    print(f"\033[91mVoyage async rerank failed: {vo_res}\033[0m")
            else:
                vo_items = vo_res

            return self._fuse_rrf({"cohere": co_items, "voyage": vo_items}, top_k)

        raise ValueError(f"Unsupported provider: {provider}")

    # ---------- Internal helpers ----------

    async def _async_single(
        self, provider: Literal["cohere", "voyage"], query: str, documents: List[str], top_k: int
    ) -> List[RerankItem]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, provider, query, documents, top_k)

    def _rerank_with_cohere(
        self, query: str, documents: List[str], top_k: int
    ) -> List[RerankItem]:
        try:
            co = self._cohere()
            res = co.rerank(
                model=self.cohere_model,
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
            )
            items = [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]
            if self.verbose:
                print(f"\033[96mCohere returned {len(items)} results\033[0m")
            return items
        except Exception as e:
            if self.verbose:
                print(f"\033[91mCohere rerank error: {e}\033[0m")
            raise

    def _rerank_with_voyage(
        self, query: str, documents: List[str], top_k: int
    ) -> List[RerankItem]:
        try:
            vo = self._voyage()
            res = vo.rerank(
                query=query,
                documents=documents,
                model=self.voyage_model,
                top_k=min(top_k, len(documents)),
            )
            items = [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]
            if self.verbose:
                print(f"\033[96mVoyage returned {len(items)} results\033[0m")
            return items
        except Exception as e:
            if self.verbose:
                print(f"\033[91mVoyage rerank error: {e}\033[0m")
            raise

    def _rerank_hybrid(
        self, query: str, documents: List[str], top_k: int
    ) -> List[RerankItem]:
        """Run both providers (sequentially, sync) and fuse with RRF."""
        co_items: List[RerankItem] = []
        vo_items: List[RerankItem] = []

        try:
            co_items = self._rerank_with_cohere(query, documents, top_k)
        except Exception as e:
            if self.verbose:
                print(f"\033[91mCohere reranking failed: {e}\033[0m")

        try:
            vo_items = self._rerank_with_voyage(query, documents, top_k)
        except Exception as e:
            if self.verbose:
                print(f"\033[91mVoyage reranking failed: {e}\033[0m")

        return self._fuse_rrf({"cohere": co_items, "voyage": vo_items}, top_k)

    def _fuse_rrf(
        self, rankings: Dict[str, List[RerankItem]], top_k: int
    ) -> List[RerankItem]:
        """
        Reciprocal Rank Fusion (RRF) with provider weights.
        We treat the input items as already ranked (descending). We ignore native scores
        and use position-based fusion (robust-in-practice). Weight per provider is applied.
        """
        # If one list is empty, return the other
        if not rankings.get("cohere") and rankings.get("voyage"):
            if self.verbose:
                print("\033[93mUsing only Voyage results (Cohere failed)\033[0m")
            return rankings["voyage"][:top_k]
        if not rankings.get("voyage") and rankings.get("cohere"):
            if self.verbose:
                print("\033[93mUsing only Cohere results (Voyage failed)\033[0m")
            return rankings["cohere"][:top_k]
        if not rankings.get("cohere") and not rankings.get("voyage"):
            raise RuntimeError("Both rerankers failed")

        k = self.rrf_k
        weights = {"cohere": self.hybrid_weights.get("cohere", 0.5),
                   "voyage": self.hybrid_weights.get("voyage", 0.5)}

        scores: Dict[int, float] = {}
        for provider_name, items in rankings.items():
            w = float(weights.get(provider_name, 0.5))
            for rank, it in enumerate(items):
                # Classic RRF: 1 / (k + rank + 1); multiply by provider weight
                contrib = w * (1.0 / (k + rank + 1))
                scores[it.index] = scores.get(it.index, 0.0) + contrib
                if self.verbose and rank < 3:
                    print(f"  {provider_name} rank {rank+1}: doc {it.index}, +{contrib:.4f}")

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        if self.verbose:
            print(f"\n\033[93mRRF Fusion (k={k}, weights={weights})\033[0m")
            for i, (doc_idx, s) in enumerate(fused[:5]):
                print(f"  Final rank {i+1}: doc {doc_idx}, fused={s:.4f}")

        # Convert back to unified items; use fused score as relevance_score
        return [RerankItem(index=idx, relevance_score=score) for idx, score in fused]
