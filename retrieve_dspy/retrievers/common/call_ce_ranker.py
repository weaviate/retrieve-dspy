from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Awaitable,
    Sequence,
    Optional,
    Tuple,
)

Provider = Literal["cohere", "voyage", "hybrid"]


# ----------------------------- Types & Data -----------------------------

@dataclass
class RerankItem:
    """Unified result for CE rerankers (and hybrid)."""
    index: int
    relevance_score: float  # in hybrid, this is the fused RRF score

# A sync reranker: (query, documents, top_k) -> List[RerankItem]
SyncReranker = Callable[[str, Sequence[str], int], List[RerankItem]]

# An async reranker: (query, documents, top_k) -> Awaitable[List[RerankItem]]
AsyncReranker = Callable[[str, Sequence[str], int], Awaitable[List[RerankItem]]]


# ---------------------------- Core Functions ----------------------------

def fuse_rrf(
    rankings: Dict[str, List[RerankItem]],
    top_k: int,
    *,
    rrf_k: int = 60,
    weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> List[RerankItem]:
    """
    Reciprocal Rank Fusion (position-based) with optional provider weights.

    rankings: provider -> ranked list of RerankItem (descending relevance)
    """
    weights = weights or {}
    # Short-circuits if one provider failed/empty
    if not rankings.get("cohere") and rankings.get("voyage"):
        if verbose:
            print("\033[93mUsing only Voyage results (Cohere empty)\033[0m")
        return rankings["voyage"][:top_k]
    if not rankings.get("voyage") and rankings.get("cohere"):
        if verbose:
            print("\033[93mUsing only Cohere results (Voyage empty)\033[0m")
        return rankings["cohere"][:top_k]
    if not rankings.get("cohere") and not rankings.get("voyage"):
        raise RuntimeError("Both rerankers returned no results")

    scores: Dict[int, float] = {}
    for provider_name, items in rankings.items():
        if not items:
            continue
        w = float(weights.get(provider_name, 0.5))
        for rank, it in enumerate(items):
            # Classic RRF score (rank is 0-based): 1 / (k + rank + 1)
            contrib = w * (1.0 / (rrf_k + rank + 1))
            scores[it.index] = scores.get(it.index, 0.0) + contrib
            if verbose and rank < 3:
                print(f"  {provider_name} rank {rank+1}: doc {it.index}, +{contrib:.4f}")

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    if verbose:
        print(f"\n\033[93mRRF Fusion (k={rrf_k}, weights={weights or {'cohere':0.5,'voyage':0.5}})\033[0m")
        for i, (doc_idx, s) in enumerate(fused[:5]):
            print(f"  Final rank {i+1}: doc {doc_idx}, fused={s:.4f}")

    return [RerankItem(index=idx, relevance_score=score) for idx, score in fused]


def rerank(
    provider: Provider,
    query: str,
    documents: Sequence[str],
    top_k: int,
    *,
    rerankers: Dict[str, SyncReranker],
    # hybrid settings
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> List[RerankItem]:
    """
    Synchronous entrypoint. For 'cohere' or 'voyage', uses that reranker.
    For 'hybrid', runs providers sequentially and fuses with RRF.
    """
    if provider in ("cohere", "voyage"):
        if provider not in rerankers:
            raise ValueError(f"Missing sync reranker for provider '{provider}'")
        return rerankers[provider](query, documents, top_k)

    if provider == "hybrid":
        results: Dict[str, List[RerankItem]] = {}
        for p in ("cohere", "voyage"):
            fn = rerankers.get(p)
            if not fn:
                if verbose:
                    print(f"\033[91mNo sync reranker registered for {p}\033[0m")
                results[p] = []
                continue
            try:
                results[p] = fn(query, documents, top_k)
            except Exception as e:
                if verbose:
                    print(f"\033[91m{p.capitalize()} rerank error: {e}\033[0m")
                results[p] = []
        return fuse_rrf(results, top_k, rrf_k=rrf_k, weights=hybrid_weights, verbose=verbose)

    raise ValueError(f"Unsupported provider: {provider}")


async def async_rerank(
    provider: Provider,
    query: str,
    documents: Sequence[str],
    top_k: int,
    *,
    # Either provide async rerankers, or we will wrap sync ones with asyncio.to_thread
    async_rerankers: Optional[Dict[str, AsyncReranker]] = None,
    rerankers: Optional[Dict[str, SyncReranker]] = None,
    # hybrid settings
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> List[RerankItem]:
    """
    Async entrypoint. If an async reranker isn't supplied, we fall back to the
    sync reranker via asyncio.to_thread. Hybrid runs all providers concurrently.
    """
    async_rerankers = async_rerankers or {}

    async def _run_one(p: str) -> List[RerankItem]:
        if p in async_rerankers:
            return await async_rerankers[p](query, documents, top_k)
        if rerankers and p in rerankers:
            return await asyncio.to_thread(rerankers[p], query, documents, top_k)
        if verbose:
            print(f"\033[91mNo reranker registered for {p}\033[0m")
        return []

    if provider in ("cohere", "voyage"):
        return await _run_one(provider)

    if provider == "hybrid":
        co_task = asyncio.create_task(_run_one("cohere"))
        vo_task = asyncio.create_task(_run_one("voyage"))
        co_items, vo_items = await asyncio.gather(co_task, vo_task, return_exceptions=False)
        results = {"cohere": co_items, "voyage": vo_items}
        return fuse_rrf(results, top_k, rrf_k=rrf_k, weights=hybrid_weights, verbose=verbose)

    raise ValueError(f"Unsupported provider: {provider}")


# ---------------------- Optional: Provider Adapters ---------------------

def make_cohere_reranker(client, model: str = "rerank-v3.5") -> SyncReranker:
    """
    Adapter for Cohere's ClientV2 (sync).
    Usage:
        import cohere
        co = cohere.ClientV2(api_key=...)
        cohere_rank = make_cohere_reranker(co, "rerank-v3.5")
    """
    def _fn(query: str, documents: Sequence[str], top_k: int) -> List[RerankItem]:
        res = client.rerank(
            model=model,
            query=query,
            documents=list(documents),
            top_n=min(top_k, len(documents)),
        )
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]
    return _fn


def make_voyage_reranker(client, model: str = "rerank-2.5") -> SyncReranker:
    """
    Adapter for VoyageAI's Client (sync).
    Usage:
        import voyageai
        vo = voyageai.Client(api_key=...)
        voyage_rank = make_voyage_reranker(vo, "rerank-2.5")
    """
    def _fn(query: str, documents: Sequence[str], top_k: int) -> List[RerankItem]:
        res = client.rerank(
            query=query,
            documents=list(documents),
            model=model,
            top_k=min(top_k, len(documents)),
        )
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]
    return _fn


if __name__ == "__main__":
    import os

    import cohere
    import voyageai
    
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    
    rerankers = {
        "cohere": make_cohere_reranker(co, "rerank-v3.5"),
        "voyage": make_voyage_reranker(vo, "rerank-2.5"),
    }
    
    docs = ["a", "b", "c", "d"]
    print(rerank("cohere", "q", docs, 3, rerankers=rerankers))
    print(rerank("voyage", "q", docs, 3, rerankers=rerankers))
    print(rerank("hybrid", "q", docs, 3, rerankers=rerankers,
                 rrf_k=60, hybrid_weights={"cohere":0.6, "voyage":0.4}, verbose=True))
    
    async def main():
        res = await async_rerank("hybrid", "q", docs, 3, rerankers=rerankers, verbose=True)
        print(res)
    
    asyncio.run(main())
