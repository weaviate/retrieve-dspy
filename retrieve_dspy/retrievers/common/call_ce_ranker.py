from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

from retrieve_dspy.models import RerankerClient  # Pydantic: name: Literal["cohere","voyage"], client: Any
from retrieve_dspy.models import ObjectFromDB, RerankItem

Provider = Literal["cohere", "voyage", "hybrid"]

# (query, documents, top_k) -> List[RerankItem]
SyncReranker = Callable[[str, List[str], int], List[RerankItem]]
# (query, documents, top_k) -> awaitable List[RerankItem]
AsyncReranker = Callable[[str, List[str], int], "asyncio.Future[List[RerankItem]]"]

def make_cohere_reranker(client: Any, model: str = "rerank-v3.5") -> SyncReranker:
    def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        rerank_call = client.rerank(
            model=model,
            query=query,
            documents=list(documents),
            top_n=min(top_k, len(documents)),
        )
        
        # If the result is a coroutine, we can't handle it in sync context
        if inspect.iscoroutine(rerank_call):
            raise RuntimeError("Cannot use async client in sync reranker. Use async_ce_rank instead.")
        
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in rerank_call.results]
    return _fn

def make_voyage_reranker(client: Any, model: str = "rerank-2.5") -> SyncReranker:
    def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        rerank_call = client.rerank(
            query=query,
            documents=list(documents),
            model=model,
            top_k=min(top_k, len(documents)),
        )
        
        # If the result is a coroutine, we can't handle it in sync context
        if inspect.iscoroutine(rerank_call):
            raise RuntimeError("Cannot use async client in sync reranker. Use async_ce_rank instead.")
        
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in rerank_call.results]
    return _fn

def make_async_cohere_reranker(client: Any, model: str = "rerank-v3.5") -> AsyncReranker:
    async def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        res = await client.rerank(
            model=model,
            query=query,
            documents=list(documents),
            top_n=min(top_k, len(documents)),
        )
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]
    return _fn

def make_async_voyage_reranker(client: Any, model: str = "rerank-2.5") -> AsyncReranker:
    async def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        res = await client.rerank(
            query=query,
            documents=list(documents),
            model=model,
            top_k=min(top_k, len(documents)),
        )
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]
    return _fn

def fuse_rrf(
    rankings: Dict[str, List[RerankItem]],
    top_k: int,
    *,
    rrf_k: int = 60,
    weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> List[RerankItem]:
    weights = weights or {}
    if not rankings.get("cohere") and rankings.get("voyage"):
        return rankings["voyage"][:top_k]
    if not rankings.get("voyage") and rankings.get("cohere"):
        return rankings["cohere"][:top_k]
    if not rankings.get("cohere") and not rankings.get("voyage"):
        raise RuntimeError("Both rerankers returned no results")

    scores: Dict[int, float] = {}
    for name, items in rankings.items():
        if not items:
            continue
        w = float(weights.get(name, 0.5))
        for rank, it in enumerate(items):
            contrib = w * (1.0 / (rrf_k + rank + 1))
            scores[it.index] = scores.get(it.index, 0.0) + contrib
            if verbose and rank < 3:
                print(f"{name} rank {rank+1}: doc {it.index}, +{contrib:.4f}")

    fused_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [RerankItem(index=i, relevance_score=s) for i, s in fused_pairs]

def _single_provider(
    provider: Literal["cohere", "voyage"],
    *,
    rerankers: Dict[str, SyncReranker],
    query: str,
    documents: List[str],
    top_k: int,
) -> List[RerankItem]:
    fn = rerankers.get(provider)
    if not fn:
        raise ValueError(f"Missing reranker for provider '{provider}'")
    return fn(query, documents, top_k)


def rerank(
    provider: Provider,
    query: str,
    documents: List[str],
    top_k: int,
    *,
    rerankers: Dict[str, SyncReranker],
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> List[RerankItem]:
    if provider in ("cohere", "voyage"):
        return _single_provider(provider, rerankers=rerankers, query=query, documents=documents, top_k=top_k)

    if provider == "hybrid":
        results: Dict[str, List[RerankItem]] = {}
        for p in ("cohere", "voyage"):
            fn = rerankers.get(p)
            if not fn:
                results[p] = []
                continue
            try:
                results[p] = fn(query, documents, top_k)
            except Exception as e:
                if verbose:
                    print(f"{p} rerank error: {e}")
                results[p] = []
        return fuse_rrf(results, top_k, rrf_k=rrf_k, weights=hybrid_weights, verbose=verbose)

    raise ValueError(f"Unsupported provider: {provider}")


async def async_rerank(
    provider: Provider,
    query: str,
    documents: List[str],
    top_k: int,
    *,
    async_rerankers: Optional[Dict[str, AsyncReranker]] = None,
    rerankers: Optional[Dict[str, SyncReranker]] = None,
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> List[RerankItem]:
    async_rerankers = async_rerankers or {}

    async def _run(p: str) -> List[RerankItem]:
        # First try async rerankers
        if p in async_rerankers:
            return await async_rerankers[p](query, documents, top_k)
        # Fallback to sync rerankers only if no async version available
        if rerankers and p in rerankers:
            return await asyncio.to_thread(rerankers[p], query, documents, top_k)
        if verbose:
            print(f"No reranker registered for {p}")
        return []

    if provider in ("cohere", "voyage"):
        return await _run(provider)

    if provider == "hybrid":
        co_task = asyncio.create_task(_run("cohere"))
        vo_task = asyncio.create_task(_run("voyage"))
        co_items, vo_items = await asyncio.gather(co_task, vo_task)
        return fuse_rrf({"cohere": co_items, "voyage": vo_items}, top_k, rrf_k=rrf_k, weights=hybrid_weights, verbose=verbose)

    raise ValueError(f"Unsupported provider: {provider}")

def _adapters_from_clients(
    clients: Optional[List[RerankerClient]],
    *,
    cohere_model: str,
    voyage_model: str,
) -> Dict[str, SyncReranker]:
    adapters: Dict[str, SyncReranker] = {}
    if not clients:
        return adapters
    for rc in clients:
        if rc.name == "cohere":
            adapters["cohere"] = make_cohere_reranker(rc.client, cohere_model) if not callable(rc.client) else rc.client  # type: ignore[assignment]
        elif rc.name == "voyage":
            adapters["voyage"] = make_voyage_reranker(rc.client, voyage_model) if not callable(rc.client) else rc.client  # type: ignore[assignment]
        else:
            raise ValueError("RerankerClient.name must be 'cohere' or 'voyage'")
    return adapters

def _async_adapters_from_clients(
    clients: Optional[List[RerankerClient]],
    *,
    cohere_model: str,
    voyage_model: str,
) -> Dict[str, AsyncReranker]:
    adapters: Dict[str, AsyncReranker] = {}
    if not clients:
        return adapters
    for rc in clients:
        if rc.name == "cohere":
            # Check if client.rerank is async by examining if it's a coroutine function
            if hasattr(rc.client, 'rerank') and inspect.iscoroutinefunction(rc.client.rerank):
                adapters["cohere"] = make_async_cohere_reranker(rc.client, cohere_model)
            # If it's already a callable async reranker, use it directly
            elif callable(rc.client) and inspect.iscoroutinefunction(rc.client):
                adapters["cohere"] = rc.client  # type: ignore[assignment]
        elif rc.name == "voyage":
            if hasattr(rc.client, 'rerank') and inspect.iscoroutinefunction(rc.client.rerank):
                adapters["voyage"] = make_async_voyage_reranker(rc.client, voyage_model)
            elif callable(rc.client) and inspect.iscoroutinefunction(rc.client):
                adapters["voyage"] = rc.client  # type: ignore[assignment]
        else:
            raise ValueError("RerankerClient.name must be 'cohere' or 'voyage'")
    return adapters


def _pick_provider(requested: Optional[Provider], available: Dict[str, Any]) -> Provider:
    have_co = "cohere" in available
    have_vo = "voyage" in available
    if requested is None:
        if have_co and have_vo:
            return "hybrid"
        if have_co:
            return "cohere"
        if have_vo:
            return "voyage"
        raise ValueError("No rerankers provided.")
    if requested == "hybrid":
        if have_co and have_vo:
            return "hybrid"
        if have_co:
            return "cohere"
        if have_vo:
            return "voyage"
        raise ValueError("Hybrid requested but no rerankers provided.")
    if requested not in ("cohere", "voyage"):
        raise ValueError(f"Unsupported provider: {requested}")
    if requested not in available:
        raise ValueError(f"Provider '{requested}' requested but not provided.")
    return requested


def ce_rank(
    query: str,
    documents: List[str],
    top_k: int,
    *,
    clients: Optional[List[RerankerClient]] = None,
    provider: Optional[Provider] = None,
    cohere_model: str = "rerank-v3.5",
    voyage_model: str = "rerank-2.5",
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> List[RerankItem]:
    adapters = _adapters_from_clients(clients, cohere_model=cohere_model, voyage_model=voyage_model)
    eff = _pick_provider(provider, adapters)
    return rerank(eff, query, documents, top_k, rerankers=adapters, rrf_k=rrf_k, hybrid_weights=hybrid_weights, verbose=verbose)


async def async_ce_rank(
    query: str,
    documents: List[str],
    top_k: int,
    *,
    clients: Optional[List[RerankerClient]] = None,
    provider: Optional[Provider] = None,
    cohere_model: str = "rerank-v3.5",
    voyage_model: str = "rerank-2.5",
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> List[RerankItem]:
    # Create both sync and async adapters
    sync_adapters = _adapters_from_clients(clients, cohere_model=cohere_model, voyage_model=voyage_model)
    async_adapters = _async_adapters_from_clients(clients, cohere_model=cohere_model, voyage_model=voyage_model)
    
    # Combine available adapters for provider selection
    all_adapters = {**sync_adapters, **async_adapters}
    eff = _pick_provider(provider, all_adapters)
    
    return await async_rerank(
        eff, 
        query, 
        documents, 
        top_k, 
        async_rerankers=async_adapters,
        rerankers=sync_adapters, 
        rrf_k=rrf_k, 
        hybrid_weights=hybrid_weights, 
        verbose=verbose
    )

def reorder(items: List[RerankItem], sources: List[ObjectFromDB]) -> List[ObjectFromDB]:
    out: List[ObjectFromDB] = []
    for i, it in enumerate(items):
        if 0 <= it.index < len(sources):
            out.append(sources[it.index])
    return out