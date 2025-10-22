from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Literal, Optional

from retrieve_dspy.models import RerankerClient, ObjectFromDB, RerankItem
from retrieve_dspy.retrievers.common.rrf import fuse_rankings_with_rrf

Provider = Literal["cohere", "voyage", "hybrid"]


# Reranker factory functions
def make_cohere_reranker(client: Any, model: str = "rerank-v3.5") -> Callable:
    def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        res = client.rerank(model=model, query=query, documents=documents, top_n=min(top_k, len(documents)))
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]
    return _fn


def make_voyage_reranker(client: Any, model: str = "rerank-2.5") -> Callable:
    def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        res = client.rerank(query=query, documents=documents, model=model, top_k=min(top_k, len(documents)))
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]
    return _fn


def make_async_cohere_reranker(client: Any, model: str = "rerank-v3.5") -> Callable:
    async def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        res = await client.rerank(model=model, query=query, documents=documents, top_n=min(top_k, len(documents)))
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]
    return _fn


def make_async_voyage_reranker(client: Any, model: str = "rerank-2.5") -> Callable:
    async def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        res = await client.rerank(query=query, documents=documents, model=model, top_k=min(top_k, len(documents)))
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]
    return _fn


def _get_model_name(provider: str, overrides: Optional[Dict[str, str]], default: str) -> str:
    return overrides.get(provider, default) if overrides else default


def _make_adapters(
    clients: Optional[List[RerankerClient]],
    overrides: Optional[Dict[str, str]],
) -> Dict[str, Callable]:
    """Create sync reranker functions from clients."""
    adapters = {}
    if not clients:
        return adapters
    
    for rc in clients:
        if callable(rc.client) and not hasattr(rc.client, 'rerank'):
            adapters[rc.name] = rc.client
        elif rc.name == "cohere":
            adapters["cohere"] = make_cohere_reranker(rc.client, _get_model_name("cohere", overrides, "rerank-v3.5"))
        elif rc.name == "voyage":
            adapters["voyage"] = make_voyage_reranker(rc.client, _get_model_name("voyage", overrides, "rerank-2.5"))
    
    return adapters


def _make_async_adapters(
    clients: Optional[List[RerankerClient]],
    overrides: Optional[Dict[str, str]],
) -> Dict[str, Callable]:
    """Create async reranker functions from clients."""
    adapters = {}
    if not clients:
        return adapters
    
    for rc in clients:
        if callable(rc.client) and inspect.iscoroutinefunction(rc.client):
            adapters[rc.name] = rc.client
        elif hasattr(rc.client, 'rerank') and inspect.iscoroutinefunction(rc.client.rerank):
            if rc.name == "cohere":
                adapters["cohere"] = make_async_cohere_reranker(rc.client, _get_model_name("cohere", overrides, "rerank-v3.5"))
            elif rc.name == "voyage":
                adapters["voyage"] = make_async_voyage_reranker(rc.client, _get_model_name("voyage", overrides, "rerank-2.5"))
    
    return adapters


def _pick_provider(requested: Optional[Provider], available: Dict[str, Any]) -> Provider:
    """Auto-select provider based on what's available."""
    has_cohere = "cohere" in available
    has_voyage = "voyage" in available
    
    if requested:
        return requested
    
    if has_cohere and has_voyage:
        return "hybrid"
    return "cohere" if has_cohere else "voyage"


def _rerank_single(provider: str, query: str, docs: List[str], top_k: int, rerankers: Dict) -> List[RerankItem]:
    """Rerank with single provider."""
    return rerankers[provider](query, docs, top_k)


def rerank(
    provider: Provider,
    query: str,
    documents: List[str],
    top_k: int,
    rerankers: Dict[str, Callable],
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
) -> List[RerankItem]:
    """Sync reranking."""
    if provider in ("cohere", "voyage"):
        return _rerank_single(provider, query, documents, top_k, rerankers)
    
    # Hybrid mode
    results = {}
    for p in ("cohere", "voyage"):
        if p in rerankers:
            try:
                results[p] = rerankers[p](query, documents, top_k)
            except Exception:
                results[p] = []
    
    return fuse_rankings_with_rrf(results, top_k, rrf_k=rrf_k, weights=hybrid_weights)


async def async_rerank(
    provider: Provider,
    query: str,
    documents: List[str],
    top_k: int,
    async_rerankers: Optional[Dict[str, Callable]] = None,
    rerankers: Optional[Dict[str, Callable]] = None,
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
) -> List[RerankItem]:
    """Async reranking."""
    async_rerankers = async_rerankers or {}
    rerankers = rerankers or {}
    
    async def _run(p: str) -> List[RerankItem]:
        if p in async_rerankers:
            return await async_rerankers[p](query, documents, top_k)
        if p in rerankers:
            return await asyncio.to_thread(rerankers[p], query, documents, top_k)
        return []
    
    if provider in ("cohere", "voyage"):
        return await _run(provider)
    
    # Hybrid mode
    co_items, vo_items = await asyncio.gather(_run("cohere"), _run("voyage"))
    return fuse_rankings_with_rrf({"cohere": co_items, "voyage": vo_items}, top_k, rrf_k=rrf_k, weights=hybrid_weights)


def ce_rank(
    query: str,
    documents: List[str],
    top_k: int,
    clients: Optional[List[RerankerClient]] = None,
    provider: Optional[Provider] = None,
    model_name_overrides: Optional[Dict[str, str]] = None,
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> List[RerankItem]:
    """Sync rerank documents."""
    adapters = _make_adapters(clients, model_name_overrides)
    eff_provider = _pick_provider(provider, adapters)
    return rerank(eff_provider, query, documents, top_k, adapters, rrf_k, hybrid_weights)


async def async_ce_rank(
    query: str,
    documents: List[str],
    top_k: int,
    clients: Optional[List[RerankerClient]] = None,
    provider: Optional[Provider] = None,
    model_name_overrides: Optional[Dict[str, str]] = None,
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> List[RerankItem]:
    """Async rerank documents."""
    sync_adapters = _make_adapters(clients, model_name_overrides)
    async_adapters = _make_async_adapters(clients, model_name_overrides)
    all_adapters = {**sync_adapters, **async_adapters}
    eff_provider = _pick_provider(provider, all_adapters)
    
    return await async_rerank(
        eff_provider, query, documents, top_k, 
        async_adapters, sync_adapters, rrf_k, hybrid_weights
    )


def reorder(items: List[RerankItem], sources: List[ObjectFromDB]) -> List[ObjectFromDB]:
    """Reorder sources and update ranks/scores."""
    out = []
    for new_rank, item in enumerate(items, start=1):
        if 0 <= item.index < len(sources):
            orig = sources[item.index]
            out.append(ObjectFromDB(
                object_id=orig.object_id,
                content=orig.content,
                relevance_rank=new_rank,
                relevance_score=item.relevance_score,
                vector=orig.vector,
                source_query=orig.source_query,
            ))
    return out