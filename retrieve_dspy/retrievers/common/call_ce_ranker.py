from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Literal, Optional

from retrieve_dspy.models import RerankerClient, ObjectFromDB, RerankItem

Provider = Literal["cohere", "voyage", "dspy", "hybrid"]


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


def make_dspy_reranker(module: Any) -> Callable:
    """Create sync DSPy reranker that runs async predictions concurrently."""
    def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        import asyncio
        import concurrent.futures
        
        async def score_all():
            async def score_doc(idx: int, doc: str) -> RerankItem:
                pred = await module.acall(query=query, candidate_document=doc)
                
                try:
                    assessment = pred.relevance_assessment
                except AttributeError:
                    try:
                        assessment = pred.get('relevance_assessment', False)
                    except (AttributeError, TypeError):
                        assessment = False
                
                score = 1.0 if assessment else 0.0
                return RerankItem(index=idx, relevance_score=score)
            
            tasks = [score_doc(idx, doc) for idx, doc in enumerate(documents)]
            return await asyncio.gather(*tasks)
        
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # Already in async context - run in thread pool to avoid nested loop
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, score_all())
                results = future.result()
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            results = asyncio.run(score_all())
        
        results = list(results)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]
    return _fn


def make_async_dspy_reranker(module: Any) -> Callable:
    """Create async DSPy reranker that runs predictions concurrently."""
    async def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        async def score_doc(idx: int, doc: str) -> RerankItem:
            # Use acall for async execution
            pred = await module.acall(query=query, candidate_document=doc)
            
            # Extract relevance assessment
            try:
                assessment = pred.relevance_assessment
            except AttributeError:
                try:
                    assessment = pred.get('relevance_assessment', False)
                except (AttributeError, TypeError):
                    assessment = False
            
            score = 1.0 if assessment else 0.0
            return RerankItem(index=idx, relevance_score=score)
        
        # Score all documents concurrently
        tasks = [score_doc(idx, doc) for idx, doc in enumerate(documents)]
        results = await asyncio.gather(*tasks)
        
        # Sort by score and return top_k
        results = list(results)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]
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
        if rc.name == "cohere":
            adapters["cohere"] = make_cohere_reranker(rc.client, _get_model_name("cohere", overrides, "rerank-v3.5"))
        elif rc.name == "voyage":
            adapters["voyage"] = make_voyage_reranker(rc.client, _get_model_name("voyage", overrides, "rerank-2.5"))
        elif rc.name == "dspy":
            adapters["dspy"] = make_dspy_reranker(rc.client)
        elif callable(rc.client) and not hasattr(rc.client, 'rerank'):
            # Custom callable reranker (already wrapped)
            adapters[rc.name] = rc.client
    
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
        if rc.name == "dspy" and hasattr(rc.client, 'acall'):
            adapters["dspy"] = make_async_dspy_reranker(rc.client)
        elif callable(rc.client) and inspect.iscoroutinefunction(rc.client):
            # Custom async callable reranker (already wrapped)
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
    has_dspy = "dspy" in available
    
    if requested:
        return requested
    
    # Auto-select: if multiple providers, use hybrid
    provider_count = sum([has_cohere, has_voyage, has_dspy])
    if provider_count > 1:
        return "hybrid"
    
    # Single provider
    if has_cohere:
        return "cohere"
    if has_voyage:
        return "voyage"
    if has_dspy:
        return "dspy"
    
    return "cohere"  # Fallback


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
    from retrieve_dspy.retrievers.common.rrf import fuse_rrf
    
    if provider in ("cohere", "voyage", "dspy"):
        return _rerank_single(provider, query, documents, top_k, rerankers)
    
    # Hybrid mode - run all available providers
    results = {}
    for p in ("cohere", "voyage", "dspy"):
        if p in rerankers:
            try:
                results[p] = rerankers[p](query, documents, top_k)
            except Exception:
                results[p] = []
    
    return fuse_rrf(results, top_k, rrf_k=rrf_k, weights=hybrid_weights)


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
    from retrieve_dspy.retrievers.common.rrf import fuse_rrf
    
    async_rerankers = async_rerankers or {}
    rerankers = rerankers or {}
    
    async def _run(p: str) -> List[RerankItem]:
        if p in async_rerankers:
            return await async_rerankers[p](query, documents, top_k)
        if p in rerankers:
            return await asyncio.to_thread(rerankers[p], query, documents, top_k)
        return []
    
    if provider in ("cohere", "voyage", "dspy"):
        return await _run(provider)
    
    # Hybrid mode - run all available providers concurrently
    tasks = {p: asyncio.create_task(_run(p)) for p in ("cohere", "voyage", "dspy")}
    results = {p: await task for p, task in tasks.items()}
    
    return fuse_rrf(results, top_k, rrf_k=rrf_k, weights=hybrid_weights)


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