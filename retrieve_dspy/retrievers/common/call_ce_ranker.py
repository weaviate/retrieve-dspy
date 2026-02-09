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


def make_dspy_reranker(module: Any, score_type: str = "binary") -> Callable:
    """Create sync DSPy reranker that runs async predictions concurrently.
    
    Args:
        module: DSPy module/predictor to use for scoring
        score_type: "binary" for AssessRelevance (0/1), "numeric" for ScoreRelevance (0.0-1.0)
    """
    def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        import asyncio
        import concurrent.futures
        import logging
        
        logger = logging.getLogger(__name__)
        
        async def score_all():
            # Reasonable semaphore limit - adjust based on your needs
            semaphore = asyncio.Semaphore(20)  # Allow 20 concurrent requests
            
            async def score_doc(idx: int, doc: str) -> RerankItem:
                async with semaphore:
                    try:
                        pred = await module.acall(query=query, candidate_document=doc)
                        
                        if score_type == "numeric":
                            # ScoreRelevance signature - extract numeric score
                            try:
                                score = float(pred.relevance_score)
                                # Clamp to [0, 1]
                                score = max(0.0, min(1.0, score))
                            except (AttributeError, ValueError, TypeError):
                                try:
                                    score = float(pred.get('relevance_score', 0.0))
                                    score = max(0.0, min(1.0, score))
                                except (AttributeError, ValueError, TypeError):
                                    score = 0.0
                        else:
                            # AssessRelevance signature - binary assessment
                            try:
                                assessment = pred.relevance_assessment
                            except AttributeError:
                                try:
                                    assessment = pred.get('relevance_assessment', False)
                                except (AttributeError, TypeError):
                                    assessment = False
                            score = 1.0 if assessment else 0.0
                        
                        return RerankItem(index=idx, relevance_score=score)
                        
                    except Exception as e:
                        # Log once at debug level to avoid spam
                        if idx == 0 or "Too many open files" not in str(e):
                            logger.debug(f"Failed to score document {idx}: {str(e)[:100]}")
                        return RerankItem(index=idx, relevance_score=0.0)
            
            tasks = [score_doc(idx, doc) for idx, doc in enumerate(documents)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert any exceptions to zero-scored items
            return [r if isinstance(r, RerankItem) else RerankItem(index=i, relevance_score=0.0) 
                    for i, r in enumerate(results)]
        
        try:
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, score_all())
                    results = future.result()
            except RuntimeError:
                results = asyncio.run(score_all())
            
            results = list(results)
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Critical error in reranker: {str(e)[:200]}")
            return [RerankItem(index=i, relevance_score=0.0) for i in range(min(top_k, len(documents)))]
    
    return _fn

def make_async_dspy_reranker(module: Any, score_type: str = "binary") -> Callable:
    """Create async DSPy reranker that runs predictions concurrently.
    
    Args:
        module: DSPy module/predictor to use for scoring
        score_type: "binary" for AssessRelevance (0/1), "numeric" for ScoreRelevance (0.0-1.0)
    """
    async def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        semaphore = asyncio.Semaphore(20)  # Limit concurrent requests
        
        async def score_doc(idx: int, doc: str) -> RerankItem:
            async with semaphore:
                try:
                    # Use acall for async execution
                    pred = await module.acall(query=query, candidate_document=doc)
                    
                    if score_type == "numeric":
                        # ScoreRelevance signature - extract numeric score
                        try:
                            score = float(pred.relevance_score)
                            score = max(0.0, min(1.0, score))
                        except (AttributeError, ValueError, TypeError):
                            try:
                                score = float(pred.get('relevance_score', 0.0))
                                score = max(0.0, min(1.0, score))
                            except (AttributeError, ValueError, TypeError):
                                score = 0.0
                    else:
                        # AssessRelevance signature - binary assessment
                        try:
                            assessment = pred.relevance_assessment
                        except AttributeError:
                            try:
                                assessment = pred.get('relevance_assessment', False)
                            except (AttributeError, TypeError):
                                assessment = False
                        score = 1.0 if assessment else 0.0
                    
                    return RerankItem(index=idx, relevance_score=score)
                    
                except Exception:
                    return RerankItem(index=idx, relevance_score=0.0)
        
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


def _detect_dspy_score_type(module: Any) -> str:
    """Detect whether a DSPy module uses binary or numeric scoring based on its signature."""
    # Check the signature's output fields
    try:
        if hasattr(module, 'signature'):
            sig = module.signature
            # Check output field names
            output_fields = getattr(sig, 'output_fields', {})
            if hasattr(output_fields, 'keys'):
                field_names = list(output_fields.keys())
            else:
                # Try to get field names from the signature class
                field_names = [f for f in dir(sig) if not f.startswith('_')]
            
            if 'relevance_score' in field_names:
                return "numeric"
            if 'relevance_assessment' in field_names:
                return "binary"
    except Exception:
        pass
    
    # Default to binary for backward compatibility
    return "binary"


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
            score_type = _detect_dspy_score_type(rc.client)
            adapters["dspy"] = make_dspy_reranker(rc.client, score_type)
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
            score_type = _detect_dspy_score_type(rc.client)
            adapters["dspy"] = make_async_dspy_reranker(rc.client, score_type)
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