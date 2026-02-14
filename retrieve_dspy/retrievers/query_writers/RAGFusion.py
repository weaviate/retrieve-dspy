import asyncio
from typing import Optional, List, Literal, Dict, Callable

import dspy
import weaviate

from retrieve_dspy.retrievers.common.rrf import reciprocal_rank_fusion
from retrieve_dspy.retrievers.common.call_ce_ranker import (
    ce_rank, 
    reorder,
    _make_adapters,
    _make_async_adapters,
    _pick_provider,
)
from retrieve_dspy.database.weaviate_database import weaviate_search_tool, async_weaviate_search_tool
from retrieve_dspy.retrievers.base_retriever import BaseRetriever
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB, RerankerClient, RerankItem
from retrieve_dspy.signatures import WriteSearchQueries, VerboseWriteSearchQueries


def interleave_results(result_sets: List[List[ObjectFromDB]], top_k: int) -> List[ObjectFromDB]:
    """
    Interleave results from multiple queries in round-robin fashion.
    
    Order: q1_r1, q2_r1, q3_r1, ..., q1_r2, q2_r2, q3_r2, ...
    
    Deduplicates by object_id, keeping the first occurrence.
    """
    interleaved = []
    seen_ids = set()
    
    # Find the max number of results across all queries
    max_results = max(len(rs) for rs in result_sets) if result_sets else 0
    
    for rank_idx in range(max_results):
        for query_idx, result_set in enumerate(result_sets):
            if rank_idx < len(result_set):
                obj = result_set[rank_idx]
                if obj.object_id not in seen_ids:
                    seen_ids.add(obj.object_id)
                    interleaved.append(obj)
                    if len(interleaved) >= top_k:
                        return interleaved
    
    return interleaved


def _dedupe_pool(result_sets: List[List[ObjectFromDB]]) -> List[ObjectFromDB]:
    """
    Pool documents from multiple result sets, deduplicating by object_id.
    Keeps the first occurrence of each document.
    """
    doc_map: Dict[str, ObjectFromDB] = {}
    for result_set in result_sets:
        for obj in result_set:
            if obj.object_id not in doc_map:
                doc_map[obj.object_id] = obj
    return list(doc_map.values())


def _extract_scores_from_rerank(rerank_items: List[RerankItem], num_docs: int) -> List[float]:
    """
    Convert RerankItem list back to scores indexed by original document position.
    """
    scores = [0.0] * num_docs
    for item in rerank_items:
        if 0 <= item.index < num_docs:
            scores[item.index] = item.relevance_score
    return scores


def cross_encoder_gated_pooling(
    original_query: str,
    sub_queries: List[str],
    result_sets: List[List[ObjectFromDB]],
    rerankers: Dict[str, Callable],
    provider: str,
    top_k: int,
    alpha: float = 0.7,
    aggregation: Literal["max", "mean"] = "max",
    verbose: bool = False,
) -> List[ObjectFromDB]:
    """
    Cross-Encoder–Gated Pooling for Decomposed Retrieval.
    
    Pools documents retrieved from multiple decomposed sub-queries while preserving:
    - Facet coverage from query decomposition
    - Global relevance to the original query
    
    Args:
        original_query: The original user query (q₀)
        sub_queries: Decomposed sub-queries [{q₁, …, qₙ}]
        result_sets: Retrieved document sets for each query
        rerankers: Dictionary of reranker functions
        provider: Which reranker provider to use
        top_k: Number of top documents to return
        alpha: Weight for original query relevance (default 0.7)
               score(d) = α · s₀(d) + (1 − α) · s_sub(d)
        aggregation: How to aggregate sub-query scores ("max" or "mean")
        verbose: Print debug information
    
    Returns:
        List of ObjectFromDB sorted by final gated score
    """
    # Step 1: Candidate Pooling - dedupe across all result sets
    pooled_docs = _dedupe_pool(result_sets)
    
    if not pooled_docs:
        return []
    
    documents = [obj.content for obj in pooled_docs]
    num_docs = len(documents)
    
    if verbose:
        print(f"CE-Gated Pooling: {num_docs} unique documents from {len(result_sets)} queries")
    
    # Step 2a: Score against original query - s₀(d) = CE(q₀, d)
    original_rerank = rerankers[provider](original_query, documents, num_docs)
    s0_scores = _extract_scores_from_rerank(original_rerank, num_docs)
    
    if verbose:
        print(f"  Original query scores computed (provider: {provider})")
    
    # Step 2b: Score against each sub-query - s_sub(d) = max_i CE(qᵢ, d) or mean
    # Collect scores for each document across all sub-queries
    sub_query_scores: List[List[float]] = []
    
    for i, sub_query in enumerate(sub_queries):
        sub_rerank = rerankers[provider](sub_query, documents, num_docs)
        sub_scores = _extract_scores_from_rerank(sub_rerank, num_docs)
        sub_query_scores.append(sub_scores)
        
        if verbose:
            print(f"  Sub-query {i+1}/{len(sub_queries)} scored: '{sub_query[:50]}...'")
    
    # Compute s_sub for each document
    s_sub_scores = []
    for doc_idx in range(num_docs):
        doc_sub_scores = [sub_query_scores[q_idx][doc_idx] for q_idx in range(len(sub_queries))]
        
        if aggregation == "max":
            s_sub = max(doc_sub_scores) if doc_sub_scores else 0.0
        else:  # mean
            s_sub = sum(doc_sub_scores) / len(doc_sub_scores) if doc_sub_scores else 0.0
        
        s_sub_scores.append(s_sub)
    
    # Step 3: Final Scoring - score(d) = α · s₀(d) + (1 − α) · s_sub(d)
    final_scores = []
    for doc_idx in range(num_docs):
        final_score = alpha * s0_scores[doc_idx] + (1 - alpha) * s_sub_scores[doc_idx]
        final_scores.append((doc_idx, final_score, s0_scores[doc_idx], s_sub_scores[doc_idx]))
    
    # Step 4: Ranking - sort by final score descending
    final_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Build output list
    results = []
    for new_rank, (doc_idx, final_score, s0, s_sub) in enumerate(final_scores[:top_k], start=1):
        obj = pooled_docs[doc_idx]
        results.append(ObjectFromDB(
            object_id=obj.object_id,
            content=obj.content,
            relevance_rank=new_rank,
            relevance_score=final_score,
            vector=obj.vector,
            source_query=obj.source_query,
        ))
    
    if verbose:
        print(f"  Final ranking: {len(results)} documents (α={alpha}, aggregation={aggregation})")
        if results:
            top_doc = final_scores[0]
            print(f"  Top doc: score={top_doc[1]:.4f} (s₀={top_doc[2]:.4f}, s_sub={top_doc[3]:.4f})")
    
    return results


async def async_cross_encoder_gated_pooling(
    original_query: str,
    sub_queries: List[str],
    result_sets: List[List[ObjectFromDB]],
    async_rerankers: Dict[str, Callable],
    sync_rerankers: Dict[str, Callable],
    provider: str,
    top_k: int,
    alpha: float = 0.7,
    aggregation: Literal["max", "mean"] = "max",
    verbose: bool = False,
) -> List[ObjectFromDB]:
    """
    Async version of Cross-Encoder–Gated Pooling for Decomposed Retrieval.
    
    Runs all cross-encoder scoring concurrently for better performance.
    """
    # Step 1: Candidate Pooling
    pooled_docs = _dedupe_pool(result_sets)
    
    if not pooled_docs:
        return []
    
    documents = [obj.content for obj in pooled_docs]
    num_docs = len(documents)
    
    if verbose:
        print(f"CE-Gated Pooling (async): {num_docs} unique documents from {len(result_sets)} queries")
    
    async def run_rerank(query: str) -> List[float]:
        """Run reranking for a single query, using async or sync reranker."""
        if provider in async_rerankers:
            rerank_items = await async_rerankers[provider](query, documents, num_docs)
        elif provider in sync_rerankers:
            rerank_items = await asyncio.to_thread(
                sync_rerankers[provider], query, documents, num_docs
            )
        else:
            return [0.0] * num_docs
        return _extract_scores_from_rerank(rerank_items, num_docs)
    
    # Step 2: Score all queries concurrently
    all_queries = [original_query] + sub_queries
    
    score_tasks = [run_rerank(q) for q in all_queries]
    all_scores = await asyncio.gather(*score_tasks)
    
    # Extract original and sub-query scores
    s0_scores = all_scores[0]
    sub_query_scores = all_scores[1:]
    
    if verbose:
        print(f"  All {len(all_queries)} queries scored concurrently (provider: {provider})")
    
    # Compute s_sub for each document
    s_sub_scores = []
    for doc_idx in range(num_docs):
        doc_sub_scores = [sub_query_scores[q_idx][doc_idx] for q_idx in range(len(sub_queries))]
        
        if aggregation == "max":
            s_sub = max(doc_sub_scores) if doc_sub_scores else 0.0
        else:  # mean
            s_sub = sum(doc_sub_scores) / len(doc_sub_scores) if doc_sub_scores else 0.0
        
        s_sub_scores.append(s_sub)
    
    # Step 3: Final Scoring
    final_scores = []
    for doc_idx in range(num_docs):
        final_score = alpha * s0_scores[doc_idx] + (1 - alpha) * s_sub_scores[doc_idx]
        final_scores.append((doc_idx, final_score, s0_scores[doc_idx], s_sub_scores[doc_idx]))
    
    # Step 4: Ranking
    final_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Build output list
    results = []
    for new_rank, (doc_idx, final_score, s0, s_sub) in enumerate(final_scores[:top_k], start=1):
        obj = pooled_docs[doc_idx]
        results.append(ObjectFromDB(
            object_id=obj.object_id,
            content=obj.content,
            relevance_rank=new_rank,
            relevance_score=final_score,
            vector=obj.vector,
            source_query=obj.source_query,
        ))
    
    if verbose:
        print(f"  Final ranking: {len(results)} documents (α={alpha}, aggregation={aggregation})")
        if results:
            top_doc = final_scores[0]
            print(f"  Top doc: score={top_doc[1]:.4f} (s₀={top_doc[2]:.4f}, s_sub={top_doc[3]:.4f})")
    
    return results


class RAGFusion(BaseRetriever):
    def __init__(
        self,
        collection_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        target_property_name: str = "content",
        number_of_queries: int = 5,  # Number of query variations to generate
        retrieved_k: int = 10,  # Results per query
        reranked_k: int = 200,
        rrf_k: int = 60,  # RRF constant
        # Fusion strategy: "rrf", "interleave", or "ce_gated"
        fusion_strategy: Literal["rrf", "interleave", "ce_gated"] = "rrf",
        use_rrf: bool = True,  # Deprecated: use fusion_strategy instead
        # Cross-encoder reranking options (for post-fusion reranking)
        use_cross_encoder: bool = False,
        reranker_clients: Optional[List[RerankerClient]] = None,
        reranker_provider: Optional[Literal["cohere", "voyage", "dspy", "hybrid"]] = None,
        ce_rerank_k: Optional[int] = None,  # Final k after CE reranking (defaults to reranked_k)
        # CE-Gated Pooling specific options
        ce_gated_alpha: float = 0.7,  # Weight for original query relevance
        ce_gated_aggregation: Literal["max", "mean"] = "max",  # How to aggregate sub-query scores
        verbose: Optional[bool] = False,
        verbose_signature: Optional[bool] = True
    ):
        """
        RAG Fusion retriever with multiple fusion strategies.
        
        Args:
            collection_name: Weaviate collection to search
            weaviate_client: Weaviate client instance
            target_property_name: Property to search within
            number_of_queries: Number of query variations to generate
            retrieved_k: Results to retrieve per query
            reranked_k: Number of results after fusion
            rrf_k: RRF constant (for "rrf" strategy)
            fusion_strategy: How to fuse results from multiple queries:
                - "rrf": Reciprocal Rank Fusion (default)
                - "interleave": Round-robin interleaving
                - "ce_gated": Cross-Encoder Gated Pooling (requires reranker_clients)
            use_rrf: Deprecated, use fusion_strategy instead
            use_cross_encoder: Apply cross-encoder reranking after fusion
            reranker_clients: List of RerankerClient for cross-encoder operations
            reranker_provider: Which provider to use for reranking
            ce_rerank_k: Final k after CE reranking
            ce_gated_alpha: For CE-Gated strategy - weight for original query (0-1)
                           score(d) = α · s₀(d) + (1 − α) · s_sub(d)
            ce_gated_aggregation: For CE-Gated strategy - "max" or "mean" for sub-query scores
            verbose: Print debug information
            verbose_signature: Use verbose query generation signature
        """
        super().__init__(
            collection_name=collection_name,
            weaviate_client=weaviate_client,
            target_property_name=target_property_name,
            verbose=verbose,
        )
        self.number_of_queries = number_of_queries

        self.retrieved_k = retrieved_k
        self.reranked_k = reranked_k
        self.rrf_k = rrf_k
        
        # Handle fusion strategy (backwards compatibility with use_rrf)
        if fusion_strategy == "rrf":
            self.fusion_strategy = "rrf" if use_rrf else "interleave"
        else:
            self.fusion_strategy = fusion_strategy
        self.use_rrf = self.fusion_strategy == "rrf"  # Deprecated but kept for compatibility
        
        # Cross-encoder config
        self.use_cross_encoder = use_cross_encoder
        self.reranker_clients = reranker_clients
        self.reranker_provider = reranker_provider
        self.ce_rerank_k = ce_rerank_k if ce_rerank_k is not None else reranked_k
        
        # CE-Gated Pooling config
        self.ce_gated_alpha = ce_gated_alpha
        self.ce_gated_aggregation = ce_gated_aggregation
        
        if verbose_signature:
            self.decompose_query = dspy.Predict(VerboseWriteSearchQueries)
        else:
            self.decompose_query = dspy.Predict(WriteSearchQueries)
    
    def forward(self, question: str, weaviate_client: Optional[weaviate.WeaviateClient] = None) -> DSPyAgentRAGResponse:
        # Generate query variations
        if weaviate_client is None:
            weaviate_client = self.weaviate_client

        search_queries_response = self.decompose_query(
            question=question,
            number_of_queries=self.number_of_queries
        )
        search_queries = search_queries_response.search_queries
        
        # Add original query if not already included
        if question not in search_queries:
            search_queries = [question] + search_queries
        
        if self.verbose:
            print(f"Search queries ({len(search_queries)}): {search_queries}")
        
        # Retrieve results for each query
        result_sets: List[List[ObjectFromDB]] = []
        for i, query in enumerate(search_queries):
            results = weaviate_search_tool(
                weaviate_client=weaviate_client,
                query=query,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k,
            )
            
            # Tag results with source query for debugging
            for obj in results:
                obj.source_query = query
            
            result_sets.append(results)
        
        # Combine results using the configured fusion strategy
        if self.fusion_strategy == "ce_gated":
            # Cross-Encoder Gated Pooling
            if not self.reranker_clients:
                raise ValueError("CE-Gated pooling requires reranker_clients to be configured")
            
            # Build reranker adapters
            adapters = _make_adapters(self.reranker_clients, None)
            provider = _pick_provider(self.reranker_provider, adapters)
            
            # Separate original query from sub-queries for CE-gated scoring
            original_query = question
            sub_queries = [q for q in search_queries if q != question]
            
            fused_results = cross_encoder_gated_pooling(
                original_query=original_query,
                sub_queries=sub_queries,
                result_sets=result_sets,
                rerankers=adapters,
                provider=provider,
                top_k=self.reranked_k,
                alpha=self.ce_gated_alpha,
                aggregation=self.ce_gated_aggregation,
                verbose=self.verbose,
            )
            fusion_method = f"CE-Gated (α={self.ce_gated_alpha}, {self.ce_gated_aggregation})"
            
        elif self.fusion_strategy == "rrf":
            fused_results = reciprocal_rank_fusion(
                result_sets=result_sets,
                k=self.rrf_k,
                top_k=self.reranked_k
            )
            fusion_method = "RRF"
        else:  # interleave
            fused_results = interleave_results(
                result_sets=result_sets,
                top_k=self.reranked_k
            )
            fusion_method = "interleaved"

        if self.verbose:
            print(f"Fused {len(fused_results)} unique documents from {sum(len(rs) for rs in result_sets)} total ({fusion_method})")
        
        # Optional cross-encoder reranking (post-fusion, for RRF and interleave strategies)
        final_results = fused_results
        if self.use_cross_encoder and self.reranker_clients and self.fusion_strategy != "ce_gated":
            documents = [obj.content for obj in fused_results]
            rerank_items = ce_rank(
                query=question,
                documents=documents,
                top_k=self.ce_rerank_k,
                clients=self.reranker_clients,
                provider=self.reranker_provider,
                verbose=self.verbose,
            )
            final_results = reorder(rerank_items, fused_results)
            
            if self.verbose:
                print(f"Cross-encoder reranked to {len(final_results)} documents (provider: {self.reranker_provider or 'auto'})")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_results,
            searches=search_queries,
            aggregations=None,
            usage={},
        )

    async def aforward(self, question: str, weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None) -> DSPyAgentRAGResponse:
        """Async version of forward using WeaviateAsyncClient."""
        if weaviate_async_client is None:
            weaviate_async_client = self.weaviate_client

        # Generate query variations asynchronously
        search_queries_response = await self.decompose_query.acall(
            question=question,
            number_of_queries=self.number_of_queries
        )
        search_queries = search_queries_response.search_queries
        
        # Add original query if not already included
        if question not in search_queries:
            search_queries = [question] + search_queries
        
        if self.verbose:
            print(f"Search queries ({len(search_queries)}): {search_queries}")
        
        # Retrieve results for each query concurrently
        search_tasks = [
            async_weaviate_search_tool(
                weaviate_async_client=weaviate_async_client,
                query=query,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k,
            )
            for query in search_queries
        ]
        
        search_results = await asyncio.gather(*search_tasks)
        
        # Tag results with source query for debugging
        result_sets: List[List[ObjectFromDB]] = []
        for query, results in zip(search_queries, search_results):
            for obj in results:
                obj.source_query = query
            result_sets.append(results)
        
        # Combine results using the configured fusion strategy
        if self.fusion_strategy == "ce_gated":
            # Cross-Encoder Gated Pooling (async)
            if not self.reranker_clients:
                raise ValueError("CE-Gated pooling requires reranker_clients to be configured")
            
            # Build reranker adapters
            sync_adapters = _make_adapters(self.reranker_clients, None)
            async_adapters = _make_async_adapters(self.reranker_clients, None)
            all_adapters = {**sync_adapters, **async_adapters}
            provider = _pick_provider(self.reranker_provider, all_adapters)
            
            # Separate original query from sub-queries for CE-gated scoring
            original_query = question
            sub_queries = [q for q in search_queries if q != question]
            
            fused_results = await async_cross_encoder_gated_pooling(
                original_query=original_query,
                sub_queries=sub_queries,
                result_sets=result_sets,
                async_rerankers=async_adapters,
                sync_rerankers=sync_adapters,
                provider=provider,
                top_k=self.reranked_k,
                alpha=self.ce_gated_alpha,
                aggregation=self.ce_gated_aggregation,
                verbose=self.verbose,
            )
            fusion_method = f"CE-Gated (α={self.ce_gated_alpha}, {self.ce_gated_aggregation})"
            
        elif self.fusion_strategy == "rrf":
            fused_results = reciprocal_rank_fusion(
                result_sets=result_sets,
                k=self.rrf_k,
                top_k=self.reranked_k
            )
            fusion_method = "RRF"
        else:  # interleave
            fused_results = interleave_results(
                result_sets=result_sets,
                top_k=self.reranked_k
            )
            fusion_method = "interleaved"

        if self.verbose:
            print(f"Fused {len(fused_results)} unique documents from {sum(len(rs) for rs in result_sets)} total ({fusion_method})")
        
        # Optional cross-encoder reranking (post-fusion, for RRF and interleave strategies)
        final_results = fused_results
        if self.use_cross_encoder and self.reranker_clients and self.fusion_strategy != "ce_gated":
            documents = [obj.content for obj in fused_results]
            rerank_items = ce_rank(
                query=question,
                documents=documents,
                top_k=self.ce_rerank_k,
                clients=self.reranker_clients,
                provider=self.reranker_provider,
                verbose=self.verbose,
            )
            final_results = reorder(rerank_items, fused_results)
            
            if self.verbose:
                print(f"Cross-encoder reranked to {len(final_results)} documents (provider: {self.reranker_provider or 'auto'})")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_results,
            searches=search_queries,
            aggregations=None,
            usage={},
        )


if __name__ == "__main__":
    import os
    
    # Initialize Weaviate client
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
    )
    
    # Test query
    test_question = "What are the implications of SBX12?"
    
    # Example 1: Standard RAG Fusion with RRF
    print("=" * 60)
    print("Example 1: RAG Fusion with RRF")
    print("=" * 60)
    
    rag_fusion_rrf = RAGFusion(
        collection_name="EnronEmails",
        weaviate_client=weaviate_client,
        target_property_name="email_body",
        verbose=True,
        verbose_signature=True,
        fusion_strategy="rrf",
        retrieved_k=20,
        reranked_k=20,
        rrf_k=60
    )
    
    try:
        response = rag_fusion_rrf.forward(test_question)
        
        print(f"\nGenerated {len(response.searches)} search queries:")
        for i, query in enumerate(response.searches, 1):
            print(f"  {i}. {query}")
        
        print(f"\nFound {len(response.sources)} sources after fusion:")
        for i, source in enumerate(response.sources[:3], 1):
            print(f"  {i}. (Score: {source.relevance_score:.4f}) {source.content[:100]}...")
            if source.source_query:
                print(f"     Source query: {source.source_query}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    # Example 2: RAG Fusion with CE-Gated Pooling
    # Requires a reranker client (e.g., Cohere or Voyage)
    print("\n" + "=" * 60)
    print("Example 2: RAG Fusion with CE-Gated Pooling")
    print("=" * 60)
    
    try:
        import cohere
        cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
        reranker_clients = [RerankerClient(name="cohere", client=cohere_client)]
        
        rag_fusion_ce_gated = RAGFusion(
            collection_name="EnronEmails",
            weaviate_client=weaviate_client,
            target_property_name="email_body",
            verbose=True,
            verbose_signature=True,
            fusion_strategy="ce_gated",
            retrieved_k=20,
            reranked_k=10,
            reranker_clients=reranker_clients,
            reranker_provider="cohere",
            # CE-Gated specific options
            ce_gated_alpha=0.7,  # 70% weight on original query relevance
            ce_gated_aggregation="max",  # Use max for sub-query scores
        )
        
        response = rag_fusion_ce_gated.forward(test_question)
        
        print(f"\nGenerated {len(response.searches)} search queries:")
        for i, query in enumerate(response.searches, 1):
            print(f"  {i}. {query}")
        
        print(f"\nFound {len(response.sources)} sources after CE-Gated pooling:")
        for i, source in enumerate(response.sources[:3], 1):
            print(f"  {i}. (Score: {source.relevance_score:.4f}) {source.content[:100]}...")
            if source.source_query:
                print(f"     Source query: {source.source_query}")
                
    except ImportError:
        print("Cohere not installed. Skipping CE-Gated example.")
    except Exception as e:
        print(f"Error during CE-Gated testing: {e}")
    
    weaviate_client.close()