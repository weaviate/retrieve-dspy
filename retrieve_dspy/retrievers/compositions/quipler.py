from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Optional, List, Dict, Any
import weaviate

from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.signatures import VerboseWriteSearchQueries, WriteSearchQueries
from retrieve_dspy.retrievers.common.rrf import reciprocal_rank_fusion
from retrieve_dspy.retrievers import CrossEncoderReranker
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB, RerankerClient

import dspy


class QUIPLER(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
        return_property_name: Optional[str] = None,
        verbose: bool = False,
        verbose_signature: bool = True,
        search_only: bool = True,
        retrieved_k: int = 50,
        reranked_k: int = 20,
        rrf_k: int = 60,
        **cross_encoder_kwargs
    ):
        super().__init__(
            weaviate_client=weaviate_client,
            collection_name=collection_name,
            target_property_name=target_property_name,
            verbose=verbose,
            verbose_signature=verbose_signature,
            search_only=search_only,
            retrieved_k=retrieved_k,
        )
        
        # Query generation
        if self.verbose_signature:
            self.query_writer = dspy.ChainOfThought(VerboseWriteSearchQueries)
        else:
            self.query_writer = dspy.Predict(WriteSearchQueries)
        
        self.rrf_k = rrf_k
        self.reranked_k = reranked_k
        
        # Cross-encoder reranker
        self.searcher = CrossEncoderReranker(
            collection_name=collection_name,
            target_property_name=target_property_name,
            weaviate_client=weaviate_client,
            reranker_clients=reranker_clients,
            return_property_name=return_property_name,
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k,
            reranked_k=reranked_k,
            **cross_encoder_kwargs
        )

    def forward(
        self,
        question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
    ) -> DSPyAgentRAGResponse:
        """Synchronous version with sequential execution."""
        if self.verbose:
            print(f"\033[95m[QUIPLER] Starting sync query expansion for: '{question}'\033[0m")
        
        # Generate multiple search queries
        query_result = self.query_writer(question=question)
        queries = query_result.search_queries
        
        if self.verbose:
            print(f"\033[95m[QUIPLER] Generated {len(queries)} queries for sequential execution:\033[0m")
            for i, q in enumerate(queries, 1):
                print(f"  {i}. {q}")
        
        # Run searches sequentially
        return self._run_sequential(queries, question, weaviate_client, reranker_clients)

    def _run_parallel_sync(
        self, 
        queries: List[str], 
        original_question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
    ) -> DSPyAgentRAGResponse:
        """Run searches in parallel from sync context."""
        
        async def parallel_search_async():
            # We'll need async clients for true parallelism
            weaviate_async_client = None
            
            # Try to set up async weaviate client
            if weaviate_client:
                try:
                    from retrieve_dspy.clients import get_and_connect_weaviate_async_client
                    weaviate_async_client = await get_and_connect_weaviate_async_client()
                except Exception as e:
                    if self.verbose:
                        print(f"\033[93m[QUIPLER] Could not create async weaviate client: {e}\033[0m")
                    # Fall back to sync execution
                    raise e
            
            # Search with all queries in parallel
            async def search_single_query(query: str, index: int) -> tuple[int, DSPyAgentRAGResponse]:
                if self.verbose:
                    print(f"\033[96m[QUIPLER] Starting parallel search {index+1}/{len(queries)}\033[0m")
                
                result = await self.searcher.aforward(
                    question=query,
                    weaviate_async_client=weaviate_async_client,
                    reranker_clients=reranker_clients
                )
                
                if self.verbose:
                    print(f"\033[96m[QUIPLER] Completed parallel search {index+1}/{len(queries)}\033[0m")
                
                return index, result
            
            # Execute all searches in parallel
            search_tasks = [
                search_single_query(query, i) 
                for i, query in enumerate(queries)
            ]
            
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Clean up async client
            if weaviate_async_client:
                await weaviate_async_client.close()
            
            return search_results
        
        # Handle event loop scenarios
        try:
            # Check if we're already in an async context
            current_loop = asyncio.get_running_loop()
            
            # If we're in an async context, we need to run in a thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(parallel_search_async()))
                search_results = future.result()
                
        except RuntimeError:
            # No event loop running, we can use asyncio.run() directly
            search_results = asyncio.run(parallel_search_async())
        
        return self._process_search_results(search_results, queries, original_question, weaviate_client, reranker_clients)

    def _run_sequential(
        self, 
        queries: List[str], 
        original_question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
    ) -> DSPyAgentRAGResponse:
        """Fallback to sequential execution."""
        if self.verbose:
            print(f"\033[95m[QUIPLER] Running {len(queries)} searches sequentially\033[0m")
        
        all_results: List[List[ObjectFromDB]] = []
        all_searches: List[str] = []
        
        for i, query in enumerate(queries):
            if self.verbose:
                print(f"\033[96m[QUIPLER] Searching with query {i+1}/{len(queries)}\033[0m")
            
            try:
                result = self.searcher.forward(
                    question=query,
                    weaviate_client=weaviate_client,
                    reranker_clients=reranker_clients
                )
                all_results.append(result.sources)
                all_searches.extend(result.searches)
            except Exception as e:
                if self.verbose:
                    print(f"\033[91m[QUIPLER] Search {i+1} failed: {e}\033[0m")
                continue
        
        if not all_results:
            if self.verbose:
                print(f"\033[91m[QUIPLER] All searches failed, using single search fallback\033[0m")
            return self.searcher.forward(
                question=original_question,
                weaviate_client=weaviate_client,
                reranker_clients=reranker_clients
            )
        
        # Aggregate results using RRF
        fused_sources = reciprocal_rank_fusion(
            result_sets=all_results,
            k=self.rrf_k,
            top_k=self.reranked_k
        )
        
        if self.verbose:
            print(f"\033[95m[QUIPLER] Final result: {len(fused_sources)} documents\033[0m")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=fused_sources,
            searches=all_searches,
            aggregations=None,
            usage={},
        )

    def _process_search_results(
        self, 
        search_results: List, 
        queries: List[str], 
        original_question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
    ) -> DSPyAgentRAGResponse:
        """Process parallel search results and fuse them."""
        all_results: List[List[ObjectFromDB]] = []
        all_searches: List[str] = []
        successful_searches = 0
        
        for result in search_results:
            if isinstance(result, Exception):
                if self.verbose:
                    print(f"\033[91m[QUIPLER] Search failed: {result}\033[0m")
                continue
            
            index, response = result
            all_results.append(response.sources)
            all_searches.extend(response.searches)
            successful_searches += 1
        
        if successful_searches == 0:
            if self.verbose:
                print(f"\033[91m[QUIPLER] All parallel searches failed, using single search fallback\033[0m")
            return self.searcher.forward(
                question=original_question,
                weaviate_client=weaviate_client,
                reranker_clients=reranker_clients
            )
        
        # Aggregate results using RRF
        if self.verbose:
            print(f"\033[95m[QUIPLER] Fusing results from {successful_searches}/{len(queries)} successful searches\033[0m")
        
        fused_sources = reciprocal_rank_fusion(
            result_sets=all_results,
            k=self.rrf_k,
            top_k=self.reranked_k
        )
        
        if self.verbose:
            print(f"\033[95m[QUIPLER] Final parallel result: {len(fused_sources)} documents\033[0m")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=fused_sources,
            searches=all_searches,
            aggregations=None,
            usage={},
        )

    async def aforward(
        self,
        question: str,
        weaviate_async_client: Optional[weaviate.AsyncWeaviateClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
    ) -> DSPyAgentRAGResponse:
        """Asynchronous version - generates queries and searches in parallel."""
        if self.verbose:
            print(f"\033[95m[QUIPLER] Starting async query expansion for: '{question}'\033[0m")
        
        # Generate multiple search queries
        query_result = self.query_writer(question=question)
        queries = query_result.search_queries
        
        if self.verbose:
            print(f"\033[95m[QUIPLER] Generated {len(queries)} queries for parallel search:\033[0m")
            for i, q in enumerate(queries, 1):
                print(f"  {i}. {q}")
        
        # Search with all queries in parallel
        async def search_single_query(query: str, index: int) -> tuple[int, DSPyAgentRAGResponse]:
            if self.verbose:
                print(f"\033[96m[QUIPLER] Starting parallel search {index+1}/{len(queries)}\033[0m")
            
            result = await self.searcher.aforward(
                question=query,
                weaviate_async_client=weaviate_async_client,
                reranker_clients=reranker_clients
            )
            
            if self.verbose:
                print(f"\033[96m[QUIPLER] Completed parallel search {index+1}/{len(queries)}\033[0m")
            
            return index, result
        
        # Execute all searches in parallel
        search_tasks = [
            search_single_query(query, i) 
            for i, query in enumerate(queries)
        ]
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process results and handle any exceptions
        all_results: List[List[ObjectFromDB]] = []
        all_searches: List[str] = []
        successful_searches = 0
        
        for result in search_results:
            if isinstance(result, Exception):
                if self.verbose:
                    print(f"\033[91m[QUIPLER] Search failed: {result}\033[0m")
                continue
            
            index, response = result
            all_results.append(response.sources)
            all_searches.extend(response.searches)
            successful_searches += 1
        
        if successful_searches == 0:
            if self.verbose:
                print(f"\033[91m[QUIPLER] All searches failed, falling back to original question\033[0m")
            # Fallback to single search with original question
            fallback_result = await self.searcher.aforward(
                question=question,
                weaviate_async_client=weaviate_async_client,
                reranker_clients=reranker_clients
            )
            return fallback_result
        
        # Aggregate results using RRF
        if self.verbose:
            print(f"\033[95m[QUIPLER] Fusing results from {successful_searches}/{len(queries)} successful searches\033[0m")
        
        fused_sources = reciprocal_rank_fusion(
            result_sets=all_results,
            k=self.rrf_k,
            top_k=self.reranked_k
        )
        
        if self.verbose:
            print(f"\033[95m[QUIPLER] Final async result: {len(fused_sources)} documents\033[0m")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=fused_sources,
            searches=all_searches,
            aggregations=None,
            usage={},
        )