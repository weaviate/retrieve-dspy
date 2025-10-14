import asyncio
import os
from typing import Optional, List, Any, Tuple

import dspy
import weaviate

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse, ListwiseRankedDocument
from retrieve_dspy.signatures import ListwiseRanking, VerboseListwiseRanking


class TopDownPartitioningReranker(BaseRAG):
    """
    Listwise reranker using top-down partitioning with pivot-based selection.
    Parry et al. 2024: https://arxiv.org/pdf/2405.14589
    
    This approach addresses the inefficiencies of sliding window reranking by:
    1. Processing documents top-down instead of bottom-up
    2. Using a pivot element for parallel comparison
    3. Reducing redundant re-scoring of top documents
    
    Algorithm:
    1. Rank the top-w documents and select a pivot at position k (typically w/2)
    2. Compare pivot against remaining documents in parallel batches
    3. Documents ranked above the pivot become candidates for top-k
    4. Recursively refine candidate pool until budget is met or no more candidates
    5. Final ranking of the candidate pool produces the top-k results
    
    Key advantages over sliding window:
    - ~33% fewer inference calls
    - Inherently parallelizable (most inferences can run concurrently)
    - Reduces repeated re-scoring of highly ranked documents
    - Better aligned with list-wise ranker biases (prefers well-ordered lists)
    """
    
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient | weaviate.WeaviateAsyncClient] = None,
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 50,
        window_size: Optional[int] = 10,
        budget: Optional[int] = None,
        use_thinking: Optional[bool] = True,
        ranking_depth: Optional[int] = 100,
    ):
        super().__init__(
            collection_name=collection_name,
            target_property_name=target_property_name,
            search_only=search_only,
            verbose=verbose,
            retrieved_k=retrieved_k
        )
        self.weaviate_client = weaviate_client
        self.window_size = window_size
        self.budget = budget if budget is not None else window_size
        self.ranking_depth = ranking_depth
        self.pivot_position = window_size // 2  # k = w/2 as per paper
        
        if use_thinking:
            if self.verbose:
                self.ranker = dspy.ChainOfThought(VerboseListwiseRanking)
            else:
                self.ranker = dspy.ChainOfThought(ListwiseRanking)
        else:
            if self.verbose:
                self.ranker = dspy.Predict(VerboseListwiseRanking)
            else:
                self.ranker = dspy.Predict(ListwiseRanking)
        
        # Track statistics for efficiency analysis
        self.inference_count = 0
        self.parallel_inference_count = 0
    
    def _extract_document_text(self, doc: Any) -> str:
        """Extract text content from a document object."""
        if hasattr(doc, self.target_property_name):
            content = getattr(doc, self.target_property_name)
        elif isinstance(doc, dict) and self.target_property_name in doc:
            content = doc[self.target_property_name]
        else:
            content = str(doc)
        
        # Truncate long documents for efficiency
        if isinstance(content, str) and len(content) > 1000:
            content = content[:1000] + "..."
        
        return content
    
    def _rank_window(
        self, 
        query: str, 
        window_docs: List[ListwiseRankedDocument],
        window_label: str = ""
    ) -> List[ListwiseRankedDocument]:
        """Rank a single window of documents."""
        if len(window_docs) <= 1:
            return window_docs
        
        self.inference_count += 1
        
        # Extract text content for ranking
        doc_texts = [self._extract_document_text(doc.content) for doc in window_docs]
        
        # Get ranking from LLM
        ranking_response = self.ranker(
            query=query,
            documents=doc_texts,
        )
        
        # Get ranked indices
        ranked_indices = ranking_response.ranked_indices
        
        # Validate indices
        valid_indices = [idx for idx in ranked_indices 
                        if 0 <= idx < len(window_docs)]
        
        # Handle missing indices
        missing = set(range(len(window_docs))) - set(valid_indices)
        valid_indices.extend(sorted(missing))
        
        # Reorder documents
        reranked = [window_docs[idx] for idx in valid_indices[:len(window_docs)]]
        
        if self.verbose:
            original_indices = [window_docs[idx].original_position for idx in valid_indices[:len(window_docs)]]
            print(f"\033[96m{window_label}Ranking: {original_indices}\033[0m")
        
        return reranked
    
    async def _arank_window(
        self, 
        query: str, 
        window_docs: List[ListwiseRankedDocument],
        window_label: str = ""
    ) -> List[ListwiseRankedDocument]:
        """Async version of rank_window."""
        if len(window_docs) <= 1:
            return window_docs
        
        self.inference_count += 1
        
        doc_texts = [self._extract_document_text(doc.content) for doc in window_docs]
        
        ranking_response = await self.ranker.acall(
            query=query,
            documents=doc_texts,
        )
        
        ranked_indices = ranking_response.ranked_indices
        valid_indices = [idx for idx in ranked_indices 
                        if 0 <= idx < len(window_docs)]
        missing = set(range(len(window_docs))) - set(valid_indices)
        valid_indices.extend(sorted(missing))
        
        reranked = [window_docs[idx] for idx in valid_indices[:len(window_docs)]]
        
        if self.verbose:
            original_indices = [window_docs[idx].original_position for idx in valid_indices[:len(window_docs)]]
            print(f"\033[96m{window_label}Ranking: {original_indices}\033[0m")
        
        return reranked
    
    def _compare_with_pivot(
        self,
        query: str,
        pivot: ListwiseRankedDocument,
        batch_docs: List[ListwiseRankedDocument],
        batch_idx: int
    ) -> Tuple[List[ListwiseRankedDocument], List[ListwiseRankedDocument]]:
        """
        Compare a batch of documents against the pivot.
        Returns (documents_above_pivot, documents_below_pivot).
        """
        if len(batch_docs) == 0:
            return [], []
        
        # Create window with pivot at the start (as per paper's suggestion)
        window = [pivot] + batch_docs
        
        if self.verbose:
            print(f"\n\033[95m--- Batch {batch_idx}: Comparing {len(batch_docs)} docs vs pivot (orig pos {pivot.original_position}) ---\033[0m")
        
        # Rank the window
        ranked_window = self._rank_window(
            query, 
            window, 
            window_label=f"Batch {batch_idx} "
        )
        
        # Find pivot position in ranked window
        pivot_rank = next(i for i, doc in enumerate(ranked_window) if doc is pivot)
        
        # Split based on pivot position
        above_pivot = ranked_window[:pivot_rank]
        below_pivot = ranked_window[pivot_rank + 1:]
        
        if self.verbose:
            print(f"\033[96mPivot ranked at position {pivot_rank}, {len(above_pivot)} docs above, {len(below_pivot)} docs below\033[0m")
        
        return above_pivot, below_pivot
    
    async def _acompare_with_pivot(
        self,
        query: str,
        pivot: ListwiseRankedDocument,
        batch_docs: List[ListwiseRankedDocument],
        batch_idx: int
    ) -> Tuple[List[ListwiseRankedDocument], List[ListwiseRankedDocument]]:
        """Async version of compare_with_pivot."""
        if len(batch_docs) == 0:
            return [], []
        
        window = [pivot] + batch_docs
        
        if self.verbose:
            print(f"\n\033[95m--- Batch {batch_idx}: Comparing {len(batch_docs)} docs vs pivot (orig pos {pivot.original_position}) ---\033[0m")
        
        ranked_window = await self._arank_window(
            query, 
            window, 
            window_label=f"Batch {batch_idx} "
        )
        
        pivot_rank = next(i for i, doc in enumerate(ranked_window) if doc is pivot)
        
        above_pivot = ranked_window[:pivot_rank]
        below_pivot = ranked_window[pivot_rank + 1:]
        
        if self.verbose:
            print(f"\033[96mPivot ranked at position {pivot_rank}, {len(above_pivot)} docs above, {len(below_pivot)} docs below\033[0m")
        
        return above_pivot, below_pivot
    
    def _partition_iteration(
        self,
        query: str,
        candidates: List[ListwiseRankedDocument],
        remaining: List[ListwiseRankedDocument],
        iteration: int
    ) -> Tuple[List[ListwiseRankedDocument], List[ListwiseRankedDocument]]:
        """
        Single iteration of the partitioning algorithm.
        Returns (new_candidates, backfill).
        """
        if self.verbose:
            print(f"\n\033[94m{'='*60}\033[0m")
            print(f"\033[94mIteration {iteration}: {len(candidates)} candidates, {len(remaining)} remaining\033[0m")
            print(f"\033[94m{'='*60}\033[0m")
        
        # Step 1: Rank the top window to find pivot
        top_window = candidates[:self.window_size]
        
        if self.verbose:
            print(f"\n\033[95m=== Ranking top {len(top_window)} candidates to find pivot ===\033[0m")
        
        ranked_top = self._rank_window(query, top_window, window_label="Initial ")
        
        # Step 2: Select pivot at position k
        pivot = ranked_top[self.pivot_position] if len(ranked_top) > self.pivot_position else ranked_top[-1]
        
        # Documents above pivot are definitely in top-k
        new_candidates = ranked_top[:self.pivot_position]
        
        # Documents below pivot go to backfill
        backfill = ranked_top[self.pivot_position + 1:]
        
        if self.verbose:
            print(f"\n\033[93mPivot selected: document at original position {pivot.original_position}\033[0m")
            print(f"\033[93m{len(new_candidates)} docs above pivot, {len(backfill)} docs below pivot\033[0m")
        
        # Step 3: Process remaining documents in batches (can be parallelized)
        remaining_candidates = candidates[self.window_size:]
        all_remaining = remaining_candidates + remaining
        
        if len(all_remaining) == 0:
            return new_candidates, backfill
        
        # Create batches of size (window_size - 1) to account for pivot
        batch_size = self.window_size - 1
        batches = [all_remaining[i:i + batch_size] for i in range(0, len(all_remaining), batch_size)]
        
        if self.verbose:
            print(f"\n\033[94mProcessing {len(batches)} batches in parallel (window_size - 1 = {batch_size} docs per batch)\033[0m")
        
        self.parallel_inference_count += len(batches)
        
        # Process each batch (in production, these could run in parallel)
        for batch_idx, batch in enumerate(batches, 1):
            above, below = self._compare_with_pivot(query, pivot, batch, batch_idx)
            new_candidates.extend(above)
            backfill.extend(below)
            
            # Early stopping if we've reached budget
            if len(new_candidates) >= self.budget:
                if self.verbose:
                    print(f"\n\033[93mBudget reached ({self.budget}), stopping early\033[0m")
                # Add remaining unprocessed docs to backfill
                remaining_batches = batches[batch_idx:]
                for remaining_batch in remaining_batches:
                    backfill.extend(remaining_batch)
                break
        
        return new_candidates, backfill
    
    async def _apartition_iteration(
        self,
        query: str,
        candidates: List[ListwiseRankedDocument],
        remaining: List[ListwiseRankedDocument],
        iteration: int
    ) -> Tuple[List[ListwiseRankedDocument], List[ListwiseRankedDocument]]:
        """Async version of partition_iteration with true parallelization."""
        if self.verbose:
            print(f"\n\033[94m{'='*60}\033[0m")
            print(f"\033[94mIteration {iteration}: {len(candidates)} candidates, {len(remaining)} remaining\033[0m")
            print(f"\033[94m{'='*60}\033[0m")
        
        # Step 1: Rank the top window to find pivot
        top_window = candidates[:self.window_size]
        
        if self.verbose:
            print(f"\n\033[95m=== Ranking top {len(top_window)} candidates to find pivot ===\033[0m")
        
        ranked_top = await self._arank_window(query, top_window, window_label="Initial ")
        
        # Step 2: Select pivot
        pivot = ranked_top[self.pivot_position] if len(ranked_top) > self.pivot_position else ranked_top[-1]
        
        new_candidates = ranked_top[:self.pivot_position]
        backfill = ranked_top[self.pivot_position + 1:]
        
        if self.verbose:
            print(f"\n\033[93mPivot selected: document at original position {pivot.original_position}\033[0m")
            print(f"\033[93m{len(new_candidates)} docs above pivot, {len(backfill)} docs below pivot\033[0m")
        
        # Step 3: Process remaining documents in batches (TRUE PARALLELIZATION)
        remaining_candidates = candidates[self.window_size:]
        all_remaining = remaining_candidates + remaining
        
        if len(all_remaining) == 0:
            return new_candidates, backfill
        
        batch_size = self.window_size - 1
        batches = [all_remaining[i:i + batch_size] for i in range(0, len(all_remaining), batch_size)]
        
        if self.verbose:
            print(f"\n\033[94mProcessing {len(batches)} batches in PARALLEL (window_size - 1 = {batch_size} docs per batch)\033[0m")
        
        self.parallel_inference_count += len(batches)
        
        # Process all batches concurrently
        batch_tasks = [
            self._acompare_with_pivot(query, pivot, batch, batch_idx)
            for batch_idx, batch in enumerate(batches, 1)
        ]
        
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Collect results and check budget
        for batch_idx, (above, below) in enumerate(batch_results, 1):
            new_candidates.extend(above)
            backfill.extend(below)
            
            if len(new_candidates) >= self.budget:
                if self.verbose:
                    print(f"\n\033[93mBudget reached ({self.budget}), discarding remaining batches\033[0m")
                # Add remaining unprocessed docs to backfill
                for remaining_idx in range(batch_idx, len(batches)):
                    backfill.extend(batches[remaining_idx])
                break
        
        return new_candidates, backfill
    
    def _top_down_partition(
        self,
        query: str,
        documents: List[Any]
    ) -> List[Any]:
        """
        Perform top-down partitioning reranking.
        
        Algorithm (from paper):
        1. Process top-w documents and select pivot at position k
        2. Compare pivot against remaining documents in parallel batches
        3. Collect documents ranked above pivot as candidates
        4. Recursively refine if candidates < k-1 and under budget
        5. Final ranking of candidate pool
        """
        if len(documents) == 0:
            return documents
        
        # Reset statistics
        self.inference_count = 0
        self.parallel_inference_count = 0
        
        # Wrap documents
        ranked_docs = [
            ListwiseRankedDocument(
                content=doc,
                original_position=i,
                current_position=i
            )
            for i, doc in enumerate(documents)
        ]
        
        # Limit to ranking depth
        docs_to_rank = ranked_docs[:self.ranking_depth]
        backfill = ranked_docs[self.ranking_depth:]
        
        if self.verbose:
            print(f"\n\033[92m{'='*60}\033[0m")
            print(f"\033[92mStarting Top-Down Partitioning Reranking\033[0m")
            print(f"\033[92mTotal documents: {len(documents)}, Ranking depth: {min(self.ranking_depth, len(documents))}\033[0m")
            print(f"\033[92mWindow size: {self.window_size}, Budget: {self.budget}, Pivot position: {self.pivot_position}\033[0m")
            print(f"\033[92m{'='*60}\033[0m")
        
        candidates = docs_to_rank
        iteration = 1
        
        # Main partitioning loop
        while len(candidates) > self.window_size and len(candidates) < self.budget:
            remaining_to_process = []
            candidates, new_backfill = self._partition_iteration(
                query, candidates, remaining_to_process, iteration
            )
            backfill.extend(new_backfill)
            iteration += 1
            
            # Check termination condition
            if len(candidates) <= self.window_size:
                if self.verbose:
                    print(f"\n\033[93mTerminating: candidates ({len(candidates)}) <= window_size ({self.window_size})\033[0m")
                break
        
        # Final ranking of candidates
        if len(candidates) > 1:
            if self.verbose:
                print(f"\n\033[95m=== Final ranking of {len(candidates)} candidates ===\033[0m")
            candidates = self._rank_window(query, candidates, window_label="Final ")
        
        # Combine final ranking with backfill
        final_ranking = candidates + backfill
        
        if self.verbose:
            print(f"\n\033[92m{'='*60}\033[0m")
            print(f"\033[92mReranking complete!\033[0m")
            print(f"\033[92mTotal inferences: {self.inference_count}\033[0m")
            print(f"\033[92mParallelizable inferences: {self.parallel_inference_count}\033[0m")
            print(f"\033[92m{'='*60}\033[0m\n")
        
        return [doc.content for doc in final_ranking]
    
    async def _atop_down_partition(
        self,
        query: str,
        documents: List[Any]
    ) -> List[Any]:
        """Async version of top-down partitioning with true parallelization."""
        if len(documents) == 0:
            return documents
        
        self.inference_count = 0
        self.parallel_inference_count = 0
        
        ranked_docs = [
            ListwiseRankedDocument(
                content=doc,
                original_position=i,
                current_position=i
            )
            for i, doc in enumerate(documents)
        ]
        
        docs_to_rank = ranked_docs[:self.ranking_depth]
        backfill = ranked_docs[self.ranking_depth:]
        
        if self.verbose:
            print(f"\n\033[92m{'='*60}\033[0m")
            print(f"\033[92mStarting Top-Down Partitioning Reranking (ASYNC)\033[0m")
            print(f"\033[92mTotal documents: {len(documents)}, Ranking depth: {min(self.ranking_depth, len(documents))}\033[0m")
            print(f"\033[92mWindow size: {self.window_size}, Budget: {self.budget}, Pivot position: {self.pivot_position}\033[0m")
            print(f"\033[92m{'='*60}\033[0m")
        
        candidates = docs_to_rank
        iteration = 1
        
        while len(candidates) > self.window_size and len(candidates) < self.budget:
            remaining_to_process = []
            candidates, new_backfill = await self._apartition_iteration(
                query, candidates, remaining_to_process, iteration
            )
            backfill.extend(new_backfill)
            iteration += 1
            
            if len(candidates) <= self.window_size:
                if self.verbose:
                    print(f"\n\033[93mTerminating: candidates ({len(candidates)}) <= window_size ({self.window_size})\033[0m")
                break
        
        if len(candidates) > 1:
            if self.verbose:
                print(f"\n\033[95m=== Final ranking of {len(candidates)} candidates ===\033[0m")
            candidates = await self._arank_window(query, candidates, window_label="Final ")
        
        final_ranking = candidates + backfill
        
        if self.verbose:
            print(f"\n\033[92m{'='*60}\033[0m")
            print(f"\033[92mReranking complete!\033[0m")
            print(f"\033[92mTotal inferences: {self.inference_count}\033[0m")
            print(f"\033[92mParallelizable inferences: {self.parallel_inference_count}\033[0m")
            print(f"\033[92m{'='*60}\033[0m\n")
        
        return [doc.content for doc in final_ranking]
    
    def forward(
        self,
        question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None
    ) -> DSPyAgentRAGResponse:
        if weaviate_client is None:
            if isinstance(self.weaviate_client, weaviate.WeaviateClient):
                weaviate_client = self.weaviate_client

        # Initial retrieval
        initial_results = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_client=weaviate_client,
        )
        
        if self.verbose:
            print(f"\n\033[92mInitial retrieval: {len(initial_results)} documents\033[0m")
        
        # Rerank using top-down partitioning
        reranked_results = self._top_down_partition(question, initial_results)
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_results,
            searches=[question],
            aggregations=None,
            usage={},
        )
    
    async def aforward(
        self,
        question: str,
        weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None
    ) -> DSPyAgentRAGResponse:
        if weaviate_async_client is None:
            if isinstance(self.weaviate_async_client, weaviate.WeaviateAsyncClient):
                weaviate_async_client = self.weaviate_async_client

        initial_results = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_async_client=weaviate_async_client,
        )
        
        if self.verbose:
            print(f"\n\033[92mInitial retrieval: {len(initial_results)} documents\033[0m")
        
        reranked_results = await self._atop_down_partition(question, initial_results)
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_results,
            searches=[question],
            aggregations=None,
            usage={},
        )


async def main():
    # Example demonstrating the efficiency gains
    test_pipeline = TopDownPartitioningReranker(
        collection_name="BrightBiology",
        target_property_name="content",
        verbose=True,
        retrieved_k=50,      # Retrieve 50 documents
        window_size=10,      # Process 10 docs at a time
        budget=20,           # Allow up to 20 candidates (can increase for weaker retrievers)
        ranking_depth=100,   # Rank to depth 100
        use_thinking=True,
    )
    
    test_q = "How many cells are in the human body?"
    
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    
    weaviate_async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    
    await weaviate_async_client.connect()
    
    print("=== Testing Sync Top-Down Partitioning ===")
    test_sync_response = test_pipeline.forward(test_q, weaviate_client=weaviate_client)
    print(f"\nTop 5 reranked results:")
    for i, doc in enumerate(test_sync_response.sources[:5]):
        print(f"{i+1}. {str(doc)[:100]}...")
    
    print("\n\n=== Testing Async Top-Down Partitioning (with parallel batches) ===")
    test_async_response = await test_pipeline.aforward(test_q, weaviate_async_client=weaviate_async_client)
    print(f"\nTop 5 reranked results:")
    for i, doc in enumerate(test_async_response.sources[:5]):
        print(f"{i+1}. {str(doc)[:100]}...")
    
    weaviate_client.close()
    await weaviate_async_client.close()


if __name__ == "__main__":
    asyncio.run(main())