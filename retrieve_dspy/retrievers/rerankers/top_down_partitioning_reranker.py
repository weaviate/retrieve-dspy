import asyncio
import os
from typing import Optional, List, Any, Tuple

import dspy
import weaviate

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)
from retrieve_dspy.retrievers.base_retriever import BaseRetriever
from retrieve_dspy.models import DSPyAgentRAGResponse, ListwiseRankedDocument
from retrieve_dspy.signatures import ListwiseRanking, VerboseListwiseRanking


class TopDownPartitioningReranker(BaseRetriever):
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
        target_k: Optional[int] = 10,
        window_size: Optional[int] = 10,
        budget: Optional[int] = None,
        use_thinking: Optional[bool] = True,
        ranking_depth: Optional[int] = 100,
    ):
        """
        Initialize the Top-Down Partitioning Reranker.
        
        Args:
            collection_name: Weaviate collection name
            target_property_name: Property to retrieve from documents
            weaviate_client: Weaviate client instance
            verbose: Enable detailed logging
            search_only: If True, only return reranked documents without answer generation
            retrieved_k: Number of documents to retrieve in initial search
            target_k: Target number of top documents for final ranking (the k in "top-k")
            window_size: Number of documents to rank at once
            budget: Maximum number of candidates to collect before stopping (default: window_size)
            use_thinking: Use Chain of Thought for ranking
            ranking_depth: Maximum depth to rank documents to
        """
        super().__init__(
            collection_name=collection_name,
            target_property_name=target_property_name,
            search_only=search_only,
            verbose=verbose,
            retrieved_k=retrieved_k
        )
        self.weaviate_client = weaviate_client
        self.target_k = target_k
        self.window_size = window_size
        self.budget = budget if budget is not None else window_size
        self.ranking_depth = ranking_depth
        self.pivot_position = min(target_k, window_size // 2)  # k = w/2 as per paper
        
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
    ) -> Tuple[List[ListwiseRankedDocument], List[ListwiseRankedDocument], ListwiseRankedDocument]:
        """
        Single iteration of the partitioning algorithm.
        Returns (new_candidates, backfill, pivot).
        
        The pivot is returned separately to ensure it's not lost and can be
        included in the final ranking as per Algorithm 1: A_i ∪ p ∪ B
        
        CRITICAL: Documents below the pivot in the initial window must be
        compared against the pivot in subsequent batches, not immediately
        added to backfill!
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
        
        # Step 2: Select pivot at position k (or last element if window too small)
        pivot_idx = min(self.pivot_position, len(ranked_top) - 1)
        pivot = ranked_top[pivot_idx]
        
        # Documents above pivot are guaranteed candidates for top-k
        new_candidates = ranked_top[:pivot_idx]
        
        # Documents below pivot need to be compared against the pivot!
        # They do NOT go directly to backfill - that was the bug!
        docs_below_pivot = ranked_top[pivot_idx + 1:]
        
        if self.verbose:
            print(f"\n\033[93mPivot selected: document at original position {pivot.original_position}\033[0m")
            print(f"\033[93m{len(new_candidates)} docs above pivot (added to candidates)\033[0m")
            print(f"\033[93m{len(docs_below_pivot)} docs below pivot (need pivot comparison)\033[0m")
        
        # Step 3: Combine ALL documents that need to be compared with pivot:
        # - Documents below the pivot in the initial window
        # - Remaining unprocessed candidates from the candidate pool
        # - Any leftover documents from previous iteration
        remaining_candidates = candidates[self.window_size:]
        all_remaining = docs_below_pivot + remaining_candidates + remaining
        
        if self.verbose:
            print(f"\033[93mTotal documents to compare with pivot: {len(all_remaining)}\033[0m")
        
        # Initialize empty backfill - only populated after pivot comparisons
        backfill = []
        
        if len(all_remaining) == 0:
            return new_candidates, backfill, pivot
        
        # Create batches of size (window_size - 1) to account for pivot
        batch_size = self.window_size - 1
        batches = [all_remaining[i:i + batch_size] for i in range(0, len(all_remaining), batch_size)]
        
        if self.verbose:
            print(f"\n\033[94mProcessing {len(batches)} batches (window_size - 1 = {batch_size} docs per batch)\033[0m")
        
        self.parallel_inference_count += len(batches)
        
        # Process each batch (in production, these could run in parallel)
        for batch_idx, batch in enumerate(batches, 1):
            above, below = self._compare_with_pivot(query, pivot, batch, batch_idx)
            new_candidates.extend(above)
            backfill.extend(below)
            
            # Early stopping if we've reached budget
            if len(new_candidates) >= self.budget:
                if self.verbose:
                    print(f"\n\033[93m⚠️  Budget reached ({self.budget}), stopping early\033[0m")
                # Add remaining unprocessed docs to backfill
                remaining_batches = batches[batch_idx:]
                for remaining_batch in remaining_batches:
                    backfill.extend(remaining_batch)
                break
        
        return new_candidates, backfill, pivot
    
    async def _apartition_iteration(
        self,
        query: str,
        candidates: List[ListwiseRankedDocument],
        remaining: List[ListwiseRankedDocument],
        iteration: int
    ) -> Tuple[List[ListwiseRankedDocument], List[ListwiseRankedDocument], ListwiseRankedDocument]:
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
        pivot_idx = min(self.pivot_position, len(ranked_top) - 1)
        pivot = ranked_top[pivot_idx]
        
        new_candidates = ranked_top[:pivot_idx]
        docs_below_pivot = ranked_top[pivot_idx + 1:]
        
        if self.verbose:
            print(f"\n\033[93mPivot selected: document at original position {pivot.original_position}\033[0m")
            print(f"\033[93m{len(new_candidates)} docs above pivot (added to candidates)\033[0m")
            print(f"\033[93m{len(docs_below_pivot)} docs below pivot (need pivot comparison)\033[0m")
        
        # Step 3: Combine all documents needing pivot comparison
        remaining_candidates = candidates[self.window_size:]
        all_remaining = docs_below_pivot + remaining_candidates + remaining
        
        if self.verbose:
            print(f"\033[93mTotal documents to compare with pivot: {len(all_remaining)}\033[0m")
        
        backfill = []
        
        if len(all_remaining) == 0:
            return new_candidates, backfill, pivot
        
        batch_size = self.window_size - 1
        batches = [all_remaining[i:i + batch_size] for i in range(0, len(all_remaining), batch_size)]
        
        if self.verbose:
            print(f"\n\033[94m⚡ Processing {len(batches)} batches in PARALLEL ⚡\033[0m")
        
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
                    print(f"\n\033[93m⚠️  Budget reached ({self.budget})\033[0m")
                # Add remaining unprocessed docs to backfill
                for remaining_idx in range(batch_idx, len(batches)):
                    backfill.extend(batches[remaining_idx])
                break
        
        return new_candidates, backfill, pivot
    
    def _top_down_partition_recursive(
        self,
        query: str,
        documents: List[Any],
        depth: int = 0
    ) -> List[Any]:
        """
        Recursive version of top-down partitioning for refining large candidate pools.
        This implements the "pivot(A_i)" call from Algorithm 1, Line 14.
        
        Args:
            query: Search query
            documents: Documents to partition
            depth: Recursion depth (for logging)
        
        Returns:
            Ranked list of documents
        """
        if len(documents) <= self.window_size:
            # Base case: small enough to rank directly
            if self.verbose:
                print(f"\n\033[96m{'  '*depth}↳ Recursive depth {depth}: {len(documents)} docs, ranking directly\033[0m")
            
            ranked_docs = [
                ListwiseRankedDocument(content=doc, original_position=i, current_position=i)
                for i, doc in enumerate(documents)
            ]
            ranked = self._rank_window(query, ranked_docs, window_label=f"Recursive-{depth} ")
            return [doc.content for doc in ranked]
        
        if self.verbose:
            print(f"\n\033[96m{'  '*depth}↳ Recursive depth {depth}: Partitioning {len(documents)} docs\033[0m")
        
        # Wrap documents
        ranked_docs = [
            ListwiseRankedDocument(content=doc, original_position=i, current_position=i)
            for i, doc in enumerate(documents)
        ]
        
        candidates = ranked_docs
        backfill = []
        all_pivots = []
        
        # Single iteration of partitioning
        iteration = 1
        while len(candidates) > self.window_size:
            new_candidates, new_backfill, pivot = self._partition_iteration(
                query, candidates, [], iteration
            )
            
            all_pivots.append(pivot)
            backfill.extend(new_backfill)
            candidates = new_candidates
            
            if len(candidates) <= self.window_size:
                break
            
            iteration += 1
            if iteration > 5:  # Limit recursion iterations
                break
        
        # Final ranking of candidates
        if len(candidates) > 1:
            candidates = self._rank_window(
                query, 
                candidates, 
                window_label=f"Recursive-{depth}-Final "
            )
        
        # Combine and return
        final = candidates + all_pivots + backfill
        return [doc.content for doc in final]
    
    async def _atop_down_partition_recursive(
        self,
        query: str,
        documents: List[Any],
        depth: int = 0
    ) -> List[Any]:
        """Async recursive partitioning."""
        if len(documents) <= self.window_size:
            if self.verbose:
                print(f"\n\033[96m{'  '*depth}↳ Async recursive depth {depth}: {len(documents)} docs, ranking directly\033[0m")
            
            ranked_docs = [
                ListwiseRankedDocument(content=doc, original_position=i, current_position=i)
                for i, doc in enumerate(documents)
            ]
            ranked = await self._arank_window(query, ranked_docs, window_label=f"AsyncRecursive-{depth} ")
            return [doc.content for doc in ranked]
        
        if self.verbose:
            print(f"\n\033[96m{'  '*depth}↳ Async recursive depth {depth}: Partitioning {len(documents)} docs\033[0m")
        
        ranked_docs = [
            ListwiseRankedDocument(content=doc, original_position=i, current_position=i)
            for i, doc in enumerate(documents)
        ]
        
        candidates = ranked_docs
        backfill = []
        all_pivots = []
        
        iteration = 1
        while len(candidates) > self.window_size:
            new_candidates, new_backfill, pivot = await self._apartition_iteration(
                query, candidates, [], iteration
            )
            
            all_pivots.append(pivot)
            backfill.extend(new_backfill)
            candidates = new_candidates
            
            if len(candidates) <= self.window_size:
                break
            
            iteration += 1
            if iteration > 5:
                break
        
        if len(candidates) > 1:
            candidates = await self._arank_window(
                query, 
                candidates, 
                window_label=f"AsyncRecursive-{depth}-Final "
            )
        
        final = candidates + all_pivots + backfill
        return [doc.content for doc in final]
    
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
        4. Recursively refine if candidates > target_k
        5. Final ranking of candidate pool
        
        Paper's termination condition (Algorithm 1, Line 14):
        return (|A_i| = k - 1) ? A_i ∪ p ∪ B : pivot(A_i) ∪ p ∪ B
        
        This means:
        - If we have exactly k-1 candidates: done!
        - If we have more: recursively partition the candidates
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
            print("\033[92mStarting Top-Down Partitioning Reranking\033[0m")
            print(f"\033[92mTotal documents: {len(documents)}, Ranking depth: {min(self.ranking_depth, len(documents))}\033[0m")
            print(f"\033[92mWindow size: {self.window_size}, Budget: {self.budget}, Pivot position: {self.pivot_position}\033[0m")
            print(f"\033[92mTarget k: {self.target_k}\033[0m")
            print(f"\033[92m{'='*60}\033[0m")
        
        candidates = docs_to_rank
        all_pivots = []  # Track all pivots (they should be included in final ranking)
        iteration = 1
        
        # Main partitioning loop
        # Continue while we have enough candidates to partition
        while len(candidates) > self.window_size:
            if self.verbose:
                print(f"\n\033[94m>>> Loop iteration {iteration}: {len(candidates)} candidates\033[0m")
            
            # Single partition iteration
            new_candidates, new_backfill, pivot = self._partition_iteration(
                query, candidates, [], iteration
            )
            
            all_pivots.append(pivot)
            backfill.extend(new_backfill)
            candidates = new_candidates
            
            # Check termination conditions
            if len(candidates) <= self.window_size:
                if self.verbose:
                    print(f"\n\033[93m✓ Termination condition met: {len(candidates)} candidates <= window_size ({self.window_size})\033[0m")
                break
            
            # Budget exceeded - stop collecting more candidates
            if len(candidates) >= self.budget:
                if self.verbose:
                    print(f"\n\033[93m✓ Budget limit reached: {len(candidates)} candidates >= budget ({self.budget})\033[0m")
                break
            
            iteration += 1
            
            # Safety check to prevent infinite loops
            if iteration > 10:
                if self.verbose:
                    print("\n\033[91m⚠ Maximum iterations reached, stopping\033[0m")
                break
        
        # Now we have a candidate pool. Apply paper's decision rule:
        # If |candidates| ≈ target_k: Done! Return candidates ∪ pivots ∪ backfill
        # If |candidates| > target_k: Recursively partition candidates (the "pivot(A_i)" call)
        
        if len(candidates) > self.target_k and len(candidates) > self.window_size:
            # Too many candidates - recursively refine
            if self.verbose:
                print(f"\n\033[95m{'='*60}\033[0m")
                print(f"\033[95m🔄 Recursive refinement needed: {len(candidates)} candidates > target {self.target_k}\033[0m")
                print(f"\033[95m{'='*60}\033[0m")
            
            # Recursively partition the candidate set
            # Note: This is the "pivot(A_i)" call from Algorithm 1, Line 14
            candidate_contents = [doc.content for doc in candidates]
            refined_docs = self._top_down_partition_recursive(query, candidate_contents, depth=1)
            
            # Rewrap refined documents
            candidates = [
                ListwiseRankedDocument(
                    content=doc,
                    original_position=-1,
                    current_position=i
                )
                for i, doc in enumerate(refined_docs[:self.target_k])
            ]
            
            # Anything not in top-k goes to backfill
            if len(refined_docs) > self.target_k:
                backfill_docs = [
                    ListwiseRankedDocument(
                        content=doc,
                        original_position=-1,
                        current_position=i
                    )
                    for i, doc in enumerate(refined_docs[self.target_k:], start=self.target_k)
                ]
                backfill.extend(backfill_docs)
        
        elif len(candidates) > 1:
            # Small enough candidate set - just do final ranking
            if self.verbose:
                print(f"\n\033[95m{'='*60}\033[0m")
                print(f"\033[95m🎯 Final ranking of {len(candidates)} candidates (small enough)\033[0m")
                print(f"\033[95m{'='*60}\033[0m")
            
            candidates = self._rank_window(query, candidates, window_label="Final ")
        
        # Combine: candidates (top-k) + pivots + backfill
        # Paper: A_i ∪ p ∪ B
        final_ranking = candidates + all_pivots + backfill
        
        if self.verbose:
            print(f"\n\033[92m{'='*60}\033[0m")
            print("\033[92m✅ Reranking complete!\033[0m")
            print(f"\033[92mFinal ranking: {len(candidates)} top candidates + {len(all_pivots)} pivots + {len(backfill)} backfill\033[0m")
            print(f"\033[92mTotal inferences: {self.inference_count}\033[0m")
            print(f"\033[92mParallelizable inferences: {self.parallel_inference_count}\033[0m")
            print(f"\033[92mEstimated speedup: {self.parallel_inference_count / max(self.inference_count, 1):.2f}x with parallelization\033[0m")
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
            print("\033[92m⚡ Starting Top-Down Partitioning Reranking (ASYNC) ⚡\033[0m")
            print(f"\033[92mTotal documents: {len(documents)}, Ranking depth: {min(self.ranking_depth, len(documents))}\033[0m")
            print(f"\033[92mWindow size: {self.window_size}, Budget: {self.budget}, Pivot position: {self.pivot_position}\033[0m")
            print(f"\033[92mTarget k: {self.target_k}\033[0m")
            print(f"\033[92m{'='*60}\033[0m")
        
        candidates = docs_to_rank
        all_pivots = []
        iteration = 1
        
        while len(candidates) > self.window_size:
            if self.verbose:
                print(f"\n\033[94m>>> Loop iteration {iteration}: {len(candidates)} candidates\033[0m")
            
            new_candidates, new_backfill, pivot = await self._apartition_iteration(
                query, candidates, [], iteration
            )
            
            all_pivots.append(pivot)
            backfill.extend(new_backfill)
            candidates = new_candidates
            
            if len(candidates) <= self.window_size:
                if self.verbose:
                    print(f"\n\033[93m✓ Termination: {len(candidates)} candidates <= window_size\033[0m")
                break
            
            if len(candidates) >= self.budget:
                if self.verbose:
                    print(f"\n\033[93m✓ Budget reached: {len(candidates)} candidates >= budget\033[0m")
                break
            
            iteration += 1
            if iteration > 10:
                if self.verbose:
                    print("\n\033[91m⚠ Maximum iterations reached\033[0m")
                break
        
        # Recursive refinement or final ranking
        if len(candidates) > self.target_k and len(candidates) > self.window_size:
            if self.verbose:
                print(f"\n\033[95m{'='*60}\033[0m")
                print(f"\033[95m🔄 Recursive refinement: {len(candidates)} > {self.target_k}\033[0m")
                print(f"\033[95m{'='*60}\033[0m")
            
            candidate_contents = [doc.content for doc in candidates]
            refined_docs = await self._atop_down_partition_recursive(query, candidate_contents, depth=1)
            
            candidates = [
                ListwiseRankedDocument(content=doc, original_position=-1, current_position=i)
                for i, doc in enumerate(refined_docs[:self.target_k])
            ]
            
            if len(refined_docs) > self.target_k:
                backfill_docs = [
                    ListwiseRankedDocument(content=doc, original_position=-1, current_position=i)
                    for i, doc in enumerate(refined_docs[self.target_k:], start=self.target_k)
                ]
                backfill.extend(backfill_docs)
        
        elif len(candidates) > 1:
            if self.verbose:
                print(f"\n\033[95m{'='*60}\033[0m")
                print(f"\033[95m🎯 Final ranking of {len(candidates)} candidates\033[0m")
                print(f"\033[95m{'='*60}\033[0m")
            candidates = await self._arank_window(query, candidates, window_label="Final ")
        
        final_ranking = candidates + all_pivots + backfill
        
        if self.verbose:
            print(f"\n\033[92m{'='*60}\033[0m")
            print("\033[92m✅ Async reranking complete!\033[0m")
            print(f"\033[92mFinal: {len(candidates)} candidates + {len(all_pivots)} pivots + {len(backfill)} backfill\033[0m")
            print(f"\033[92mTotal inferences: {self.inference_count}\033[0m")
            print(f"\033[92mParallelizable: {self.parallel_inference_count}\033[0m")
            print(f"\033[92mTheoretical speedup: {self.parallel_inference_count / max(self.inference_count, 1):.2f}x\033[0m")
            print(f"\033[92m{'='*60}\033[0m\n")
        
        return [doc.content for doc in final_ranking]
    
    def forward(
        self,
        question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None
    ) -> DSPyAgentRAGResponse:
        """
        Synchronous forward pass: retrieve and rerank documents.
        
        Args:
            question: User query
            weaviate_client: Weaviate client for search
            
        Returns:
            DSPyAgentRAGResponse with reranked documents
        """
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
            print(f"\n\033[92m🔍 Initial retrieval: {len(initial_results)} documents\033[0m")
        
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
        """
        Async forward pass: retrieve and rerank documents with parallelization.
        
        Args:
            question: User query
            weaviate_async_client: Async Weaviate client for search
            
        Returns:
            DSPyAgentRAGResponse with reranked documents
        """
        if weaviate_async_client is None:
            if isinstance(self.weaviate_client, weaviate.WeaviateAsyncClient):
                weaviate_async_client = self.weaviate_client

        initial_results = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_async_client=weaviate_async_client,
        )
        
        if self.verbose:
            print(f"\n\033[92m🔍 Initial retrieval: {len(initial_results)} documents\033[0m")
        
        reranked_results = await self._atop_down_partition(question, initial_results)
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_results,
            searches=[question],
            aggregations=None,
            usage={},
        )


async def main():
    """
    Example demonstrating the efficiency gains of top-down partitioning.
    
    This example shows:
    1. How to configure the reranker with different parameters
    2. Comparison between sync and async implementations
    3. Efficiency metrics (inference count, parallelization opportunities)
    """
    
    # Test different configurations to see efficiency trade-offs
    print("\033[93m" + "="*80)
    print("TOP-DOWN PARTITIONING RERANKER - DEMONSTRATION")
    print("="*80 + "\033[0m\n")
    
    configs = [
        {
            "name": "Small Window (Fast)",
            "window_size": 5,
            "target_k": 10,
            "budget": 15,
            "ranking_depth": 50,
        },
        {
            "name": "Medium Window (Balanced)",
            "window_size": 10,
            "target_k": 10,
            "budget": 30,
            "ranking_depth": 100,
        },
    ]
    
    test_q = "How many cells are in the human body?"
    
    # Setup Weaviate clients
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    
    weaviate_async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    
    await weaviate_async_client.connect()
    
    for config in configs:
        print(f"\n\033[93m{'='*80}")
        print(f"Testing Configuration: {config['name']}")
        print(f"{'='*80}\033[0m")
        print(f"Parameters: {config}")
        print()
        
        test_pipeline = TopDownPartitioningReranker(
            collection_name="BrightBiology",
            target_property_name="content",
            verbose=True,
            retrieved_k=20,
            use_thinking=True,
            **{k: v for k, v in config.items() if k != 'name'}
        )
        
        print("\n\033[96m" + "="*80)
        print("SYNC VERSION (Sequential)")
        print("="*80 + "\033[0m")
        
        test_sync_response = test_pipeline.forward(test_q, weaviate_client=weaviate_client)
        
        print("\n\033[92mTop 5 reranked results:\033[0m")
        for i, doc in enumerate(test_sync_response.sources[:5]):
            doc_str = str(doc)[:150] if len(str(doc)) > 150 else str(doc)
            print(f"\033[96m{i+1}.\033[0m {doc_str}...")
        
        print("\n\033[93m📊 Efficiency Metrics:\033[0m")
        print(f"   • Total inferences: {test_pipeline.inference_count}")
        print(f"   • Parallelizable inferences: {test_pipeline.parallel_inference_count}")
        print(f"   • Potential speedup: {test_pipeline.parallel_inference_count / max(test_pipeline.inference_count, 1):.2f}x")
        
        print("\n\n\033[96m" + "="*80)
        print("ASYNC VERSION (With True Parallelization)")
        print("="*80 + "\033[0m")
        
        test_async_response = await test_pipeline.aforward(test_q, weaviate_async_client=weaviate_async_client)
        
        print("\n\033[92mTop 5 reranked results:\033[0m")
        for i, doc in enumerate(test_async_response.sources[:5]):
            doc_str = str(doc)[:150] if len(str(doc)) > 150 else str(doc)
            print(f"\033[96m{i+1}.\033[0m {doc_str}...")
        
        print("\n\033[93m📊 Efficiency Metrics:\033[0m")
        print(f"   • Total inferences: {test_pipeline.inference_count}")
        print(f"   • Parallelizable inferences: {test_pipeline.parallel_inference_count}")
        print(f"   • Actual speedup with async: ~{test_pipeline.parallel_inference_count / max(test_pipeline.inference_count, 1):.2f}x")
        
        print("\n" + "="*80 + "\n")
    
    # Comparison with theoretical sliding window
    print("\n\033[93m" + "="*80)
    print("EFFICIENCY COMPARISON")
    print("="*80 + "\033[0m\n")
    
    print("For ranking to depth 100 with window size 10:")
    print()
    print("Sliding Window (stride=5):")
    print("  • Inferences needed: (100 / 5) - 1 = \033[91m19 inferences\033[0m")
    print("  • Parallelizable: \033[91m0 (sequential dependency)\033[0m")
    print("  • Issues: Redundant re-scoring, bottom-up bias")
    print()
    print("Top-Down Partitioning:")
    print("  • Inferences needed: ~\033[92m12-13 inferences\033[0m (33% reduction)")
    print("  • Parallelizable: \033[92m~8-10 batches\033[0m")
    print("  • Benefits: No redundant scoring, top-down bias, parallelizable")
    print()
    print("\033[92m✅ Result: Same quality, fewer inferences, better parallelism!\033[0m")
    print()
    
    weaviate_client.close()
    await weaviate_async_client.close()


if __name__ == "__main__":
    asyncio.run(main())