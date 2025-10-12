import asyncio
import os
from typing import Optional, List, Any

import dspy
import weaviate

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse, ListwiseRankedDocument
from retrieve_dspy.signatures import ListwiseRanking, VerboseListwiseRanking


class SlidingWindowReranker(BaseRAG):
    """
    Listwise reranker using a sliding window approach.
    
    Processes documents in overlapping windows from bottom to top,
    progressively sorting and bubbling the most relevant documents to the top.
    
    Example with window_size=5, stride=3, total_docs=15:
    Window 1: docs [10-14] → sort within window
    Window 2: docs [7-11] → sort, best bubble up
    Window 3: docs [4-8] → sort, best bubble up
    Window 4: docs [1-5] → sort, best bubble up
    Window 5: docs [0-4] → sort, best bubble up
    Final result: fully sorted list
    """
    
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 50,
        window_size: Optional[int] = 10,
        stride: Optional[int] = 5,
        use_thinking: Optional[bool] = True,
    ):
        super().__init__(
            collection_name=collection_name,
            target_property_name=target_property_name,
            search_only=search_only,
            verbose=verbose,
            retrieved_k=retrieved_k
        )
        self.window_size = window_size
        self.stride = stride
        
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
    
    def _rerank_window(
        self, 
        query: str, 
        window_docs: List[ListwiseRankedDocument]
    ) -> List[ListwiseRankedDocument]:
        """Rerank a single window of documents."""
        if len(window_docs) <= 1:
            return window_docs
        
        # Extract text content for ranking
        doc_texts = [self._extract_document_text(doc.content) for doc in window_docs]
        
        # Get ranking from LLM
        ranking_response = self.ranker(
            query=query,
            documents=doc_texts,
        )
        
        # Parse ranked indices
        ranked_indices = self._parse_ranked_indices(ranking_response.ranked_indices)
        
        # Handle parsing failure - return original order
        if ranked_indices is None:
            if self.verbose:
                print(f"\033[91mFailed to parse ranking output. Using original order.\033[0m")
            return window_docs
        
        # Validate indices
        valid_indices = [idx for idx in ranked_indices 
                        if 0 <= idx < len(window_docs)]
        
        # Handle missing indices
        missing = set(range(len(window_docs))) - set(valid_indices)
        valid_indices.extend(sorted(missing))
        
        # Reorder documents
        reranked = [window_docs[idx] for idx in valid_indices[:len(window_docs)]]
        
        if self.verbose:
            # Show original indices instead of window-relative indices
            original_indices = [window_docs[idx].original_position for idx in valid_indices[:len(window_docs)]]
            print(f"\033[96mWindow ranking: {original_indices}\033[0m")
        
        return reranked
    
    async def _arerank_window(
        self, 
        query: str, 
        window_docs: List[ListwiseRankedDocument]
    ) -> List[ListwiseRankedDocument]:
        """Async version of rerank_window."""
        if len(window_docs) <= 1:
            return window_docs
        
        doc_texts = [self._extract_document_text(doc.content) for doc in window_docs]
        
        ranking_response = await self.ranker.acall(
            query=query,
            documents=doc_texts,
        )
        
        ranked_indices = self._parse_ranked_indices(ranking_response.ranked_indices)
        
        # Handle parsing failure - return original order
        if ranked_indices is None:
            if self.verbose:
                print(f"\033[91mFailed to parse ranking output. Using original order.\033[0m")
            return window_docs

        valid_indices = [idx for idx in ranked_indices 
                        if 0 <= idx < len(window_docs)]
        missing = set(range(len(window_docs))) - set(valid_indices)
        valid_indices.extend(sorted(missing))
        
        reranked = [window_docs[idx] for idx in valid_indices[:len(window_docs)]]
        
        if self.verbose:
            # Show original indices instead of window-relative indices
            original_indices = [window_docs[idx].original_position for idx in valid_indices[:len(window_docs)]]
            print(f"\033[96mWindow ranking: {original_indices}\033[0m")
        
        return reranked
    
    def _sliding_window_rerank(
        self, 
        query: str, 
        documents: List[Any]
    ) -> List[Any]:
        """
        Perform sliding window reranking from bottom to top.
        
        Process windows from the end of the list backwards, allowing
        highly relevant documents to "bubble up" to the top.
        """
        if len(documents) == 0:
            return documents
        
        # Wrap documents with ListwiseRankedDocument
        ranked_docs = [
            ListwiseRankedDocument(
                content=doc, 
                original_position=i,
                current_position=i
            ) 
            for i, doc in enumerate(documents)
        ]
        
        # Calculate window positions (from bottom to top)
        num_docs = len(ranked_docs)
        window_starts = list(range(num_docs - self.window_size, -1, -self.stride))
        
        # Ensure we cover the beginning
        if window_starts[-1] != 0:
            window_starts.append(0)
        
        window_starts = sorted(set(window_starts), reverse=True)
        
        if self.verbose:
            print(f"\033[94mReranking {num_docs} documents with {len(window_starts)} windows\033[0m")
            print(f"\033[94mWindow starts: {window_starts}\033[0m")
        
        # Process each window
        for window_idx, start in enumerate(window_starts):
            end = min(start + self.window_size, num_docs)
            
            if self.verbose:
                print(f"\n\033[95m=== Window {window_idx + 1}/{len(window_starts)}: docs [{start}:{end}] ===\033[0m")
            
            # Extract window
            window = ranked_docs[start:end]
            
            # Rerank window
            reranked_window = self._rerank_window(query, window)
            
            # Update the main list with reranked window
            ranked_docs[start:end] = reranked_window
        
        # Extract final ordered documents
        return [doc.content for doc in ranked_docs]
    
    async def _asliding_window_rerank(
        self, 
        query: str, 
        documents: List[Any]
    ) -> List[Any]:
        """Async version of sliding window rerank."""
        if len(documents) == 0:
            return documents
        
        ranked_docs = [
            ListwiseRankedDocument(
                content=doc, 
                original_position=i,
                current_position=i
            ) 
            for i, doc in enumerate(documents)
        ]
        
        num_docs = len(ranked_docs)
        window_starts = list(range(num_docs - self.window_size, -1, -self.stride))
        
        if window_starts[-1] != 0:
            window_starts.append(0)
        
        window_starts = sorted(set(window_starts), reverse=True)
        
        if self.verbose:
            print(f"\033[94mReranking {num_docs} documents with {len(window_starts)} windows\033[0m")
            print(f"\033[94mWindow starts: {window_starts}\033[0m")
        
        for window_idx, start in enumerate(window_starts):
            end = min(start + self.window_size, num_docs)
            
            if self.verbose:
                print(f"\n\033[95m=== Window {window_idx + 1}/{len(window_starts)}: docs [{start}:{end}] ===\033[0m")
            
            window = ranked_docs[start:end]
            reranked_window = await self._arerank_window(query, window)
            ranked_docs[start:end] = reranked_window
        
        return [doc.content for doc in ranked_docs]
    
    def forward(
        self, 
        question: str, 
        weaviate_client: Optional[weaviate.WeaviateClient] = None
    ) -> DSPyAgentRAGResponse:
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
        
        # Rerank using sliding windows
        reranked_results = self._sliding_window_rerank(question, initial_results)
        
        if self.verbose:
            print(f"\n\033[92mReranking complete!\033[0m\n")
        
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
        initial_results = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_async_client=weaviate_async_client,
        )
        
        if self.verbose:
            print(f"\n\033[92mInitial retrieval: {len(initial_results)} documents\033[0m")
        
        reranked_results = await self._asliding_window_rerank(question, initial_results)
        
        if self.verbose:
            print(f"\n\033[92mReranking complete!\033[0m\n")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_results,
            searches=[question],
            aggregations=None,
            usage={},
        )


async def main():
    # Example with smaller numbers for demonstration
    test_pipeline = SlidingWindowReranker(
        collection_name="BrightBiology",
        target_property_name="content",
        verbose=True,
        retrieved_k=20,  # Get top 20 documents
        window_size=5,   # Process 5 docs at a time
        stride=3,        # Move 3 positions each time
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
    
    print("=== Testing Sync Reranking ===")
    test_sync_response = test_pipeline.forward(test_q, weaviate_client=weaviate_client)
    print(f"\nTop 3 reranked results:")
    for i, doc in enumerate(test_sync_response.sources[:3]):
        print(f"{i+1}. {str(doc)[:100]}...")
    
    print("\n\n=== Testing Async Reranking ===")
    test_async_response = await test_pipeline.aforward(test_q, weaviate_async_client=weaviate_async_client)
    print(f"\nTop 3 reranked results:")
    for i, doc in enumerate(test_async_response.sources[:3]):
        print(f"{i+1}. {str(doc)[:100]}...")
    
    weaviate_client.close()
    await weaviate_async_client.close()


if __name__ == "__main__":
    asyncio.run(main())