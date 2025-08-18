import asyncio
import os
from typing import Optional, List, Literal, Union, Any, Dict, Tuple
from collections import defaultdict

import cohere
import voyageai
import dspy

from retrieve_dspy.tools.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import QuerySummarizer

class CrossEncoderReranker(BaseRAG):
    def __init__(
        self, 
        collection_name: str, 
        target_property_name: str,
        return_property_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 50,
        reranked_k: Optional[int] = 20,
        reranker_provider: Literal["cohere", "voyage", "hybrid"] = "cohere",
        cohere_model: Optional[str] = "rerank-v3.5",
        voyage_model: Optional[str] = "rerank-2.5",
        cohere_api_key: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
        summarize_query: Optional[bool] = False,
        rrf_k: Optional[int] = 60,
        hybrid_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the Cross Encoder Reranker.
        
        Args:
            collection_name: Weaviate collection name
            target_property_name: Property to search in Weaviate
            verbose: Whether to print debug information
            search_only: Whether to only search without generating answers
            retrieved_k: Number of documents to retrieve initially
            reranked_k: Number of documents to keep after reranking
            reranker_provider: Which reranker to use ("cohere", "voyage", or "hybrid")
            cohere_model: Cohere reranking model to use
            voyage_model: Voyage reranking model to use
            cohere_api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            voyage_api_key: Voyage API key (defaults to VOYAGE_API_KEY env var)
            summarize_query: Whether to summarize the query before reranking
            rrf_k: K parameter for Reciprocal Rank Fusion (default 60)
            hybrid_weights: Optional weights for each reranker in hybrid mode 
                          (e.g., {"cohere": 0.6, "voyage": 0.4})
        """
        super().__init__(
            collection_name=collection_name, 
            target_property_name=target_property_name, 
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k,
        )
        self.return_property_name = return_property_name
        self.reranked_k = reranked_k
        self.reranker_provider = reranker_provider
        self.cohere_model = cohere_model
        self.voyage_model = voyage_model
        self.summarize_query = summarize_query
        self.query_summarizer = dspy.Predict(QuerySummarizer)
        self.rrf_k = rrf_k
        self.hybrid_weights = hybrid_weights or {"cohere": 0.5, "voyage": 0.5}
        
        # Initialize clients based on provider
        if reranker_provider in ["cohere", "hybrid"]:
            api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("COHERE_API_KEY must be provided or set as environment variable")
            self.co = cohere.ClientV2(api_key)
            
        if reranker_provider in ["voyage", "hybrid"]:
            api_key = voyage_api_key or os.getenv("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError("VOYAGE_API_KEY must be provided or set as environment variable")
            self.vo = voyageai.Client(api_key=api_key)
            
        if reranker_provider not in ["cohere", "voyage", "hybrid"]:
            raise ValueError(f"Unsupported reranker provider: {reranker_provider}")
    
    def _rerank_with_cohere(
        self, 
        query: str, 
        documents: List[str]
    ) -> List[Any]:
        """
        Rerank documents using Cohere's Cross Encoder.
        
        Args:
            query: User query
            documents: List of document texts to rerank
            
        Returns:
            Reranked results from Cohere
        """
        try:
            response = self.co.rerank(
                model=self.cohere_model,
                query=query,
                documents=documents,
                top_n=min(self.reranked_k, len(documents))
            )
            return response.results
        except Exception as e:
            if self.verbose:
                print(f"\033[91mError during Cohere reranking: {e}\033[0m")
            raise
    
    def _rerank_with_voyage(
        self, 
        query: str, 
        documents: List[str]
    ) -> List[Any]:
        """
        Rerank documents using Voyage's Cross Encoder.
        
        Args:
            query: User query
            documents: List of document texts to rerank
            
        Returns:
            Reranked results from Voyage
        """
        try:
            response = self.vo.rerank(
                query=query,
                documents=documents,
                model=self.voyage_model,
                top_k=min(self.reranked_k, len(documents))
            )
            return response.results
        except Exception as e:
            if self.verbose:
                print(f"\033[91mError during Voyage reranking: {e}\033[0m")
            raise
    
    def _reciprocal_rank_fusion(
        self,
        rankings: Dict[str, List[Tuple[int, float]]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Combine multiple rankings using Reciprocal Rank Fusion.
        
        Args:
            rankings: Dictionary mapping ranker name to list of (doc_index, score) tuples
            k: RRF constant (default 60)
            
        Returns:
            Combined ranking as list of (doc_index, fused_score) tuples
        """
        rrf_scores = defaultdict(float)
        
        for ranker_name, ranked_docs in rankings.items():
            weight = self.hybrid_weights.get(ranker_name, 0.5)
            
            for rank, (doc_idx, original_score) in enumerate(ranked_docs):
                # RRF formula: 1 / (k + rank)
                # rank is 0-based, so we add 1 to get the actual position
                rrf_score = weight * (1.0 / (k + rank + 1))
                rrf_scores[doc_idx] += rrf_score
                
                if self.verbose and rank < 3:
                    print(f"  {ranker_name} - Rank {rank+1}: Doc {doc_idx}, "
                          f"Original score: {original_score:.4f}, "
                          f"RRF contribution: {rrf_score:.4f}")
        
        # Sort by RRF score in descending order
        fused_ranking = sorted(
            rrf_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if self.verbose:
            print(f"\n\033[93mRRF Fusion Results (k={k}):\033[0m")
            for i, (doc_idx, score) in enumerate(fused_ranking[:5]):
                print(f"  Final Rank {i+1}: Doc {doc_idx}, RRF Score: {score:.4f}")
        
        return fused_ranking[:self.reranked_k]
    
    def _rerank_hybrid(
        self,
        query: str,
        documents: List[str]
    ) -> List[Tuple[int, float]]:
        """
        Rerank using both Cohere and Voyage, then fuse with RRF.
        
        Args:
            query: User query
            documents: List of document texts to rerank
            
        Returns:
            Fused ranking as list of (doc_index, fused_score) tuples
        """
        if self.verbose:
            print(f"\n\033[95mHybrid Reranking Mode - Using both Cohere and Voyage\033[0m")
            print(f"Weights: Cohere={self.hybrid_weights['cohere']}, "
                  f"Voyage={self.hybrid_weights['voyage']}")
        
        rankings = {}
        
        # Get Cohere rankings
        try:
            cohere_results = self._rerank_with_cohere(query, documents)
            # Store as (doc_index, relevance_score) tuples
            rankings["cohere"] = [
                (result.index, result.relevance_score) 
                for result in cohere_results
            ]
            if self.verbose:
                print(f"\n\033[96mCohere returned {len(cohere_results)} results\033[0m")
        except Exception as e:
            if self.verbose:
                print(f"\033[91mCohere reranking failed: {e}\033[0m")
            rankings["cohere"] = []
        
        # Get Voyage rankings
        try:
            voyage_results = self._rerank_with_voyage(query, documents)
            rankings["voyage"] = [
                (result.index, result.relevance_score) 
                for result in voyage_results
            ]
            if self.verbose:
                print(f"\033[96mVoyage returned {len(voyage_results)} results\033[0m")
        except Exception as e:
            if self.verbose:
                print(f"\033[91mVoyage reranking failed: {e}\033[0m")
            rankings["voyage"] = []
        
        # If one ranker fails, use the other's results
        if not rankings["cohere"] and rankings["voyage"]:
            if self.verbose:
                print("\033[93mUsing only Voyage results (Cohere failed)\033[0m")
            return rankings["voyage"]
        elif not rankings["voyage"] and rankings["cohere"]:
            if self.verbose:
                print("\033[93mUsing only Cohere results (Voyage failed)\033[0m")
            return rankings["cohere"]
        elif not rankings["cohere"] and not rankings["voyage"]:
            raise RuntimeError("Both rerankers failed")
        
        # Fuse rankings using RRF
        return self._reciprocal_rank_fusion(rankings, self.rrf_k)
    
    async def _async_rerank_hybrid(
        self,
        query: str,
        documents: List[str]
    ) -> List[Tuple[int, float]]:
        """
        Asynchronously rerank using both providers and fuse with RRF.
        
        Args:
            query: User query
            documents: List of document texts to rerank
            
        Returns:
            Fused ranking as list of (doc_index, fused_score) tuples
        """
        if self.verbose:
            print(f"\n\033[95mAsync Hybrid Reranking - Using both Cohere and Voyage\033[0m")
        
        loop = asyncio.get_event_loop()
        
        # Run both rerankers concurrently
        tasks = []
        rankings = {}
        
        # Cohere task
        async def get_cohere_rankings():
            try:
                results = await loop.run_in_executor(
                    None, self._rerank_with_cohere, query, documents
                )
                return "cohere", [(r.index, r.relevance_score) for r in results]
            except Exception as e:
                if self.verbose:
                    print(f"\033[91mCohere async reranking failed: {e}\033[0m")
                return "cohere", []
        
        # Voyage task
        async def get_voyage_rankings():
            try:
                results = await loop.run_in_executor(
                    None, self._rerank_with_voyage, query, documents
                )
                return "voyage", [(r.index, r.relevance_score) for r in results]
            except Exception as e:
                if self.verbose:
                    print(f"\033[91mVoyage async reranking failed: {e}\033[0m")
                return "voyage", []
        
        # Execute both tasks concurrently
        results = await asyncio.gather(
            get_cohere_rankings(),
            get_voyage_rankings()
        )
        
        for ranker_name, ranked_docs in results:
            rankings[ranker_name] = ranked_docs
            if self.verbose and ranked_docs:
                print(f"\033[96m{ranker_name.capitalize()} returned {len(ranked_docs)} results\033[0m")
        
        # Handle failures
        if not rankings["cohere"] and rankings["voyage"]:
            return rankings["voyage"]
        elif not rankings["voyage"] and rankings["cohere"]:
            return rankings["cohere"]
        elif not rankings["cohere"] and not rankings["voyage"]:
            raise RuntimeError("Both rerankers failed")
        
        # Fuse rankings
        return self._reciprocal_rank_fusion(rankings, self.rrf_k)
    
    def _rerank_documents(
        self, 
        query: str, 
        documents: List[str]
    ) -> Union[List[Any], List[Tuple[int, float]]]:
        """
        Rerank documents using the configured provider.
        
        Args:
            query: User query
            documents: List of document texts to rerank
            
        Returns:
            Reranked results from the provider
        """
        if self.reranker_provider == "cohere":
            return self._rerank_with_cohere(query, documents)
        elif self.reranker_provider == "voyage":
            return self._rerank_with_voyage(query, documents)
        elif self.reranker_provider == "hybrid":
            return self._rerank_hybrid(query, documents)
        else:
            raise ValueError(f"Unsupported reranker provider: {self.reranker_provider}")

    async def _async_rerank_documents(
        self, 
        query: str, 
        documents: List[str]
    ) -> Union[List[Any], List[Tuple[int, float]]]:
        """
        Asynchronously rerank documents using the configured provider.
        
        Args:
            query: User query
            documents: List of document texts to rerank
            
        Returns:
            Reranked results from the provider
        """
        if self.reranker_provider == "hybrid":
            return await self._async_rerank_hybrid(query, documents)
        else:
            # For single rerankers, use the existing logic
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                self._rerank_documents, 
                query, 
                documents
            )

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        """
        Execute the retrieval and reranking pipeline.
        
        Args:
            question: User query
            
        Returns:
            DSPyAgentRAGResponse with reranked sources
        """            
        # Get initial search results
        search_results, sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
            return_format="rerank"
        )
        
        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(search_results)} documents\033[0m")
            print(f"Query: '{question}'")
            print(f"Using {self.reranker_provider} for reranking")
        
        # Extract document content directly from search results
        documents = []
        for result in search_results:
            # SearchResult objects have a 'content' attribute
            doc_text = result.content if hasattr(result, 'content') else str(result)
            documents.append(doc_text)
            
        if self.verbose:
            print(f"\n\033[93mPreparing {len(documents)} documents for reranking...\033[0m")
            for i, doc in enumerate(documents[:3]):  # Show first 3
                preview = doc[:100] + "..." if len(doc) > 100 else doc
                print(f"  Doc {i+1} preview: {preview}")
        
        if self.summarize_query:
            question_pred = self.query_summarizer(question=question)
            question = question_pred.summary
            if self.verbose:
                print(f"\033[96mSummarized query: {question}\033[0m")
            
        # Rerank with configured provider
        reranked_results = self._rerank_documents(question, documents)
        
        if self.verbose:
            provider_name = self.reranker_provider.capitalize()
            print(f"\n\033[93m{provider_name} reranking complete.\033[0m")
        
        # Reorder sources based on reranking results
        reranked_sources = []
        
        # Handle different result formats
        if self.reranker_provider == "hybrid":
            # Hybrid mode returns list of (doc_index, fused_score) tuples
            for i, (doc_idx, score) in enumerate(reranked_results):
                if 0 <= doc_idx < len(sources):
                    reranked_sources.append(sources[doc_idx])
                    
                    if self.verbose and i < 5:
                        print(f"Rank {i + 1}: "
                              f"Document {doc_idx + 1} "
                              f"(RRF score: {score:.4f})")
        else:
            # Single reranker mode
            for i, result in enumerate(reranked_results):
                if 0 <= result.index < len(sources):
                    reranked_sources.append(sources[result.index])
                    
                    if self.verbose and i < 5:
                        print(f"Rank {i + 1}: "
                              f"Document {result.index + 1} "
                              f"(relevance: {result.relevance_score:.4f})")
        
        if self.verbose:
            print(f"\n\033[96mReranked: Returning {len(reranked_sources)} documents\033[0m")
            
            # Additional diagnostics for low scores (single reranker mode)
            if (self.reranker_provider != "hybrid" and 
                reranked_results and 
                reranked_results[0].relevance_score < 0.1):
                print(f"\033[91mWarning: Low relevance scores detected! "
                      f"Top score: {reranked_results[0].relevance_score:.4f}\033[0m")
                print("This might indicate:")
                print("- Documents don't contain relevant content for the query")
                print("- The collection might not have documents about this topic")
        
        # Return response
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage={},
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        """
        Asynchronously execute the retrieval and reranking pipeline.
        
        Args:
            question: User query
            
        Returns:
            DSPyAgentRAGResponse with reranked sources
        """            
        # Get initial search results
        search_results, sources = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
            return_format="rerank"
        )
        
        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(sources)} documents\033[0m")
            print(f"Using {self.reranker_provider} for reranking")
        
        # Extract document content directly from search results
        documents = []
        for result in search_results:
            # SearchResult objects have a 'content' attribute
            doc_text = result.content if hasattr(result, 'content') else str(result)
            documents.append(doc_text)
        
        if self.summarize_query:
            question_pred = self.query_summarizer(question=question)
            question = question_pred.summary
            if self.verbose:
                print(f"\033[96mSummarized query: {question}\033[0m")

        # Rerank with configured provider (async)
        reranked_results = await self._async_rerank_documents(question, documents)
        
        # Reorder sources based on reranking results
        reranked_sources = []
        
        # Handle different result formats
        if self.reranker_provider == "hybrid":
            # Hybrid mode returns list of (doc_index, fused_score) tuples
            for doc_idx, score in reranked_results:
                if 0 <= doc_idx < len(sources):
                    reranked_sources.append(sources[doc_idx])
                    
                    if self.verbose and len(reranked_sources) <= 5:
                        print(f"Rank {len(reranked_sources)}: "
                              f"Document {doc_idx + 1} "
                              f"(RRF score: {score:.4f})")
        else:
            # Single reranker mode
            for result in reranked_results:
                if 0 <= result.index < len(sources):
                    reranked_sources.append(sources[result.index])
                    
                    if self.verbose and len(reranked_sources) <= 5:
                        print(f"Rank {len(reranked_sources)}: "
                              f"Document {result.index + 1} "
                              f"(relevance: {result.relevance_score:.4f})")
        
        if self.verbose:
            print(f"\033[96mReranked: Returning {len(reranked_sources)} documents\033[0m")
        
        # Return response
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage={},
        )


async def main():
    """Test the Cross Encoder Reranker with all providers including hybrid"""
    import os
    
    # Test with Cohere
    if os.getenv("COHERE_API_KEY"):
        print("\n=== Testing with Cohere Reranker ===")
        cohere_reranker = CrossEncoderReranker(
            collection_name="EnronEmails",
            target_property_name="email_body_vector",
            return_property_name="email_body",
            retrieved_k=20,
            reranked_k=10,
            reranker_provider="cohere",
            verbose=True
        )
        
        test_query = "Where will Governor Gray Davis host a party for the delegates, according to the article “Davis faces dire political consequences if power woes linger?"
        
        # Test synchronous execution
        print("\n--- Synchronous Reranking (Cohere) ---")
        response = cohere_reranker.forward(test_query)
        print(f"Returned {len(response.sources)} reranked documents")
        
        # Test asynchronous execution
        print("\n--- Asynchronous Reranking (Cohere) ---")
        async_response = await cohere_reranker.aforward(test_query)
        print(f"Returned {len(async_response.sources)} reranked documents")
    
    # Test with Voyage
    if os.getenv("VOYAGE_API_KEY"):
        print("\n=== Testing with Voyage Reranker ===")
        voyage_reranker = CrossEncoderReranker(
            collection_name="EnronEmails",
            target_property_name="email_body_vector",
            return_property_name="email_body",
            retrieved_k=20,
            reranked_k=10,
            reranker_provider="voyage",
            voyage_model="rerank-2.5",
            verbose=True
        )
                
        # Test synchronous execution
        print("\n--- Synchronous Reranking (Voyage) ---")
        response = voyage_reranker.forward(test_query)
        print(f"Returned {len(response.sources)} reranked documents")
        
        # Test asynchronous execution
        print("\n--- Asynchronous Reranking (Voyage) ---")
        async_response = await voyage_reranker.aforward(test_query)
        print(f"Returned {len(async_response.sources)} reranked documents")
    
    # Test with Hybrid mode
    if os.getenv("COHERE_API_KEY") and os.getenv("VOYAGE_API_KEY"):
        print("\n=== Testing with Hybrid Reranker (RRF) ===")
        
        # Test with equal weights
        hybrid_reranker = CrossEncoderReranker(
            collection_name="EnronEmails",
            target_property_name="email_body_vector",
            return_property_name="email_body",
            retrieved_k=20,
            reranked_k=10,
            reranker_provider="hybrid",
            verbose=True,
            rrf_k=60,
            hybrid_weights={"cohere": 0.5, "voyage": 0.5}
        )
        
        # Test synchronous execution
        print("\n--- Synchronous Hybrid Reranking (Equal Weights) ---")
        response = hybrid_reranker.forward(test_query)
        print(f"Returned {len(response.sources)} reranked documents")
        
        # Test asynchronous execution
        print("\n--- Asynchronous Hybrid Reranking (Equal Weights) ---")
        async_response = await hybrid_reranker.aforward(test_query)
        print(f"Returned {len(async_response.sources)} reranked documents")
        
        # Test with weighted preference for Cohere
        print("\n=== Testing Hybrid with Cohere Preference (0.7/0.3) ===")
        weighted_reranker = CrossEncoderReranker(
            collection_name="EnronEmails",
            target_property_name="email_body_vector",
            return_property_name="email_body",
            retrieved_k=20,
            reranked_k=10,
            reranker_provider="hybrid",
            verbose=True,
            rrf_k=60,
            hybrid_weights={"cohere": 0.7, "voyage": 0.3}
        )
        
        print("\n--- Weighted Hybrid Reranking ---")
        response = weighted_reranker.forward(test_query)
        print(f"Returned {len(response.sources)} reranked documents")


if __name__ == "__main__":
    asyncio.run(main())