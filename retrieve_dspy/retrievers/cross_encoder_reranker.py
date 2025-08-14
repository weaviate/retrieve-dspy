import asyncio
import os
from typing import Optional, List

import cohere
from cohere import RerankResponseResultsItem
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
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 50,
        reranked_k: Optional[int] = 20,
        cohere_model: Optional[str] = "rerank-v3.5",
        cohere_api_key: Optional[str] = None,
        summarize_query: Optional[bool] = False
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
            cohere_model: Cohere reranking model to use
            cohere_api_key: Cohere API key (defaults to COHERE_API_KEY env var)
        """
        super().__init__(
            collection_name=collection_name, 
            target_property_name=target_property_name, 
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k,
        )
        self.reranked_k = reranked_k
        self.cohere_model = cohere_model
        self.summarize_query = summarize_query
        self.query_summarizer = dspy.Predict(QuerySummarizer)
        
        # Initialize Cohere client
        api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY must be provided or set as environment variable")
        
        self.co = cohere.ClientV2(api_key)
    
    def _rerank_with_cohere(
        self, 
        query: str, 
        documents: List[str]
    ) -> List[RerankResponseResultsItem]:
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

    async def _async_rerank_with_cohere(
        self, 
        query: str, 
        documents: List[str]
    ) -> List[RerankResponseResultsItem]:
        """
        Asynchronously rerank documents using Cohere's Cross Encoder.
        
        Args:
            query: User query
            documents: List of document texts to rerank
            
        Returns:
            Reranked results from Cohere
        """
        # Cohere SDK doesn't have native async support, so we run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._rerank_with_cohere, 
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
            retrieved_k=self.retrieved_k,
            return_format="rerank"
        )
        
        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(search_results)} documents\033[0m")
            print(f"Query: '{question}'")
        
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
            
        # Rerank with Cohere
        reranked_results = self._rerank_with_cohere(question, documents)
        
        if self.verbose:
            print("\n\033[93mCohere reranking complete. Top scores:\033[0m")
        
        # Reorder sources based on Cohere's reranking
        reranked_sources = []
        for i, result in enumerate(reranked_results):
            # Cohere returns 0-based indices
            if 0 <= result.index < len(sources):
                reranked_sources.append(sources[result.index])
                
                if self.verbose and i < 5:
                    print(f"Rank {i + 1}: "
                          f"Document {result.index + 1} "
                          f"(relevance: {result.relevance_score:.4f})")
        
        if self.verbose:
            print(f"\n\033[96mReranked: Returning {len(reranked_sources)} documents\033[0m")
            
            # Additional diagnostics for low scores
            if reranked_results and reranked_results[0].relevance_score < 0.1:
                print(f"\033[91mWarning: Low relevance scores detected! "
                      f"Top score: {reranked_results[0].relevance_score:.4f}\033[0m")
                print("This might indicate:")
                print("- Documents don't contain relevant content for the query")
                print("- The collection might not have documents about this topic")
        
        # Return response
        # Note: We don't have token usage info from Cohere's rerank API
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage={},  # Cohere rerank doesn't provide token usage
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
            retrieved_k=self.retrieved_k,
            return_format="rerank"
        )
        
        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(sources)} documents\033[0m")
        
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

        # Rerank with Cohere (async)
        reranked_results = await self._async_rerank_with_cohere(question, documents)
        
        # Reorder sources based on Cohere's reranking
        reranked_sources = []
        for result in reranked_results:
            # Cohere returns 0-based indices
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
            usage={},  # Cohere rerank doesn't provide token usage
        )


async def main():
    """Test the Cross Encoder Reranker"""
    import os
    
    # Ensure API keys are set
    if not os.getenv("COHERE_API_KEY"):
        raise ValueError("COHERE_API_KEY environment variable is required")
    
    # Initialize the reranker
    reranker = CrossEncoderReranker(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=20,
        reranked_k=10,
        verbose=True
    )
    
    # Test query
    test_query = "How do I integrate Weaviate and Langchain?"
    
    # Test synchronous execution
    print("\n--- Synchronous Reranking ---")
    response = reranker.forward(test_query)
    print(f"Returned {len(response.sources)} reranked documents")
    
    # Test asynchronous execution
    print("\n--- Asynchronous Reranking ---")
    async_response = await reranker.aforward(test_query)
    print(f"Returned {len(async_response.sources)} reranked documents")


if __name__ == "__main__":
    asyncio.run(main())