import asyncio
import os
from typing import Optional, List, Literal, Union, Any

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
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 50,
        reranked_k: Optional[int] = 20,
        reranker_provider: Literal["cohere", "voyage"] = "cohere",
        cohere_model: Optional[str] = "rerank-v3.5",
        voyage_model: Optional[str] = "rerank-2.5",
        cohere_api_key: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
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
            reranker_provider: Which reranker to use ("cohere" or "voyage")
            cohere_model: Cohere reranking model to use
            voyage_model: Voyage reranking model to use
            cohere_api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            voyage_api_key: Voyage API key (defaults to VOYAGE_API_KEY env var)
            summarize_query: Whether to summarize the query before reranking
        """
        super().__init__(
            collection_name=collection_name, 
            target_property_name=target_property_name, 
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k,
        )
        self.reranked_k = reranked_k
        self.reranker_provider = reranker_provider
        self.cohere_model = cohere_model
        self.voyage_model = voyage_model
        self.summarize_query = summarize_query
        self.query_summarizer = dspy.Predict(QuerySummarizer)
        
        # Initialize the appropriate client based on provider
        if reranker_provider == "cohere":
            api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("COHERE_API_KEY must be provided or set as environment variable")
            self.co = cohere.ClientV2(api_key)
            
        elif reranker_provider == "voyage":
            api_key = voyage_api_key or os.getenv("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError("VOYAGE_API_KEY must be provided or set as environment variable")
            self.vo = voyageai.Client(api_key=api_key)
        else:
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
    
    def _rerank_documents(
        self, 
        query: str, 
        documents: List[str]
    ) -> Union[List[Any], List[Any]]:
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
        else:
            raise ValueError(f"Unsupported reranker provider: {self.reranker_provider}")

    async def _async_rerank_documents(
        self, 
        query: str, 
        documents: List[str]
    ) -> Union[List[Any], List[Any]]:
        """
        Asynchronously rerank documents using the configured provider.
        
        Args:
            query: User query
            documents: List of document texts to rerank
            
        Returns:
            Reranked results from the provider
        """
        # Neither Cohere nor Voyage SDK has native async support, so we run in executor
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
            print(f"\n\033[93m{provider_name} reranking complete. Top scores:\033[0m")
        
        # Reorder sources based on reranking results
        reranked_sources = []
        for i, result in enumerate(reranked_results):
            # Both Cohere and Voyage return 0-based indices
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
        # Note: Neither Cohere nor Voyage rerank APIs provide token usage info
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
        for result in reranked_results:
            # Both Cohere and Voyage return 0-based indices
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
    """Test the Cross Encoder Reranker with both providers"""
    import os
    
    # Test with Cohere
    if os.getenv("COHERE_API_KEY"):
        print("\n=== Testing with Cohere Reranker ===")
        cohere_reranker = CrossEncoderReranker(
            collection_name="FreshstackLangchain",
            target_property_name="docs_text",
            retrieved_k=20,
            reranked_k=10,
            reranker_provider="cohere",
            verbose=True
        )
        
        test_query = "How do I integrate Weaviate and Langchain?"
        
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
            collection_name="FreshstackLangchain",
            target_property_name="docs_text",
            retrieved_k=20,
            reranked_k=10,
            reranker_provider="voyage",
            voyage_model="rerank-2.5",
            verbose=True
        )
        
        test_query = "How do I integrate Weaviate and Langchain?"
        
        # Test synchronous execution
        print("\n--- Synchronous Reranking (Voyage) ---")
        response = voyage_reranker.forward(test_query)
        print(f"Returned {len(response.sources)} reranked documents")
        
        # Test asynchronous execution
        print("\n--- Asynchronous Reranking (Voyage) ---")
        async_response = await voyage_reranker.aforward(test_query)
        print(f"Returned {len(async_response.sources)} reranked documents")


if __name__ == "__main__":
    asyncio.run(main())