import asyncio
import os
from typing import Optional, List

import dspy
import cohere
from cohere import RerankResponseResultsItem

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)
from retrieve_dspy.retrievers.base_rag import BaseRAG   
from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import ExpandQuery


class QueryExpanderWithReranker(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 50,
        reranked_k: Optional[int] = 20,
        cohere_model: Optional[str] = "rerank-v3.5",
        cohere_api_key: Optional[str] = None
    ):
        super().__init__(
            collection_name=collection_name,
            target_property_name=target_property_name,
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k,
        )
        self.reranked_k = reranked_k
        self.cohere_model = cohere_model
        
        self.expand_query = dspy.Predict(ExpandQuery)
        
        api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY must be provided or set as environment variable")
        
        self.co = cohere.ClientV2(api_key)
    
    def _rerank_with_cohere(
        self,
        query: str,
        documents: List[str]
    ) -> List[RerankResponseResultsItem]:
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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._rerank_with_cohere,
            query,
            documents
        )

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        expanded_query = self.expand_query(question=question).expanded_query

        if self.verbose:
            print(f"\033[95mExpanded query from:\n'{question}'\nto:\n'{expanded_query}'\033[0m")

        sources = weaviate_search_tool(
            query=expanded_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(sources)} documents\033[0m")

        documents = [s.content for s in sources]

        if self.verbose:
            print(f"\n\033[93mPreparing {len(documents)} documents for reranking...\033[0m")
            for i, doc in enumerate(documents[:3]):  # Show first 3
                preview = doc[:100] + "..." if len(doc) > 100 else doc
                print(f"  Doc {i+1} preview: {preview}")

        reranked_results = self._rerank_with_cohere(question, documents)

        if self.verbose:
            print("\n\033[93mCohere reranking complete. Top scores:\033[0m")

        reranked_sources = []
        for i, result in enumerate(reranked_results):
            if 0 <= result.index < len(sources):
                reranked_sources.append(sources[result.index])

                if self.verbose and i < 5:
                    print(f"Rank {i + 1}: "
                          f"Document {result.index + 1} "
                          f"(relevance: {result.relevance_score:.4f})")

        if self.verbose:
            print(f"\n\033[96mPipeline complete: Returning {len(reranked_sources)} reranked documents\033[0m")
            
            if reranked_results and reranked_results[0].relevance_score < 0.1:
                print(f"\033[91mWarning: Low relevance scores detected! "
                      f"Top score: {reranked_results[0].relevance_score:.4f}\033[0m")
                print("This might indicate:")
                print("- Documents don't contain relevant content for the query")
                print("- The collection might not have documents about this topic")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question, expanded_query],
            aggregations=None,
            usage={},
        )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        expanded_query_pred = await self.expand_query.acall(question=question)
        expanded_query = expanded_query_pred.expanded_query

        if self.verbose:
            print(f"\033[95mExpanded query from:\n'{question}'\nto:\n'{expanded_query}'\033[0m")

        sources = await async_weaviate_search_tool(
            query=expanded_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(sources)} documents\033[0m")

        documents = [s.content for s in sources]

        reranked_results = await self._async_rerank_with_cohere(question, documents)

        if self.verbose:
            print("\n\033[93mCohere reranking complete.\033[0m")

        reranked_sources = []
        for result in reranked_results:
            if 0 <= result.index < len(sources):
                reranked_sources.append(sources[result.index])

                if self.verbose and len(reranked_sources) <= 5:
                    print(f"Rank {len(reranked_sources)}: "
                          f"Document {result.index + 1} "
                          f"(relevance: {result.relevance_score:.4f})")

        if self.verbose:
            print(f"\033[96mPipeline complete: Returning {len(reranked_sources)} reranked documents\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question, expanded_query],
            aggregations=None,
            usage={},
        )


async def main():
    import os
    
    if not os.getenv("COHERE_API_KEY"):
        raise ValueError("COHERE_API_KEY environment variable is required")
    
    pipeline = QueryExpanderWithReranker(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=30,
        reranked_k=10,
        verbose=True
    )
    
    test_query = "How do I integrate Weaviate and Langchain?"
    
    print("\n" + "="*60)
    print("--- Synchronous Query Expansion + Reranking ---")
    print("="*60)
    response = pipeline.forward(test_query)
    print(f"\nFinal result: {len(response.sources)} reranked documents")
    print(f"Queries used: {response.searches}")
    
    print("\n" + "="*60)
    print("--- Asynchronous Query Expansion + Reranking ---")
    print("="*60)
    async_response = await pipeline.aforward(test_query)
    print(f"\nFinal result: {len(async_response.sources)} reranked documents")
    print(f"Queries used: {async_response.searches}")


if __name__ == "__main__":
    asyncio.run(main())