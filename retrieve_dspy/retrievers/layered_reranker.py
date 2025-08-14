import asyncio
import os
from typing import Optional

import cohere
from cohere import RerankResponseResultsItem
import dspy
import re

from retrieve_dspy.tools.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.retrievers.base_rag import BaseRAG

from retrieve_dspy.models import DSPyAgentRAGResponse, SearchResult
from retrieve_dspy.signatures import DiversityRanker


class LayeredReranker(BaseRAG):
    def __init__(
        self, 
        collection_name: str, 
        target_property_name: str, 
        verbose: bool = False,
        search_only: bool = True,
        retrieved_k: int = 100,
        reranked_N: int = 50,
        reranked_M: int = 20,
        cohere_model: str = "rerank-v3.5",
        cohere_api_key: Optional[str] = None
    ):
        super().__init__(
            collection_name=collection_name,
            target_property_name=target_property_name,
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k
        )
        self.reranked_N = reranked_N
        self.reranked_M = reranked_M
        self.cohere_model = cohere_model
        
        # Initialize Cohere client
        api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY must be provided or set as environment variable")
        
        self.co = cohere.ClientV2(api_key)
        
        # Initialize diversity ranker
        self.diversity_ranker = dspy.Predict(DiversityRanker)

    def _rerank_with_cohere(
        self, 
        query: str, 
        documents: list[str],
        top_n: int
    ) -> list[RerankResponseResultsItem]:
        """
        Rerank documents using Cohere's Cross Encoder.
        
        Args:
            query: User query
            documents: List of document texts to rerank
            top_n: Number of top documents to return
            
        Returns:
            Reranked results from Cohere
        """
        try:
            response = self.co.rerank(
                model=self.cohere_model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents))
            )
            return response.results
        except Exception as e:
            if self.verbose:
                print(f"\033[91mError during Cohere reranking: {e}\033[0m")
            raise

    async def _async_rerank_with_cohere(
        self, 
        query: str, 
        documents: list[str],
        top_n: int
    ) -> list[RerankResponseResultsItem]:
        """
        Asynchronously rerank documents using Cohere's Cross Encoder.
        
        Args:
            query: User query
            documents: List of document texts to rerank
            top_n: Number of top documents to return
            
        Returns:
            Reranked results from Cohere
        """
        # Cohere SDK doesn't have native async support, so we run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._rerank_with_cohere, 
            query, 
            documents,
            top_n
        )

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        # first search with the original query
        search_results, sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            return_format="rerank"
        )
        
        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(search_results)} documents\033[0m")
        
        # Extract document content for reranking
        documents = []
        for result in search_results:
            doc_text = result.content if hasattr(result, 'content') else str(result)
            documents.append(doc_text)
        
        # then apply the cross encoder reranker to truncate the results to N
        reranked_results = self._rerank_with_cohere(question, documents, self.reranked_N)
        
        # Reorder sources based on Cohere's reranking
        cross_encoder_sources = []
        for result in reranked_results:
            if 0 <= result.index < len(sources):
                cross_encoder_sources.append(sources[result.index])
        
        if self.verbose:
            print(f"\033[93mCross encoder reranking: {len(cross_encoder_sources)} documents\033[0m")
        
        # then apply the diversity ranker to truncate the results to M
        if len(cross_encoder_sources) > self.reranked_M:
            # Prepare SearchResult objects for diversity ranking
            cross_encoder_search_results = []
            for i, source in enumerate(cross_encoder_sources):
                if hasattr(source, 'content'):
                    content = source.content
                elif hasattr(source, 'text'):
                    content = source.text
                else:
                    content = str(source)
                
                search_result = SearchResult(
                    id=i,
                    initial_rank=i,
                    content=content
                )
                cross_encoder_search_results.append(search_result)
            
            # Apply diversity ranking
            diversity_result = self.diversity_ranker(
                query=question,
                search_results=cross_encoder_search_results,
                top_k=self.reranked_M
            )
            
            # Extract the ranked document indices
            ranked_indices = []
            if hasattr(diversity_result, 'reranked_ids'):
                ranked_indices = diversity_result.reranked_ids
            elif hasattr(diversity_result, 'prediction'):
                # Parse the prediction to extract indices
                prediction = diversity_result.prediction
                if isinstance(prediction, str):
                    indices = re.findall(r'\d+', prediction)
                    ranked_indices = [int(i) for i in indices if int(i) < len(cross_encoder_sources)]

            
            # Reorder sources based on diversity ranking
            final_sources = []
            for idx in ranked_indices[:self.reranked_M]:
                if 0 <= idx < len(cross_encoder_sources):
                    final_sources.append(cross_encoder_sources[idx])
            
            # Fill remaining slots if needed
            if len(final_sources) < self.reranked_M:
                for i, source in enumerate(cross_encoder_sources):
                    if i not in ranked_indices and len(final_sources) < self.reranked_M:
                        final_sources.append(source)
        else:
            final_sources = cross_encoder_sources
        
        if self.verbose:
            print(f"\033[96mDiversity reranking: Returning {len(final_sources)} documents\033[0m")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_sources,
            searches=[question],
            aggregations=None,
            usage={},
        )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        # first search with the original query
        search_results, sources = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            return_format="rerank"
        )
        
        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(search_results)} documents\033[0m")
        
        # Extract document content for reranking
        documents = []
        for result in search_results:
            doc_text = result.content if hasattr(result, 'content') else str(result)
            documents.append(doc_text)
        
        # then apply the cross encoder reranker to truncate the results to N
        reranked_results = await self._async_rerank_with_cohere(question, documents, self.reranked_N)
        
        # Reorder sources based on Cohere's reranking
        cross_encoder_sources = []
        for result in reranked_results:
            if 0 <= result.index < len(sources):
                cross_encoder_sources.append(sources[result.index])
        
        if self.verbose:
            print(f"\033[93mCross encoder reranking: {len(cross_encoder_sources)} documents\033[0m")
        
        # then apply the diversity ranker to truncate the results to M
        if len(cross_encoder_sources) > self.reranked_M:
            # Prepare SearchResult objects for diversity ranking
            cross_encoder_search_results = []
            for i, source in enumerate(cross_encoder_sources):
                if hasattr(source, 'content'):
                    content = source.content
                elif hasattr(source, 'text'):
                    content = source.text
                else:
                    content = str(source)
                
                search_result = SearchResult(
                    id=i,
                    initial_rank=i,
                    content=content
                )
                cross_encoder_search_results.append(search_result)
            
            # Apply diversity ranking
            diversity_result = self.diversity_ranker(
                query=question,
                search_results=cross_encoder_search_results,
                top_k=self.reranked_M
            )
            
            # Extract the ranked document indices
            ranked_indices = []
            if hasattr(diversity_result, 'reranked_ids'):
                ranked_indices = diversity_result.reranked_ids
            elif hasattr(diversity_result, 'prediction'):
                # Parse the prediction to extract indices
                prediction = diversity_result.prediction
                if isinstance(prediction, str):
                    indices = re.findall(r'\d+', prediction)
                    ranked_indices = [int(i) for i in indices if int(i) < len(cross_encoder_sources)]
            
            # Reorder sources based on diversity ranking
            final_sources = []
            for idx in ranked_indices[:self.reranked_M]:
                if 0 <= idx < len(cross_encoder_sources):
                    final_sources.append(cross_encoder_sources[idx])
            
            # Fill remaining slots if needed
            if len(final_sources) < self.reranked_M:
                for i, source in enumerate(cross_encoder_sources):
                    if i not in ranked_indices and len(final_sources) < self.reranked_M:
                        final_sources.append(source)
        else:
            final_sources = cross_encoder_sources
        
        if self.verbose:
            print(f"\033[96mDiversity reranking: Returning {len(final_sources)} documents\033[0m")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_sources,
            searches=[question],
            aggregations=None,
            usage={},
        )

async def main():
    rag_pipeline = LayeredReranker(
        collection_name="FreshstackAngular",
        target_property_name="docs_text",
        retrieved_k=100,
        reranked_N=50,
        reranked_M=20,
        cohere_model="rerank-v3.5",
    )
    print("Testing sync forward")
    response = rag_pipeline.forward("What is the best way to learn Angular?")
    print(response)
    print("Testing async forward")
    response = await rag_pipeline.aforward("What is the best way to learn Angular?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())