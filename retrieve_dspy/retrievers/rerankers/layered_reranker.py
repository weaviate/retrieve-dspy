import asyncio
import os
import re
from typing import Optional, Any

import dspy
import voyageai

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool
)
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse, SearchResult
from retrieve_dspy.signatures import RelevanceRanker, IdentifyMostRelevantPassage


class LayeredReranker(BaseRAG):
    def __init__(
        self, 
        collection_name: str, 
        target_property_name: str, 
        return_property_name: str,
        verbose: bool = False,
        search_only: bool = True,
        retrieved_k: int = 100,
        reranked_N: int = 50,
        reranked_M: int = 20,
        voyage_model: str = "rerank-2.5",
        voyage_api_key: Optional[str] = None
    ):
        super().__init__(
            collection_name=collection_name,
            target_property_name=target_property_name,
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k
        )
        self.return_property_name = return_property_name
        self.reranked_N = reranked_N
        self.reranked_M = reranked_M
        self.voyage_model = voyage_model
        
        # Initialize Cohere client
        api_key = voyage_api_key or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY must be provided or set as environment variable")
        
        # Need Async Client for async case here
        self.vo = voyageai.Client(api_key)
        
        # Initialize Listwise Reranker
        if self.reranked_M == 1:
            self.listwise_reranker = dspy.Predict(IdentifyMostRelevantPassage)
        else:
            self.listwise_reranker = dspy.Predict(RelevanceRanker)

    def _rerank_with_voyage(
        self, 
        query: str, 
        documents: list[str]
    ) -> list[Any]:
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
                top_k=min(self.reranked_N, len(documents))
            )
            return response.results
        except Exception as e:
            if self.verbose:
                print(f"\033[91mError during Voyage reranking: {e}\033[0m")
            raise

    async def _async_rerank_with_voyage(
        self, 
        query: str, 
        documents: list[str],
    ):
        pass

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        # first search with the original query
        sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
        )
        
        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(sources)} documents\033[0m")
        
        # Extract document content for reranking
        documents = [s.content for s in sources]
        
        # then apply the cross encoder reranker to truncate the results to N
        reranked_results = self._rerank_with_voyage(question, documents)
        
        # Reorder sources based on Cohere's reranking
        cross_encoder_sources = []
        for result in reranked_results:
            if 0 <= result.index < len(sources):
                cross_encoder_sources.append(sources[result.index])
        
        if self.verbose:
            print(f"\033[93mCross encoder reranking: {len(cross_encoder_sources)} documents\033[0m")
        
        if len(cross_encoder_sources) > self.reranked_M:
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
            
            listwise_reranked_result = self.listwise_reranker(
                query=question,
                search_results=cross_encoder_search_results,
                top_k=self.reranked_M
            )
            
            if self.reranked_M == 1:
                final_sources = [cross_encoder_sources[listwise_reranked_result.most_relevant_passage]]
                if self.verbose:
                    print(f"\033[92mListwise reranking: Returning {len(final_sources)} documents\033[0m")
                return DSPyAgentRAGResponse(
                    final_answer="",
                    sources=final_sources,
                    searches=[question],
                    aggregations=None,
                    usage={},
                )


            ranked_indices = []
            if hasattr(listwise_reranked_result, 'reranked_ids'):
                ranked_indices = listwise_reranked_result.reranked_ids
            elif hasattr(listwise_reranked_result, 'prediction'):
                prediction = listwise_reranked_result.prediction
                if isinstance(prediction, str):
                    indices = re.findall(r'\d+', prediction)
                    ranked_indices = [int(i) for i in indices if int(i) < len(cross_encoder_sources)]

            
            final_sources = []
            for idx in ranked_indices[:self.reranked_M]:
                if 0 <= idx < len(cross_encoder_sources):
                    final_sources.append(cross_encoder_sources[idx])
            
            if len(final_sources) < self.reranked_M:
                for i, source in enumerate(cross_encoder_sources):
                    if i not in ranked_indices and len(final_sources) < self.reranked_M:
                        final_sources.append(source)
            
            if self.verbose:
                print(f"\033[92mListwise reranking: Returning {len(final_sources)} documents\033[0m")
        
        else:
            final_sources = cross_encoder_sources[:self.reranked_M]
        
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_sources,
            searches=[question],
            aggregations=None,
            usage={},
        )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        pass

async def main():
    rag_pipeline = LayeredReranker(
        collection_name="EnronEmails",
        target_property_name="email_body_vector",
        return_property_name="email_body",
        retrieved_k=50,
        reranked_N=20,
        reranked_M=5,
        voyage_model="rerank-2.5",
        verbose=True
    )
    print("Testing sync forward")
    test_query = "Where will Governor Gray Davis host a party for the delegates, according to the article “Davis faces dire political consequences if power woes linger?"
    response = rag_pipeline.forward(test_query)
    print(response)
    #print("Testing async forward")
    #response = await rag_pipeline.aforward("What is the best way to learn Angular?")
    #print(response)

if __name__ == "__main__":
    asyncio.run(main())