import asyncio
from typing import Optional

import dspy

from retrieve_dspy.tools.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.models import DSPyAgentRAGResponse, SearchResult
from retrieve_dspy.signatures import RelevanceRanker

class IsolatedListwiseReranker():
    def __init__(
        self, 
        reranked_k: int = 1,
    ):
        self.reranked_k = reranked_k
        self.reranker = dspy.Predict(RelevanceRanker)

    def forward(self, question: str, candidates: list[SearchResult]) -> DSPyAgentRAGResponse:
        # Perform reranking
        rerank_pred = self.reranker(
            query=question,
            search_results=candidates,
            top_k=self.reranked_k,
        )
        
        # Reorder sources based on reranking
        reranked_sources = []
        reranked_results = []
        for rank_id in rerank_pred.reranked_ids:
            # Find the source corresponding to this rank_id
            # rank_id is 1-based, sources list is 0-based
            source_index = rank_id - 1
            if 0 <= source_index < len(candidates):
                reranked_sources.append(candidates[source_index])
                reranked_results.append(candidates[source_index])
        
        if self.verbose:
            print(f"\033[96mReranked: Returning {len(reranked_sources)} Sources!\033[0m")
            print("\nTop 5 reranked results:")
            for i, result in enumerate(reranked_results[:5], 1):
                print(f"New Rank {i} (was {result.initial_rank}).")
        
        # Get usage from reranker
        usage = rerank_pred.get_lm_usage() or {}
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage=usage,
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        pass

async def main():
    import os
    import dspy
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    lm = dspy.LM("openai/gpt-4.1-mini", api_key=openai_api_key)
    dspy.configure(lm=lm, track_usage=True)
    print(f"DSPy configured with: {lm}")

    test_pipeline = IsolatedListwiseReranker(
        reranked_k=1,
    )
    test_q = "What number did David Ortiz wear when he played for the Boston Red Sox?"
    candidates = [
        SearchResult(id=1, content="David Ortiz wore the number 34 when he played for the Boston Red Sox."),
        SearchResult(id=2, content="Derek Jeter wore the number 2 for the New York Yankees throughout his career."),
        SearchResult(id=3, content="The Boston Red Sox retired David Ortiz's number 34 in 2017, making him the 11th player to receive this honor."),
    ]
    response = test_pipeline.forward(test_q, candidates)
    print(response)
    async_response = await test_pipeline.aforward(test_q)
    print(async_response)

if __name__ == "__main__":
    asyncio.run(main())