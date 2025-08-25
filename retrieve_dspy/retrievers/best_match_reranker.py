import asyncio
import os

import dspy


from retrieve_dspy.models import DSPyAgentRAGResponse, SearchResult
from retrieve_dspy.signatures import BestMatchRanker

class BestMatchReranker(dspy.Module):
    def __init__(
        self,
        verbose: bool = False,
    ):
        # init LLM
        self.lm = dspy.LM("openai/gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY"))
        dspy.configure(lm=self.lm, track_usage=True)

        self.verbose = verbose
        self.reranker = dspy.ChainOfThought(BestMatchRanker) # update to send rationale through to metric

    def forward(self, question: str, candidates: list[SearchResult]) -> DSPyAgentRAGResponse:
        # Perform reranking
        rerank_pred = self.reranker(
            query=question,
            search_results=candidates,
        )
        
        # Find the best match result based on the returned ID
        best_match_result = None
        for candidate in candidates:
            if candidate.id == rerank_pred.best_match_id:
                best_match_result = candidate
                break
        
        reranked_sources = [best_match_result] if best_match_result else []
        
        if self.verbose:
            print(f"\033[96mReranked: Returning {len(reranked_sources)} Sources!\033[0m")
            if best_match_result:
                # Find the original position of this result in candidates
                original_rank = candidates.index(best_match_result) + 1  # +1 for 1-based ranking
                print(f"Best match ID: {rerank_pred.best_match_id} (was rank {original_rank})")
        
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

    test_pipeline = BestMatchReranker(
        verbose=True,
    )
    test_q = "What number did David Ortiz wear when he played for the Boston Red Sox?"
    candidates = [
        SearchResult(id=1, content="David Ortiz wore the number 34 when he played for the Boston Red Sox."),
        SearchResult(id=2, content="Derek Jeter wore the number 2 for the New York Yankees throughout his career."),
        SearchResult(id=3, content="The Boston Red Sox retired David Ortiz's number 34 in 2017, making him the 11th player to receive this honor."),
    ]
    response = test_pipeline.forward(test_q, candidates)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())