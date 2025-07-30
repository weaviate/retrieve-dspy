import asyncio
from typing import Optional

import dspy

from retrieve_dspy.tools.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import SummarizeSearchRelevance, RerankWithSummaries

class SummarizedListwiseReranker(BaseRAG):
    def __init__(
        self, 
        collection_name: str, 
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 100,
        reranked_k: Optional[int] = 20 
    ):
        super().__init__(collection_name, target_property_name, retrieved_k=retrieved_k)
        self.summarizer = dspy.Predict(SummarizeSearchRelevance)
        self.reranker = dspy.Predict(RerankWithSummaries)
        self.retrieved_k = retrieved_k
        self.reranked_k = reranked_k

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        # Get search results
        search_results, sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            return_format="rerank"
        )
        
        if self.verbose:
            print(f"\033[96mInitial results: {len(sources)} Sources!\033[0m")
        
        # Summarize relevance for each result
        summaries = []
        total_usage = {}
        
        for i, (result, source) in enumerate(zip(search_results, sources)):
            summary_pred = self.summarizer(
                query=question,
                passage=result.content,
                passage_id=result.id
            )
            
            summaries.append({
                "passage_id": result.id,
                "relevance_summary": summary_pred.relevance_summary,
                "relevance_score": summary_pred.relevance_score
            })
            
            # Aggregate usage
            total_usage = summary_pred.get_lm_usage() or {}
        
        if self.verbose:
            print(f"\033[96mGenerated {len(summaries)} relevance summaries\033[0m")
        
        # Perform reranking based on summaries
        rerank_pred = self.reranker(
            query=question,
            passage_summaries=summaries,
            top_k=self.reranked_k
        )
        
        # Aggregate reranker usage
        total_usage = rerank_pred.get_lm_usage() or {}
        
        # Reorder sources based on reranking
        reranked_sources = []
        for rank_id in rerank_pred.reranked_ids:
            # Find the source corresponding to this rank_id
            for i, source in enumerate(sources):
                if search_results[i].id == rank_id:
                    reranked_sources.append(source)
                    break
        
        if self.verbose:
            print(f"\033[96mReranked: Returning {len(reranked_sources)} Sources!\033[0m")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage=total_usage,
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        # Get search results
        search_results, sources = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            return_format="rerank"
        )
        
        if self.verbose:
            print(f"\033[96mInitial results: {len(sources)} Sources!\033[0m")
        
        # Summarize relevance for each result in parallel
        summary_tasks = []
        passage_ids = []  # Track passage IDs separately
        for i, (result, source) in enumerate(zip(search_results, sources)):
            task = self.summarizer.acall(
                query=question,
                passage=result.content,
                passage_id=result.id
            )
            summary_tasks.append(task)
            passage_ids.append(result.id)  # Store the ID
        
        # Wait for all summaries to complete
        summary_preds = await asyncio.gather(*summary_tasks)
        
        # Process summaries and aggregate usage
        summaries = []
        total_usage = {}
        
        for i, summary_pred in enumerate(summary_preds):
            summaries.append({
                "passage_id": passage_ids[i],  # Use stored ID
                "relevance_summary": summary_pred.relevance_summary,
                "relevance_score": summary_pred.relevance_score
            })
            
            # Aggregate usage
            total_usage = summary_pred.get_lm_usage() or {}
        
        if self.verbose:
            print(f"\033[96mGenerated {len(summaries)} relevance summaries in parallel\033[0m")
        
        # Perform reranking based on summaries
        rerank_pred = await self.reranker.acall(
            query=question,
            passage_summaries=summaries,
            top_k=self.reranked_k
        )
        
        # Aggregate reranker usage
        total_usage = rerank_pred.get_lm_usage() or {}
        
        # Reorder sources based on reranking
        reranked_sources = []
        for rank_id in rerank_pred.reranked_ids:
            # Find the source corresponding to this rank_id
            for i, source in enumerate(sources):
                if search_results[i].id == rank_id:
                    reranked_sources.append(source)
                    break
        
        if self.verbose:
            print(f"\033[96mReranked: Returning {len(reranked_sources)} Sources!\033[0m")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage=total_usage,
        )

async def main():
    import os
    import dspy
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    lm = dspy.LM("openai/gpt-4.1-mini", api_key=openai_api_key)
    dspy.configure(lm=lm, track_usage=True)
    print(f"DSPy configured with: {lm}")
    
    test_pipeline = SummarizedListwiseReranker(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=50,
        reranked_k=20
    )
    test_q = "How do I integrate Weaviate and Langchain?"
    response = test_pipeline.forward(test_q)
    print(response)
    async_response = await test_pipeline.aforward(test_q)
    print(async_response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())