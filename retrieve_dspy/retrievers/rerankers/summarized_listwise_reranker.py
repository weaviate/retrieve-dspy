import asyncio
from typing import Optional

import dspy

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB
from retrieve_dspy.signatures import SummarizeSearchRelevance, RelevanceRanker

def aggregate_usage(total_usage: dict, new_usage: dict) -> dict:
    """Helper function to safely aggregate usage dictionaries with nested structures."""
    for key, value in new_usage.items():
        if key not in total_usage:
            total_usage[key] = value
        elif isinstance(value, dict) and isinstance(total_usage[key], dict):
            # Recursively aggregate nested dictionaries
            total_usage[key] = aggregate_usage(total_usage[key], value)
        elif isinstance(value, (int, float)) and isinstance(total_usage[key], (int, float)):
            # Add numeric values
            total_usage[key] = total_usage[key] + value
        else:
            # For other types, just overwrite (or keep the existing value)
            total_usage[key] = value
    return total_usage

class SummarizedListwiseReranker(BaseRAG):
    def __init__(
        self, 
        collection_name: str, 
        target_property_name: Optional[str] = "content",
        return_property_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 100,
        reranked_k: Optional[int] = 20 
    ):
        super().__init__(
            collection_name=collection_name, 
            target_property_name=target_property_name, 
            retrieved_k=retrieved_k, 
            verbose=verbose,
            search_only=search_only
        )
        self.return_property_name = return_property_name
        self.summarizer = dspy.Predict(SummarizeSearchRelevance)
        self.reranker = dspy.ChainOfThought(RelevanceRanker)
        self.retrieved_k = retrieved_k
        self.reranked_k = reranked_k

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        # Get search results
        sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
        )
        
        if self.verbose:
            print(f"\033[92mQuestion: {question}\033[0m")
            print(f"\033[96mInitial results: {len(sources)} Sources!\033[0m")
        
        # Summarize relevance for each result
        summaries = []
        total_usage = {}
        
        for i, source in enumerate(sources):
            summary_pred = self.summarizer(
                query=question,
                passage=source.content,
            )
            
            summaries.append({
                "passage_id": i + 1,
                "initial_rank": i,
                "relevance_summary": summary_pred.relevance_summary,
            })
            
            # Aggregate usage from each summarization call
            summary_usage = summary_pred.get_lm_usage()
            if summary_usage:
                total_usage = aggregate_usage(total_usage, summary_usage)
        
        if self.verbose:
            print(f"\033[96mGenerated {len(summaries)} relevance summaries\033[0m")
            for summary in summaries:
                print(f"\033[96m{summary['relevance_summary']}\033[0m\n")
        
        # Convert summaries to list of SearchResult objects for reranker
        search_results_list = [ObjectFromDB(
            id=result["passage_id"],
            initial_rank=result["initial_rank"],
            content=result["relevance_summary"]
        ) for result in summaries]

        rerank_pred = self.reranker(
            query=question,
            search_results=search_results_list,
            top_k=self.reranked_k
        )
        
        # Aggregate reranker usage
        reranker_usage = rerank_pred.get_lm_usage()
        if reranker_usage:
            total_usage = aggregate_usage(total_usage, reranker_usage)
        
        # Reorder sources based on reranking
        reranked_sources = []
        for rank_id in rerank_pred.reranked_ids:
            # Find the source corresponding to this rank_id
            for i, source in enumerate(sources):
                if search_results_list[i].id == rank_id:
                    reranked_sources.append(source)
                    break
        
        if self.verbose:
            print(f"\033[96mReranked: Returning {len(reranked_sources)} Sources!\033[0m")
            reranker_cot = rerank_pred.reasoning
            print(f"\033[96mReranker COT: {reranker_cot}\033[0m")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=[question],
            aggregations=None,
            usage=total_usage,
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        # Get search results
        sources = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
        )
        
        if self.verbose:
            print(f"\033[96mInitial results: {len(sources)} Sources!\033[0m")
        
        # Summarize relevance for each result in parallel
        summary_tasks = []
        passage_ids = []  # Track passage IDs separately
        for i, source in enumerate(sources):
            task = self.summarizer.acall(
                query=question,
                passage=source.content,
                initial_rank=i
            )
            summary_tasks.append(task)
            passage_ids.append(i + 1)  # Store the ID
        
        # Wait for all summaries to complete
        summary_preds = await asyncio.gather(*summary_tasks)
        
        # Process summaries and aggregate usage
        summaries = []
        total_usage = {}
        
        for i, summary_pred in enumerate(summary_preds):
            summaries.append({
                "passage_id": passage_ids[i],  # Use stored ID
                "initial_rank": i,
                "relevance_summary": summary_pred.relevance_summary,
            })
            
            # Aggregate usage from each summarization call
            summary_usage = summary_pred.get_lm_usage()
            if summary_usage:
                total_usage = aggregate_usage(total_usage, summary_usage)
        
        if self.verbose:
            print(f"\033[96mGenerated {len(summaries)} relevance summaries in parallel\033[0m")
        
        # Perform reranking based on summaries
        # Convert search results to list of SearchResult objects
        search_results_list = [ObjectFromDB(
            id=result["passage_id"],
            initial_rank=result["initial_rank"],
            content=result["relevance_summary"]
        ) for result in summaries]

        rerank_pred = await self.reranker.acall(
            query=question,
            search_results=search_results_list,
            top_k=self.reranked_k
        )
        
        # Aggregate reranker usage
        reranker_usage = rerank_pred.get_lm_usage()
        if reranker_usage:
            total_usage = aggregate_usage(total_usage, reranker_usage)
        
        # Reorder sources based on reranking
        reranked_sources = []
        for rank_id in rerank_pred.reranked_ids:
            # Find the source corresponding to this rank_id
            for i, source in enumerate(sources):
                if search_results_list[i].id == rank_id:
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
        collection_name="EnronEmails",
        target_property_name="email_body_vector",
        return_property_name="email_body",
        retrieved_k=5,
        reranked_k=5
    )
    test_q = "What are the implications of SBX18?"
    response = test_pipeline.forward(test_q)
    print(response)
    async_response = await test_pipeline.aforward(test_q)
    print(async_response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())