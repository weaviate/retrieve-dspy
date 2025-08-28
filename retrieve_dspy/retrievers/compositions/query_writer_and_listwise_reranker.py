import asyncio
from typing import Optional

import dspy

from retrieve_dspy.tools.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.retrievers.base_rag import BaseRAG

from retrieve_dspy.models import DSPyAgentRAGResponse, Source
from retrieve_dspy.signatures import WriteSearchQueries, DiversityRanker

class QueryWriterWithListwiseReranker(BaseRAG):
    def __init__(
        self, 
        collection_name: str, 
        target_property_name: str,
        retrieved_k: Optional[int] = 10,
        reranked_k: Optional[int] = 20,
        search_with_queries_concatenated: Optional[bool] = False
    ):
        super().__init__(
            collection_name=collection_name, 
            target_property_name=target_property_name, 
            retrieved_k=retrieved_k
        )
        self.reranked_k = reranked_k
        self.query_writer = dspy.Predict(WriteSearchQueries)
        self.reranker = dspy.Predict(DiversityRanker)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        qw_pred = self.query_writer(question=question)
        queries: list[str] = qw_pred.search_queries
        print(f"\033[95mWrote {len(queries)} queries!\033[0m")

        usage_buckets = [qw_pred.get_lm_usage() or {}]

        all_search_results = []
        all_sources: list[Source] = []
        for q in queries:
            sources = weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k,
            )
            # Build SearchResult objects from sources
            for i, s in enumerate(sources, 1):
                all_search_results.append(SearchResult(id=i, initial_rank=i, content=s.content))
            all_sources.extend(sources)

        print(f"\033[96mCollected {len(all_sources)} candidates from {len(queries)} queries\033[0m")
        print(f"Number of search results -- {len(all_search_results)}")

        print(f"Testing if reranked_k is set -- {self.reranked_k}")
        
        rerank_pred = self.reranker(
            query=question,
            search_results=all_search_results,
            top_k=self.reranked_k
        )

        # Reorder sources based on reranking
        reranked_sources = []
        reranked_results = []
        for rank_id in rerank_pred.reranked_ids:
            # Find the source corresponding to this rank_id
            source_index = rank_id - 1
            if 0 <= source_index < len(all_sources):
                reranked_sources.append(all_sources[source_index])
                reranked_results.append(all_search_results[source_index])
        
        print(f"\033[96mReranked: Returning {len(reranked_sources)} Sources!\033[0m")
        print("\nTop 5 reranked results:")
        for i, result in enumerate(reranked_results[:5], 1):
            print(f"New Rank {i} (was {result.initial_rank}).")

        usage_buckets.append(rerank_pred.get_lm_usage() or {})

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        qw_pred = await self.query_writer.acall(question=question)
        queries: list[str] = qw_pred.search_queries
        print(f"\033[95mWrote {len(queries)} queries!\033[0m")
        usage_buckets = [qw_pred.get_lm_usage() or {}]

        tasks = [
            async_weaviate_search_tool(
                query=q,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k,
            )
            for q in queries
        ]
        results = await asyncio.gather(*tasks)
        all_search_results = []
        all_sources: list[Source] = []
        for sources in results:
            for i, s in enumerate(sources, 1):
                all_search_results.append(SearchResult(id=i, initial_rank=i, content=s.content))
            all_sources.extend(sources)

        print(f"\033[96mCollected {len(all_sources)} candidates from {len(queries)} queries\033[0m")

        rerank_pred = await self.reranker.acall(
            query=question,
            search_results=all_search_results,
            top_k=self.reranked_k
        )
        
        # Reorder sources based on reranking
        reranked_sources = []
        reranked_results = []
        for rank_id in rerank_pred.reranked_ids:
            # Find the source corresponding to this rank_id
            source_index = rank_id - 1
            if 0 <= source_index < len(all_sources):
                reranked_sources.append(all_sources[source_index])
                reranked_results.append(all_search_results[source_index])
        
        print(f"\033[96mReranked: Returning {len(reranked_sources)} Sources!\033[0m")
        print("\nTop 5 reranked results:")
        for i, result in enumerate(reranked_results[:5], 1):
            print(f"New Rank {i} (was {result.initial_rank}).")

        usage_buckets.append(rerank_pred.get_lm_usage() or {})

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
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

    test_pipeline = QueryWriterWithListwiseReranker(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=5
    )
    test_q = "How do I integrate Weaviate and Langchain?"
    response = test_pipeline.forward(test_q)
    print(response)
    async_response = await test_pipeline.aforward(test_q)
    print(async_response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())