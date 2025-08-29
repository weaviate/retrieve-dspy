from typing import Optional
import asyncio
import os

import dspy

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import WriteFollowUpQueries

class LoopingQueryWriter(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: Optional[str] = "content",
        max_loops: Optional[int] = 1,
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 20
    ):
        super().__init__(collection_name, target_property_name, search_only=search_only, verbose=verbose, retrieved_k=retrieved_k)
        self.max_loops = max_loops
        self.looping_query_writer = dspy.Predict(WriteFollowUpQueries)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        all_contexts = []
        all_sources = []
        all_searches = [question]
        usage_buckets = []
        
        # Initial search
        sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
        )
        
        all_contexts.extend([s.content for s in sources])
        all_sources.extend(sources)

        if self.verbose:
            print(f"\033[96m Initial search returned {len(sources)} Sources!\033[0m")
                
        loop_count = 0
        while loop_count < self.max_loops:
            contexts_str = "\n".join(all_contexts)
            
            follow_up_result = self.looping_query_writer(
                question=question,
                contexts=contexts_str,
            )

            usage_buckets.append(follow_up_result.get_lm_usage() or {})

            if follow_up_result.follow_up_queries_needed and follow_up_result.follow_up_queries:
                if self.verbose:
                    print(f"\033[94m Loop {loop_count + 1}: Generated {len(follow_up_result.follow_up_queries)} follow-up queries\033[0m")
                
                for follow_up_query in follow_up_result.follow_up_queries:
                    new_sources = weaviate_search_tool(
                        query=follow_up_query,
                        collection_name=self.collection_name,
                        target_property_name=self.target_property_name,
                        retrieved_k=self.retrieved_k,
                    )
                    
                    all_contexts.extend([s.content for s in new_sources])
                    all_sources.extend(new_sources)
                    all_searches.append(follow_up_query)
                    
                    if self.verbose:
                        print(f"\033[92m Follow-up query '{follow_up_query}' returned {len(new_sources)} sources\033[0m")
            else:
                if self.verbose:
                    print(f"\033[93m No follow-up queries needed, stopping at loop {loop_count + 1}\033[0m")
                break
            
            loop_count += 1

        # Remove duplicates while preserving order
        unique_sources = []
        seen_ids = set()
        for source in all_sources:
            if hasattr(source, 'id') and source.id not in seen_ids:
                unique_sources.append(source)
                seen_ids.add(source.id)
            elif not hasattr(source, 'id'):
                unique_sources.append(source)

        if self.verbose:
            print(f"\033[96m Total unique sources after {loop_count + 1} iterations: {len(unique_sources)}\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=unique_sources,
            searches=all_searches,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        all_contexts = []
        all_sources = []
        all_searches = [question]
        usage_buckets = []
        
        # Initial search
        sources = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
        )
        
        all_contexts.extend([s.content for s in sources])
        all_sources.extend(sources)

        if self.verbose:
            print(f"\033[96m Initial search returned {len(sources)} Sources!\033[0m")
                
        loop_count = 0
        while loop_count < self.max_loops:
            contexts_str = "\n".join(all_contexts)
            
            follow_up_result = await self.looping_query_writer.acall(
                question=question,
                contexts=contexts_str,
            )

            usage_buckets.append(follow_up_result.get_lm_usage() or {})

            if follow_up_result.follow_up_queries_needed and follow_up_result.follow_up_queries:
                if self.verbose:
                    print(f"\033[94m Loop {loop_count + 1}: Generated {len(follow_up_result.follow_up_queries)} follow-up queries\033[0m")
                
                for follow_up_query in follow_up_result.follow_up_queries:
                    new_sources = await async_weaviate_search_tool(
                        query=follow_up_query,
                        collection_name=self.collection_name,
                        target_property_name=self.target_property_name,
                        retrieved_k=self.retrieved_k,
                    )
                    
                    all_contexts.extend([s.content for s in new_sources])
                    all_sources.extend(new_sources)
                    all_searches.append(follow_up_query)
                    
                    if self.verbose:
                        print(f"\033[92m Follow-up query '{follow_up_query}' returned {len(new_sources)} sources\033[0m")
            else:
                if self.verbose:
                    print(f"\033[93m No follow-up queries needed, stopping at loop {loop_count + 1}\033[0m")
                break
            
            loop_count += 1

        # Remove duplicates while preserving order
        unique_sources = []
        seen_ids = set()
        for source in all_sources:
            if hasattr(source, 'id') and source.id not in seen_ids:
                unique_sources.append(source)
                seen_ids.add(source.id)
            elif not hasattr(source, 'id'):
                unique_sources.append(source)

        if self.verbose:
            print(f"\033[96m Total unique sources after {loop_count + 1} iterations: {len(unique_sources)}\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=unique_sources,
            searches=all_searches,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )

async def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    lm = dspy.LM("openai/gpt-4.1-mini", api_key=openai_api_key)
    dspy.configure(lm=lm, track_usage=True)
    print(f"DSPy configured with: {lm}")

    test_pipeline = LoopingQueryWriter(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=5,
        max_loops=2,
        verbose=True
    )
    test_q = "How do I integrate Weaviate and Langchain?"
    response = test_pipeline.forward(test_q)
    print(response)
    async_response = await test_pipeline.aforward(test_q)
    print(async_response)

if __name__ == "__main__":
    asyncio.run(main())