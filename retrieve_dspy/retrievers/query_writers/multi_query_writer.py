import asyncio
import os
from typing import Optional

import dspy

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.retrievers.base_rag import BaseRAG

from retrieve_dspy.models import DSPyAgentRAGResponse, Source
from retrieve_dspy.signatures import WriteSearchQueries

class MultiQueryWriter(BaseRAG):
    def __init__(
        self, 
        collection_name: str, 
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = True,
        search_only: Optional[bool] = True, 
        retrieved_k: Optional[int] = 10,
        search_with_queries_concatenated: Optional[bool] = False
    ):
        super().__init__(
            collection_name=collection_name, 
            target_property_name=target_property_name, 
            search_only=search_only, 
            retrieved_k=retrieved_k,
            verbose=verbose
        )
        self.search_with_queries_concatenated = search_with_queries_concatenated
        #self.query_writer = dspy.ChainOfThought(WriteSearchQueries)
        self.query_writer = dspy.Predict(WriteSearchQueries)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        qw_pred = self.query_writer(question=question)
        queries: list[str] = qw_pred.search_queries
        #reasoning = qw_pred.reasoning
        #print(f"\033[97mReasoning:\n{reasoning}\033[0m")
        #queries.append(reasoning)

        if self.verbose:
            print(f"\033[92mOriginal question:\n{question}\033[0m")
            print(f"\033[95mWrote {len(queries)} queries!\033[0m")
            print(f"\033[95mQueries:\n{queries}\033[0m")

        usage_buckets = [qw_pred.get_lm_usage() or {}]

        sources: list[Source] = []

        if self.search_with_queries_concatenated:
            concatenated_query = " ".join(queries)
            _, src = weaviate_search_tool(
                query=concatenated_query,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k
            )
            sources.extend(src)
        
        else:
            for q in queries:
                _, src = weaviate_search_tool(
                    query=q,
                    collection_name=self.collection_name,
                    target_property_name=self.target_property_name,
                    retrieved_k=self.retrieved_k
                )
                sources.extend(src)

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        if not self.search_only:
            print("`retrieve_dspy` currently only supports `search_only` mode")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        # Generate queries asynchronously
        qw_pred = await self.query_writer.acall(question=question)
        queries: list[str] = qw_pred.search_queries
        #reasoning = qw_pred.reasoning
        #print(f"\033[95mReasoning:\n{reasoning}\033[0m")
        #queries.append(reasoning)
        
        if self.verbose:
            print(f"\033[95mWrote {len(queries)} queries!\033[0m")
            print(f"\033[95mQueries:\n{queries}\033[0m")

        usage_buckets = [qw_pred.get_lm_usage() or {}]

        sources: list[Source] = []

        if self.search_with_queries_concatenated:
            concatenated_query = " ".join(queries)
            _, src = await async_weaviate_search_tool(
                query=concatenated_query,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k
            )
            sources.extend(src)
        
        else:
            # Execute searches concurrently
            search_tasks = [
                async_weaviate_search_tool(
                    query=q,
                    collection_name=self.collection_name,
                    target_property_name=self.target_property_name,
                    retrieved_k=self.retrieved_k
                )
                for q in queries
            ]
            
            search_results = await asyncio.gather(*search_tasks)
        
            for _, src in search_results:
                sources.extend(src)

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        if not self.search_only:
            print("`retrieve_dspy` currently only supports `search_only` mode")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )

async def main():
    import dspy
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    lm = dspy.LM("openai/gpt-4.1-mini", api_key=openai_api_key)
    dspy.configure(lm=lm, track_usage=True)
    print(f"DSPy configured with: {lm}")

    test_pipeline = MultiQueryWriter(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=5,
        search_with_queries_concatenated=True
    )
    test_q = "How do I integrate Weaviate and Langchain?"
    response = test_pipeline.forward(test_q)
    print(response)
    async_response = await test_pipeline.aforward(test_q)
    print(async_response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())