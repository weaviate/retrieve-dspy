import asyncio

import dspy

from retrieve_dspy.tools.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.retrievers.base_rag import BaseRAG

from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import DecomposeQueryWithHint, ExpandQueryWithHint

class DecomposeAndExpandWithHints(BaseRAG):
    def __init__(
        self, 
        collection_name: str,
        target_property_name: str,
        verbose: bool = True,
        search_only: bool = True,
        retrieved_k: int = 10
    ):
        super().__init__(
            collection_name=collection_name,
            target_property_name=target_property_name,
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k
        )
        self.query_decomposer = dspy.Predict(DecomposeQueryWithHint)
        self.query_expander = dspy.Predict(ExpandQueryWithHint)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        initial_search_results, _ = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=5
        )
        decomposed_queries_pred = self.query_decomposer(
            user_question=question,
            initial_search_results=initial_search_results
        )
        decomposed_queries = decomposed_queries_pred.sub_queries

        if self.verbose:
            print(f"\033[92mOriginal question:\n{question}\033[0m")
            print(f"Decomposed query into: {len(decomposed_queries)} sub-queries")
            print(f"\033[95mDecomposed queries:\n{decomposed_queries}\033[0m")

        final_search_results = []
        for decomposed_query in decomposed_queries:
            results_before_expansion, _ = weaviate_search_tool(
                query=decomposed_query,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=5
            )
            expanded_query_pred = self.query_expander(
                question=decomposed_query, 
                initial_search_results=results_before_expansion
            )
            expanded_query = expanded_query_pred.expanded_query
            if self.verbose:
                print(f"\033[93mOriginal (Decomposed) query:\n{decomposed_query}\033[0m")
                print(f"\033[93mExpanded query:\n{expanded_query}\033[0m")
            search_results, sources = weaviate_search_tool(
                query=expanded_query,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k
            )
            final_search_results.extend(sources)

        if self.verbose:
            print(f"Returning {len(final_search_results)} search results")


        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_search_results,
            searches=[question],
            aggregations=None,
            usage={},
        )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        initial_search_results, _ = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=5
        )
        decomposed_queries_pred = self.query_decomposer(
            user_question=question,
            initial_search_results=initial_search_results
        )
        decomposed_queries = decomposed_queries_pred.sub_queries

        if self.verbose:
            print(f"Decomposed query into: {len(decomposed_queries)} sub-queries")

        final_search_results = []
        tasks = []
        for decomposed_query in decomposed_queries:
            async def process_query(query):
                results_before_expansion, _ = await async_weaviate_search_tool(
                    query=query,
                    collection_name=self.collection_name,
                    target_property_name=self.target_property_name,
                    retrieved_k=5
                )
                expanded_query_pred = await self.query_expander.acall(
                    question=query, 
                    initial_search_results=results_before_expansion
                )
                expanded_query = expanded_query_pred.expanded_query
                if self.verbose:
                    print(f"\033[93mOriginal (Decomposed) query:\n{query}\033[0m")
                    print(f"\033[93mExpanded query:\n{expanded_query}\033[0m")
                search_results, sources = await async_weaviate_search_tool(
                    query=expanded_query,
                    collection_name=self.collection_name,
                    target_property_name=self.target_property_name,
                    retrieved_k=self.retrieved_k,
                )
                return sources
            tasks.append(process_query(decomposed_query))
        
        results = await asyncio.gather(*tasks)
        for sources in results:
            final_search_results.extend(sources)

        if self.verbose:
            print(f"Returning {len(final_search_results)} search results")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_search_results,
            searches=[question],
            aggregations=None,
            usage={},
        )

async def main():
    retriever = DecomposeAndExpandWithHints(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        verbose=True,
        search_only=True,
        retrieved_k=10
    )
    print("Testing sync forward")
    results = retriever.forward("How do I use Weaviate with Langchain?")
    print(results)
    print("Testing async forward")
    results = await retriever.aforward("How do I use Weaviate with Langchain?")
    print(results)

if __name__ == "__main__":
    asyncio.run(main())