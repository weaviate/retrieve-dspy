import asyncio
import os
from typing import Optional, Literal

import dspy
import numpy as np
from pyversity import diversify, Strategy
import weaviate

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.retrievers.base_rag import BaseRAG

from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB
from retrieve_dspy.signatures import WriteSearchQueries, VerboseWriteSearchQueries

DiversityStrategy = Literal["mmr", "msd", "dpp", "cover"]

STRATEGY_MAP = {
    "mmr": Strategy.MMR,
    "msd": Strategy.MSD,
    "dpp": Strategy.DPP,
    "cover": Strategy.COVER,
}

class ConcatenatedQuerySearcher(BaseRAG):
    """
    Writes multiple search queries, concatenates them, and performs a single search.
    Optionally applies pyversity diversity ranking algorithms to the results.
    
    Diversity Strategies:
    - mmr: Maximal Marginal Relevance - balances relevance and diversity
    - msd: Max Sum of Distances - maximizes diversity by selecting distant points
    - dpp: Determinantal Point Process - probabilistic diversity sampling
    - cover: Coverage-based selection - ensures coverage of the embedding space
    """
    def __init__(
        self, 
        collection_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient | weaviate.WeaviateAsyncClient] = None,
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = True,
        verbose_signature: Optional[bool] = True,
        search_only: Optional[bool] = True, 
        retrieved_k: Optional[int] = 10,
        number_of_queries: Optional[int] = 5,
        diversity_weight: Optional[float] = 0.0,
        diversity_strategy: Optional[DiversityStrategy] = "mmr",
        final_k: Optional[int] = None,
    ):
        """
        Args:
            collection_name: Name of the Weaviate collection to search
            weaviate_client: Optional Weaviate client (sync or async)
            target_property_name: Property name to search against
            verbose: Whether to print debug information
            verbose_signature: Whether to use verbose DSPy signatures (more detailed prompts)
            search_only: Currently only search_only mode is supported
            retrieved_k: Number of results to retrieve from the search
            number_of_queries: Number of search queries to generate and concatenate
            diversity_weight: Weight for diversity ranking (0.0 = pure relevance, 1.0 = pure diversity)
            diversity_strategy: Pyversity strategy to use ('mmr', 'msd', 'dpp', 'cover')
            final_k: Number of final results after diversity ranking (defaults to retrieved_k)
        """
        super().__init__(
            collection_name=collection_name,
            weaviate_client=weaviate_client,
            target_property_name=target_property_name, 
            search_only=search_only, 
            verbose=verbose,
            verbose_signature=verbose_signature
        )
        self.retrieved_k = retrieved_k
        self.number_of_queries = number_of_queries
        self.diversity_weight = diversity_weight
        self.diversity_strategy = diversity_strategy
        self.final_k = final_k or retrieved_k
        
        # Use verbose or standard query writing signature
        query_signature = VerboseWriteSearchQueries if verbose_signature else WriteSearchQueries
        self.query_writer = dspy.Predict(query_signature)

    def _normalize_query_punctuation(self, queries: list[str]) -> list[str]:
        """Ensure each query ends with a period or question mark."""
        normalized = []
        for query in queries:
            query = query.strip()
            if query and not query.endswith(('.', '?', '!')):
                # Add question mark if it looks like a question, otherwise add period
                if query.lower().startswith(('how', 'what', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'can', 'could', 'would', 'should', 'do', 'does', 'did')):
                    query += '?'
                else:
                    query += '.'
            normalized.append(query)
        return normalized

    def _apply_diversity_ranking(self, sources: list[ObjectFromDB]) -> list[ObjectFromDB]:
        """Apply pyversity diversity ranking to the sources."""
        if self.diversity_weight <= 0 or len(sources) == 0:
            return sources[:self.final_k]
        
        vectors = np.array([source.vector for source in sources])
        scores = np.array([source.relevance_score for source in sources])
        
        strategy = STRATEGY_MAP.get(self.diversity_strategy, Strategy.MMR)
        
        diversified_result = diversify(
            embeddings=vectors,
            scores=scores,
            k=min(self.final_k, len(sources)),
            strategy=strategy,
            diversity=self.diversity_weight,
        )
        
        diversity_ranked_indices = diversified_result.indices
        return [sources[i] for i in diversity_ranked_indices]

    def forward(
        self, 
        question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
    ) -> DSPyAgentRAGResponse:
        if weaviate_client is None:
            if isinstance(self.weaviate_client, weaviate.WeaviateClient):
                weaviate_client = self.weaviate_client
        
        qw_pred = self.query_writer(question=question, number_of_queries=self.number_of_queries)
        queries: list[str] = qw_pred.search_queries
        
        # Normalize punctuation for each query
        queries = self._normalize_query_punctuation(queries)

        if self.verbose:
            print(f"\033[92mOriginal question:\n{question}\033[0m")
            print(f"\033[95mWrote {len(queries)} queries!\033[0m")
            print(f"\033[95mQueries:\n{queries}\033[0m")

        usage_buckets = [qw_pred.get_lm_usage() or {}]

        # Concatenate all queries for a single search
        concatenated_query = " ".join(queries)
        
        if self.verbose:
            print(f"\033[93mConcatenated query:\n{concatenated_query}\033[0m")

        # Retrieve more results if applying diversity ranking
        retrieve_k = self.retrieved_k * 2 if self.diversity_weight > 0 else self.retrieved_k

        sources = weaviate_search_tool(
            query=concatenated_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=retrieve_k,
            weaviate_client=weaviate_client,
            return_vector=self.diversity_weight > 0,
            return_score=self.diversity_weight > 0
        )

        # Apply diversity ranking if enabled
        if self.diversity_weight > 0:
            if self.verbose:
                print(f"\033[94mApplying {self.diversity_strategy.upper()} diversity ranking (weight={self.diversity_weight})\033[0m")
            sources = self._apply_diversity_ranking(sources)

        if self.verbose:
            print(f"\033[96mReturning {len(sources)} Sources!\033[0m")

        if not self.search_only:
            print("`retrieve_dspy` currently only supports `search_only` mode")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )
    
    async def aforward(
        self, 
        question: str,
        weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None,
    ) -> DSPyAgentRAGResponse:
        if weaviate_async_client is None:
            if isinstance(self.weaviate_client, weaviate.WeaviateAsyncClient):
                weaviate_async_client = self.weaviate_client
        
        # Generate queries asynchronously
        qw_pred = await self.query_writer.acall(question=question, number_of_queries=self.number_of_queries)
        queries: list[str] = qw_pred.search_queries
        
        # Normalize punctuation for each query
        queries = self._normalize_query_punctuation(queries)
        
        if self.verbose:
            print(f"\033[92mOriginal question:\n{question}\033[0m")
            print(f"\033[95mWrote {len(queries)} queries!\033[0m")
            print(f"\033[95mQueries:\n{queries}\033[0m")

        usage_buckets = [qw_pred.get_lm_usage() or {}]

        # Concatenate all queries for a single search
        concatenated_query = " ".join(queries)
        
        if self.verbose:
            print(f"\033[93mConcatenated query:\n{concatenated_query}\033[0m")

        # Retrieve more results if applying diversity ranking
        retrieve_k = self.retrieved_k * 2 if self.diversity_weight > 0 else self.retrieved_k

        sources = await async_weaviate_search_tool(
            query=concatenated_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=retrieve_k,
            weaviate_async_client=weaviate_async_client,
            return_vector=self.diversity_weight > 0,
            return_score=self.diversity_weight > 0
        )

        # Apply diversity ranking if enabled
        if self.diversity_weight > 0:
            if self.verbose:
                print(f"\033[94mApplying {self.diversity_strategy.upper()} diversity ranking (weight={self.diversity_weight})\033[0m")
            sources = self._apply_diversity_ranking(sources)

        if self.verbose:
            print(f"\033[96mReturning {len(sources)} Sources!\033[0m")

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
    """Test the ConcatenatedQuerySearcher with different diversity strategies."""
    import dspy
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    lm = dspy.LM("openai/gpt-4.1-mini", api_key=openai_api_key)
    dspy.configure(lm=lm, track_usage=True)
    print(f"DSPy configured with: {lm}")

    test_q = "How do I integrate Weaviate and Langchain?"
    
    # Test without diversity ranking
    print("\n" + "="*60)
    print("Testing WITHOUT diversity ranking")
    print("="*60)
    test_pipeline = ConcatenatedQuerySearcher(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=5,
        diversity_weight=0.0,
    )
    response = test_pipeline.forward(test_q)
    print(f"Retrieved {len(response.sources)} sources")
    
    # Test with MMR diversity ranking
    print("\n" + "="*60)
    print("Testing with MMR diversity ranking")
    print("="*60)
    test_pipeline_mmr = ConcatenatedQuerySearcher(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=10,
        final_k=5,
        diversity_weight=0.5,
        diversity_strategy="mmr",
    )
    response_mmr = test_pipeline_mmr.forward(test_q)
    print(f"Retrieved {len(response_mmr.sources)} sources with MMR")
    
    # Test with MSD diversity ranking
    print("\n" + "="*60)
    print("Testing with MSD diversity ranking")
    print("="*60)
    test_pipeline_msd = ConcatenatedQuerySearcher(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=10,
        final_k=5,
        diversity_weight=0.5,
        diversity_strategy="msd",
    )
    response_msd = test_pipeline_msd.forward(test_q)
    print(f"Retrieved {len(response_msd.sources)} sources with MSD")
    
    # Test async version
    print("\n" + "="*60)
    print("Testing async with DPP diversity ranking")
    print("="*60)
    test_pipeline_dpp = ConcatenatedQuerySearcher(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=10,
        final_k=5,
        diversity_weight=0.5,
        diversity_strategy="dpp",
    )
    async_response = await test_pipeline_dpp.aforward(test_q)
    print(f"Retrieved {len(async_response.sources)} sources with DPP (async)")


if __name__ == "__main__":
    asyncio.run(main())

