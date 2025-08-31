import asyncio
from typing import Optional, List

import dspy
import weaviate

from retrieve_dspy.retrievers.common.rrf import reciprocal_rank_fusion
from retrieve_dspy.database.weaviate_database import weaviate_search_tool
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB
from retrieve_dspy.signatures import WriteSearchQueries, VerboseWriteSearchQueries

class RAGFusion(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        num_queries: int = 4,  # Number of query variations to generate
        rrf_k: int = 60,  # RRF constant
        verbose: Optional[bool] = False,
        verbose_signature: Optional[bool] = True
    ):
        super().__init__(collection_name, target_property_name, verbose)
        self.num_queries = num_queries
        self.rrf_k = rrf_k
        
        if verbose_signature:
            self.decompose_query = dspy.Predict(VerboseWriteSearchQueries)
        else:
            self.decompose_query = dspy.Predict(WriteSearchQueries)
    
    def forward(self, weaviate_client: weaviate.WeaviateClient, question: str) -> DSPyAgentRAGResponse:
        # Generate query variations
        search_queries_response = self.decompose_query(question=question)
        search_queries = search_queries_response.search_queries[:self.num_queries]
        
        # Add original query if not already included
        if question not in search_queries:
            search_queries = [question] + search_queries[:self.num_queries-1]
        
        if self.verbose:
            print(f"Search queries: {search_queries}")
        
        # Retrieve results for each query
        result_sets: List[List[ObjectFromDB]] = []
        for query in search_queries:
            results = weaviate_search_tool(
                weaviate_client=weaviate_client,
                query=query,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=self.retrieved_k,
                return_score=True  # Enable score retrieval
            )
            
            # Tag results with source query for debugging
            for obj in results:
                obj.source_query = query
            
            result_sets.append(results)
        
        # Apply RRF to combine results
        fused_results = reciprocal_rank_fusion(
            result_sets=result_sets,
            k=self.rrf_k,
            top_k=self.retrieved_k
        )
        
        if self.verbose:
            print(f"Fused {len(fused_results)} unique documents from {sum(len(rs) for rs in result_sets)} total")
        
        # Generate answer using fused results
        context = "\n\n".join([obj.content for obj in fused_results])
        
        # Use your existing answer generation logic here
        # For now, returning structured response
        return DSPyAgentRAGResponse(
            final_answer="",  # You'll need to add answer generation
            sources=fused_results,
            searches=search_queries,
            aggregations=None,
            usage={},
        )