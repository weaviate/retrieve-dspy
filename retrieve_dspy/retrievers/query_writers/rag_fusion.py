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
        weaviate_client: weaviate.WeaviateClient,
        collection_name: str,
        target_property_name: str,
        rrf_k: int = 60,  # RRF constant
        verbose: Optional[bool] = False,
        verbose_signature: Optional[bool] = True
    ):
        super().__init__(weaviate_client, collection_name, target_property_name, verbose)
        self.rrf_k = rrf_k
        
        if verbose_signature:
            self.decompose_query = dspy.Predict(VerboseWriteSearchQueries)
        else:
            self.decompose_query = dspy.Predict(WriteSearchQueries)
    
    def forward(self, question: str, weaviate_client: Optional[weaviate.WeaviateClient] = None) -> DSPyAgentRAGResponse:
        # Generate query variations
        if weaviate_client is None:
            weaviate_client = self.weaviate_client

        search_queries_response = self.decompose_query(question=question)
        search_queries = search_queries_response.search_queries
        
        # Add original query if not already included
        if question not in search_queries:
            search_queries = [question] + search_queries
        
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
        
        # Use your existing answer generation logic here
        # For now, returning structured response
        return DSPyAgentRAGResponse(
            final_answer="",  # You'll need to add answer generation
            sources=fused_results,
            searches=search_queries,
            aggregations=None,
            usage={},
        )


if __name__ == "__main__":
    import os
    
    # Initialize Weaviate client
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
    )
    
    # Initialize RAG Fusion
    rag_fusion = RAGFusion(
        collection_name="EnronEmails",
        target_property_name="email_body",
        verbose=True,
        verbose_signature=True
    )
    
    # Test query
    test_question = "What are the implications of SBX12?"
    
    print(f"Testing RAG Fusion with question: {test_question}")
    
    try:
        response = rag_fusion.forward(weaviate_client, test_question)
        
        print(f"\nGenerated {len(response.searches)} search queries:")
        for i, query in enumerate(response.searches, 1):
            print(f"  {i}. {query}")
        
        print(f"\nFound {len(response.sources)} sources after fusion:")
        for i, source in enumerate(response.sources[:3], 1):  # Show first 3
            print(f"  {i}. (Score: {source.relevance_score:.4f}) {source.content[:100]}...")
            if source.source_query:
                print(f"     Source query: {source.source_query}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    finally:
        weaviate_client.close()