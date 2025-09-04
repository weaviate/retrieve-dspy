from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.signatures import WriteSearchQueries

import dspy
import retrieve_dspy

class QUIPLER(BaseRAG):
    def __init__(
            self, 
            collection_name: str, 
            target_property_name: str, 
            verbose: bool = False, 
            search_only: bool = True, 
            retrieved_k: int = 20):
        super().__init__(collection_name, target_property_name, verbose, search_only, retrieved_k)
        self.query_writer = dspy.Predict(WriteSearchQueries)

        self.searcher = retrieve_dspy.retrievers.rerankers.CrossEncoderReranker(
            collection_name=collection_name,
            target_property_name=target_property_name,
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k
        )

    def forward(self, question: str):
        return self.searcher(question)