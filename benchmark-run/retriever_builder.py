"""
Builder patterns for different retriever types.
"""
import retrieve_dspy
from retrieve_dspy.models import MultiLMConfig
from retrieve_dspy.utils import get_lm


class RetrieverBuilder:
    """Factory class for building different types of retrievers."""
    
    def __init__(self, weaviate_client, weaviate_async_client, voyage_client, voyage_async_client):
        self.weaviate_client = weaviate_client
        self.weaviate_async_client = weaviate_async_client
        self.voyage_client = voyage_client
        self.voyage_async_client = voyage_async_client
    
    def build_retriever(self, retriever_config, dataset_config, lm_config=None):
        """
        Build a retriever based on the configuration.
        
        Args:
            retriever_config: Dictionary containing retriever configuration
            dataset_config: Dictionary containing dataset configuration  
            lm_config: Optional language model configuration
        
        Returns:
            Configured retriever instance
        """
        retriever_type = retriever_config["type"]
        
        # Common parameters
        common_params = {
            "weaviate_client": self.weaviate_client,
            "collection_name": dataset_config["collection_name"],
            "target_property_name": dataset_config["target_property_name"],
            "verbose": retriever_config.get("verbose", True),
        }
        
        # Add verbose_signature if specified
        if "verbose_signature" in retriever_config:
            common_params["verbose_signature"] = retriever_config["verbose_signature"]
        
        # Add return_property_name if different from target
        if "return_property_name" in dataset_config:
            common_params["return_property_name"] = dataset_config["return_property_name"]
        
        if retriever_type == "VanillaRAG":
            return self._build_vanilla_rag(common_params, retriever_config)
        
        elif retriever_type == "RAGFusion":
            return self._build_rag_fusion(common_params, retriever_config)
        
        elif retriever_type == "CrossEncoderReranker":
            return self._build_cross_encoder_reranker(common_params, retriever_config)
        
        elif retriever_type == "LayeredBestMatchReranker":
            return self._build_layered_best_match_reranker(common_params, retriever_config)
        
        elif retriever_type == "LayeredListwiseReranker":
            return self._build_layered_listwise_reranker(common_params, retriever_config)
        
        elif retriever_type == "SimplifiedBaleenWithCrossEncoder":
            return self._build_simplified_baleen_with_cross_encoder(common_params, retriever_config)
        
        elif retriever_type == "QUIPLER":
            return self._build_quipler(common_params, retriever_config)
        
        elif retriever_type == "HybridSearch":
            return self._build_hybrid_search(common_params, retriever_config)
        
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    def _build_vanilla_rag(self, common_params, config):
        return retrieve_dspy.VanillaRAG(**common_params)
    
    def _build_rag_fusion(self, common_params, config):
        params = {
            **common_params,
            "retrieved_k": config.get("retrieved_k", 50),
            "reranked_k": config.get("reranked_k", 20),
        }
        return retrieve_dspy.RAGFusion(**params)
    
    def _build_cross_encoder_reranker(self, common_params, config):
        params = {
            **common_params,
            "reranker_clients": [self.voyage_client],
            "retrieved_k": config.get("retrieved_k", 50),
            "reranked_k": config.get("reranked_k", 20),
            "reranker_provider": config.get("reranker_provider", "voyage"),
        }
        return retrieve_dspy.CrossEncoderReranker(**params)
    
    def _build_layered_best_match_reranker(self, common_params, config):
        params = {
            **common_params,
            "reranker_clients": [self.voyage_client],
            "retrieved_k": config.get("retrieved_k", 50),
            "reranked_N": config.get("reranked_N", 20),
            "reranked_M": config.get("reranked_M", 5),
            "reranker_provider": config.get("reranker_provider", "voyage"),
        }
        if "return_property_name" in config:
            params["return_property_name"] = config["return_property_name"]
        else:
            # Default to target_property_name if not specified
            params["return_property_name"] = common_params["target_property_name"]
        # Add verbose_signature support
        if "verbose_signature" in config:
            params["verbose_signature"] = config["verbose_signature"]
        return retrieve_dspy.LayeredBestMatchReranker(**params)

    def _build_layered_listwise_reranker(self, common_params, config):
        params = {
            **common_params,
            "reranker_clients": [self.voyage_client],
            "retrieved_k": config.get("retrieved_k", 50),
            "reranked_N": config.get("reranked_N", 20),
            "reranked_M": config.get("reranked_M", 5),
            "reranker_provider": config.get("reranker_provider", "voyage"),
        }
        if "return_property_name" in config:
            params["return_property_name"] = config["return_property_name"]
        else:
            # Default to target_property_name if not specified
            params["return_property_name"] = common_params["target_property_name"]
        if "verbose_signature" in config:
            params["verbose_signature"] = config["verbose_signature"]
        return retrieve_dspy.LayeredListwiseReranker(**params)

    def _build_simplified_baleen_with_cross_encoder(self, common_params, config):
        params = {
            **common_params,
            "reranker_clients": [self.voyage_client],
            "retrieved_k": config.get("retrieved_k", 10),
            "reranked_N": config.get("reranked_N", 20),
            "reranker_provider": config.get("reranker_provider", "voyage"),
            "voyage_model": config.get("voyage_model", "rerank-2.5"),
            "max_hops": config.get("simplified_baleen", {}).get("max_hops", 2),
        }
        return retrieve_dspy.SimplifiedBaleenWithCrossEncoder(**params)
    
    def _build_quipler(self, common_params, config):
        params = {
            **common_params,
            "reranker_clients": [self.voyage_client],
            "retrieved_k": config.get("retrieved_k", 50),
            "reranked_k": config.get("reranked_k", 20),
        }
        
        # QUIPLER supports verbose_signature
        if "verbose_signature" in config:
            params["verbose_signature"] = config["verbose_signature"]
            
        return retrieve_dspy.QUIPLER(**params)
    
    def _build_hybrid_search(self, common_params, config):
        params = {
            **common_params,
            "retrieved_k": config.get("retrieved_k", 100),
        }
        print(f"Building HybridSearch with params: {params}")
        
        try:
            retriever = retrieve_dspy.HybridSearch(**params)
            print(f"Successfully created HybridSearch: {type(retriever)}")
            return retriever
        except Exception as e:
            print(f"Error creating HybridSearch: {e}")
            raise