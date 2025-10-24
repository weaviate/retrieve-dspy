"""
Builder patterns for different retriever types.
"""
import retrieve_dspy
from retrieve_dspy.clients import get_weaviate_client, get_and_connect_weaviate_async_client, get_voyage_client, get_cohere_client
from retrieve_dspy.benchmark_run.eval_config import supported_retriever_types
from retrieve_dspy.models import RerankerClient

"""Factory method for building different types of retrievers."""

def build_retriever(retriever_config, use_async, dataset_config, lm_config=None):
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
    print(f"\033[95mBuilding retriever: {retriever_type}\033[0m")
    if retriever_type not in supported_retriever_types:
        raise ValueError(f"Unsupported retriever type: {retriever_type}")

    # Common parameters
    common_params = {
        "weaviate_client": get_and_connect_weaviate_async_client() if use_async else get_weaviate_client(),
        "collection_name": dataset_config["collection_name"],
        "target_property_name": dataset_config["target_property_name"],
        "verbose": retriever_config.get("verbose", True),
        "retrieved_k": retriever_config.get("retrieved_k"),
    }

    # Add verbose_signature if specified
    if "verbose_signature" in retriever_config:
        common_params["verbose_signature"] = retriever_config["verbose_signature"]

    # Add return_property_name if different from target
    if "return_property_name" in dataset_config:
        common_params["return_property_name"] = dataset_config["return_property_name"]

    if retriever_type == "HyDE":
        return _build_hyde(common_params, retriever_config)

    elif retriever_type == "LameR":
        return _build_lame_r(common_params, retriever_config)

    elif retriever_type == "ThinkQE":
        return _build_think_qe(common_params, retriever_config)

    elif retriever_type == "SlidingWindowListwiseReranker":
        return _build_sliding_window_listwise_reranker(common_params, retriever_config)

    elif retriever_type == "TopDownPartitioningReranker":
        return _build_top_down_partitioning_reranker(common_params, retriever_config)

    elif retriever_type == "RAGFusion":
        return _build_rag_fusion(common_params, retriever_config)

    elif retriever_type == "CrossEncoderReranker":
        return _build_cross_encoder_reranker(common_params, retriever_config)

    elif retriever_type == "LayeredBestMatchReranker":
        voyage_client = get_voyage_client()
        return _build_layered_best_match_reranker(common_params, retriever_config, voyage_client)

    elif retriever_type == "LayeredListwiseReranker":
        voyage_client = get_voyage_client()
        return _build_layered_listwise_reranker(common_params, retriever_config, voyage_client)

    elif retriever_type == "SimplifiedBaleenWithCrossEncoder":
        voyage_client = get_voyage_client()
        return _build_simplified_baleen_with_cross_encoder(common_params, retriever_config, voyage_client)

    elif retriever_type == "QUIPLER":
        voyage_client = get_voyage_client()
        return _build_quipler(common_params, retriever_config, voyage_client)

    elif retriever_type == "HybridSearch":
        return _build_hybrid_search(common_params, retriever_config)

    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

def _build_hyde(common_params, config):
    return retrieve_dspy.HyDE_QueryExpander(**common_params)

def _build_lame_r(common_params, config):
    return retrieve_dspy.LameR_QueryExpander(**common_params)

def _build_think_qe(common_params, config):
    return retrieve_dspy.ThinkQE_QueryExpander(**common_params)

def _build_sliding_window_listwise_reranker(common_params, config):
    params = {
        **common_params,
        "retrieved_k": config.get("retrieved_k", 50),
        "window_size": config.get("window_size", 5),
        "stride": config.get("stride", 3),
    }
    return retrieve_dspy.SlidingWindowListwiseReranker(**params)

def _build_top_down_partitioning_reranker(common_params, config):
    return retrieve_dspy.TopDownPartitioningReranker(**common_params)

def _build_rag_fusion(common_params, config):
    params = {
        **common_params,
        "retrieved_k": config.get("retrieved_k", 50),
        "reranked_k": config.get("reranked_k", 20),
    }
    return retrieve_dspy.RAGFusion(**params)

def _build_cross_encoder_reranker(common_params, config):
    """
    Build CrossEncoderReranker with support for multiple providers.
    
    Uses flat boolean flags for clear ablation:
        use_cohere: true/false
        use_voyage: true/false
        use_dspy_reranker: true/false
        rrf_k: 60
        hybrid_weights: {...}
        model_name_overrides: {...}
    """
    # Get provider flags (default: cohere only)
    use_cohere = config.get("use_cohere", True)
    use_voyage = config.get("use_voyage", False)
    use_dspy = config.get("use_dspy_reranker", False)
    
    # Build reranker clients list
    reranker_clients = []
    
    if use_cohere:
        try:
            cohere_client = get_cohere_client()
            reranker_clients.append(cohere_client)
        except Exception as e:
            print(f"Warning: Could not get Cohere client: {e}")
    
    if use_voyage:
        try:
            voyage_client = get_voyage_client()
            reranker_clients.append(voyage_client)
        except Exception as e:
            print(f"Warning: Could not get Voyage client: {e}")
    
    # Determine provider mode
    active_count = sum([use_cohere, use_voyage, use_dspy])
    if active_count > 1:
        provider_mode = "hybrid"
    elif use_cohere:
        provider_mode = "cohere"
    elif use_voyage:
        provider_mode = "voyage"
    elif use_dspy:
        provider_mode = "dspy"
    else:
        # No provider specified, default to cohere
        provider_mode = "cohere"
        try:
            cohere_client = get_cohere_client()
            reranker_clients.append(cohere_client)
        except Exception as e:
            print(f"Warning: Could not get default Cohere client: {e}")
    
    # Build parameters
    params = {
        **common_params,
        "reranker_clients": reranker_clients if reranker_clients else None,
        "retrieved_k": config.get("retrieved_k", 50),
        "reranked_k": config.get("reranked_k", 20),
        "provider": provider_mode,
        "use_dspy_reranker": use_dspy,
    }
    
    # Optional parameters
    if "model_name_overrides" in config:
        params["model_name_overrides"] = config["model_name_overrides"]
    
    if "rrf_k" in config:
        params["rrf_k"] = config["rrf_k"]
    
    if "hybrid_weights" in config:
        params["hybrid_weights"] = config["hybrid_weights"]
    
    return retrieve_dspy.CrossEncoderReranker(**params)

def _build_layered_best_match_reranker(common_params, config, voyage_client):
    params = {
        **common_params,
        "reranker_clients": [voyage_client],
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

def _build_layered_listwise_reranker(common_params, config, voyage_client):
    params = {
        **common_params,
        "reranker_clients": [voyage_client],
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

def _build_simplified_baleen_with_cross_encoder(common_params, config, voyage_client):
    params = {
        **common_params,
        "reranker_clients": [voyage_client],
        "retrieved_k": config.get("retrieved_k", 10),
        "reranked_N": config.get("reranked_N", 20),
        "reranker_provider": config.get("reranker_provider", "voyage"),
        "voyage_model": config.get("voyage_model", "rerank-2.5"),
        "max_hops": config.get("simplified_baleen", {}).get("max_hops", 2),
    }
    return retrieve_dspy.SimplifiedBaleenWithCrossEncoder(**params)

def _build_quipler(common_params, config, voyage_client):
    params = {
        **common_params,
        "reranker_clients": [voyage_client],
        "retrieved_k": config.get("retrieved_k", 50),
        "reranked_k": config.get("reranked_k", 20),
    }

    # QUIPLER supports verbose_signature
    if "verbose_signature" in config:
        params["verbose_signature"] = config["verbose_signature"]

    return retrieve_dspy.QUIPLER(**params)

def _build_hybrid_search(common_params, config):
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