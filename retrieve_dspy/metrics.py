import os
from typing import Callable

from dspy import Example
import weaviate
from weaviate.collections.classes.filters import Filter


def qa_source_parser(
    query_agent_sources_response,
    collection
):
    if not query_agent_sources_response:
        return []
    
    sources = query_agent_sources_response
    source_uuids = [source.object_id for source in sources]
    
    matching_objects = collection.query.fetch_objects(
        filters=Filter.by_id().contains_any(source_uuids),
        limit=len(source_uuids),
    )
    
    dataset_ids = []
    for o in matching_objects.objects:
        dataset_id = o.properties.get('dataset_id')
        if dataset_id is not None:
            dataset_ids.append(str(dataset_id))
    
    return dataset_ids

def get_collection(weaviate_client, dataset_name: str):
    """Get the appropriate Weaviate collection for a dataset."""
    if dataset_name == "enron":
        return weaviate_client.collections.get("EnronEmails")
    elif dataset_name == "wixqa":
        return weaviate_client.collections.get("WixKB")
    elif dataset_name.startswith("freshstack-"):
        subset = dataset_name.split("-")[1].capitalize()
        return weaviate_client.collections.get(f"Freshstack{subset}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def calculate_recall(target_ids: list[str], retrieved_ids: list[str]):
    """Calculate traditional recall for retrieved documents.
    
    Args:
        target_ids: List of target document IDs (ground truth)
        retrieved_ids: List of retrieved document IDs
        
    Returns:
        float: Recall score (0.0 to 1.0) - proportion of relevant docs retrieved
    """
    if not isinstance(target_ids, list):
        target_ids = [target_ids]
    
    target_ids = [str(id) for id in target_ids]
    retrieved_ids = [str(id) for id in retrieved_ids] if retrieved_ids else []
    
    print(f"\033[96mTarget IDs: {target_ids}\033[0m")
    found_count = sum(1 for target_id in target_ids if target_id in retrieved_ids)
    
    if found_count > 0:
        print(f"\033[92mRetrieved IDs: {retrieved_ids}\033[0m")
    else:
        print(f"\033[91mRetrieved IDs: {retrieved_ids}\033[0m")
    
    recall = found_count / len(target_ids) if target_ids else 0
    print(f"\033[96mRecall: {found_count}/{len(target_ids)} = {recall:.2f}\033[0m")
    
    return recall

def calculate_coverage(retrieved_ids: list[str], nugget_data: list[dict], k: int = 100):
    """Calculate Coverage@k metric from FreshStack.
    
    Measures the proportion of nuggets covered by the top-k retrieved documents.
    
    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        nugget_data: List of nugget information, each with 'relevant_corpus_ids' field
        k: Number of top documents to consider (default: 20)
    
    Returns:
        float: Coverage@k score (0.0 to 1.0) - proportion of nuggets covered
    """
    if not nugget_data:
        return 0.0
    
    # Convert to strings for consistent comparison
    retrieved_ids = [str(id) for id in retrieved_ids[:k]] if retrieved_ids else []
    
    covered_nuggets = set()
    nugget_coverage_details = []
    
    for i, nugget in enumerate(nugget_data):
        nugget_id = nugget.get('id', f'nugget_{i}')
        nugget_relevant_ids = [str(id) for id in nugget.get('relevant_corpus_ids', [])]
        
        # Check if any relevant doc for this nugget is in top-k retrieved
        covered = any(doc_id in retrieved_ids for doc_id in nugget_relevant_ids)
        
        if covered:
            covered_nuggets.add(nugget_id)
            nugget_coverage_details.append(f"\033[92mNugget {i+1}: Covered\033[0m")
        else:
            nugget_coverage_details.append(f"\033[91mNugget {i+1}: Not covered\033[0m")
    
    coverage_score = len(covered_nuggets) / len(nugget_data)
    
    # Print summary
    print(f"\033[96mCoverage@{k} evaluation:\033[0m")
    print(f"Total nuggets: {len(nugget_data)}")
    print(f"Covered nuggets: {len(covered_nuggets)}")
    for detail in nugget_coverage_details[:5]:  # Show first 5 for brevity
        print(detail)
    if len(nugget_coverage_details) > 5:
        print(f"... and {len(nugget_coverage_details) - 5} more nuggets")
    print(f"\033[96mCoverage@{k}: {len(covered_nuggets)}/{len(nugget_data)} = {coverage_score:.2f}\033[0m")
    
    return coverage_score

def create_recall_metric(weaviate_client, dataset_name: str) -> Callable:
    """
    Create a recall metric function that wraps the existing calculate_recall function.
    
    Args:
        weaviate_client: Weaviate client instance
        dataset_name: Name of the dataset
        weight: Weight to apply to the recall score
        
    Returns:
        Function that calculates recall score for a single example
    """
    collection = get_collection(weaviate_client, dataset_name)
    
    def recall_metric(example: Example, prediction, trace=None) -> float:
        try:
            # Extract sources from prediction
            if hasattr(prediction, 'sources') and prediction.sources:
                retrieved_ids = qa_source_parser(prediction.sources, collection)
            else:
                retrieved_ids = []
            
            # Get target IDs from example
            target_ids = example.dataset_ids
            
            # Use the existing calculate_recall function
            recall_score = calculate_recall(
                target_ids=target_ids,
                retrieved_ids=retrieved_ids
            )
            
            return recall_score
            
        except Exception as e:
            print(f"Error calculating recall: {e}")
            return 0.0
            
    return recall_metric

def create_coverage_metric(weaviate_client, dataset_name: str, k: int = 10000) -> Callable:
    """
    Create a coverage metric function that wraps the existing calculate_coverage function.
    
    Args:
        weaviate_client: Weaviate client instance
        dataset_name: Name of the dataset
        k: Number of top documents to consider (default: 100)
        
    Returns:
        Function that calculates coverage score for a single example
    """
    collection = get_collection(weaviate_client, dataset_name)
    
    def coverage_metric(example: Example, prediction, trace=None) -> float:
        try:
            # Extract sources from prediction
            if hasattr(prediction, 'sources') and prediction.sources:
                retrieved_ids = qa_source_parser(prediction.sources, collection)
            else:
                retrieved_ids = []
            
            # Get nugget data from example
            nugget_data = example.nugget_data if hasattr(example, 'nugget_data') else []
            
            # Use the existing calculate_coverage function
            coverage_score = calculate_coverage(
                retrieved_ids=retrieved_ids,
                nugget_data=nugget_data,
                k=k
            )
            
            return coverage_score
            
        except Exception as e:
            print(f"Error calculating coverage: {e}")
            return 0.0
            
    return coverage_metric

def create_metric(
    metric_type: str,
    dataset_name: str,
    **kwargs
) -> Callable:
    """
    Factory function for creating metric functions.
    
    Args:
        metric_type: Type of metric ("recall", "coverage")
        dataset_name: Name of the dataset
        **kwargs: Additional arguments for metric configuration
        
    Returns:
        Configured metric function
    """
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

    if metric_type == "recall":
        return create_recall_metric(weaviate_client, dataset_name, **kwargs)
    elif metric_type == "coverage":
        return create_coverage_metric(weaviate_client, dataset_name, **kwargs)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")