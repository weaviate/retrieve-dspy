import asyncio
import os
from typing import Callable

from dspy import Example, Prediction
import weaviate
from weaviate.collections.classes.filters import Filter

def calculate_recall_at_k(
    target_ids: list[str],
    retrieved_ids: list[str],
    k: int,
    verbose: bool = True
):
    """Calculate traditional recall@k for retrieved documents.
    
    Args:
        target_ids: List of target document IDs (ground truth).
        retrieved_ids: List of retrieved document IDs.
        k: The number of top results to consider for recall calculation.
        
    Returns:
        float: Recall@k score (0.0 to 1.0) - proportion of relevant docs
               found in the top k retrieved results.
    """
    if not isinstance(target_ids, list):
        target_ids = [target_ids]
    
    # Use sets for efficient lookup
    target_id_set = {str(id) for id in target_ids}
    retrieved_ids = [str(id) for id in retrieved_ids] if retrieved_ids else []
    
    # Consider only the top k retrieved IDs
    retrieved_ids_at_k = retrieved_ids[:k]
    
    if verbose:
        print(f"\033[96mTarget IDs: {target_id_set}\033[0m")
    
    # Find the number of relevant documents found in the top k
    found_count = sum(1 for retrieved_id in retrieved_ids_at_k if retrieved_id in target_id_set)
    
    if found_count > 0:
        if verbose:
            print(f"\033[92mRetrieved IDs @{k}: {retrieved_ids_at_k}\033[0m")
    else:
        if verbose:
            print(f"\033[91mRetrieved IDs @{k}: {retrieved_ids_at_k}\033[0m")
    
    recall = found_count / len(target_id_set) if target_id_set else 0
    if verbose:
        print(f"\033[96mRecall@{k}: {found_count}/{len(target_id_set)} = {recall:.2f}\033[0m")
    
    return recall

def calculate_coverage(retrieved_ids: list[str], nugget_data: list[dict], k: int = 1000):
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

def create_recall_metric(k: int, verbose: bool = True) -> Callable:
    """
    Create a recall metric function that wraps the existing calculate_recall function.
    
    Args:
        weaviate_client: Weaviate client instance
        dataset_name: Name of the dataset
        weight: Weight to apply to the recall score
        
    Returns:
        Function that calculates recall score for a single example
    """
    
    def recall_metric(example: Example, prediction, trace=None) -> float:
        try:
            # Extract sources from prediction
            retrieved_ids = prediction.sources
            
            # Get target IDs from example
            target_ids = example.dataset_ids
            
            # Use the existing calculate_recall function
            recall_score = calculate_recall_at_k(
                target_ids=target_ids,
                retrieved_ids=retrieved_ids,
                k=k,
                verbose=verbose
            )
            
            return recall_score
            
        except Exception as e:
            print(f"Error calculating recall: {e}")
            return 0.0
            
    return recall_metric

def create_coverage_metric(k: int = 1000) -> Callable:
    """
    Create a coverage metric function that wraps the existing calculate_coverage function.
    
    Args:
        weaviate_client: Weaviate client instance
        dataset_name: Name of the dataset
        k: Number of top documents to consider (default: 100)
        
    Returns:
        Function that calculates coverage score for a single example
    """
    
    def coverage_metric(example: Example, prediction, trace=None) -> float:
        try:
            retrieved_ids = prediction.sources
            
            nugget_data = example.nugget_data if hasattr(example, 'nugget_data') else []
            
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
        return create_recall_metric(**kwargs)
    elif metric_type == "coverage":
        return create_coverage_metric(**kwargs)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

def create_coverage_metric_with_feedback(
    k: int = 1000
) -> Callable:
    """
    Create a GEPA-compatible coverage metric with nuggets covered and uncovered feedback.
    """
    
    def coverage_metric_with_feedback(
        example: Example, 
        prediction, 
        trace=None,
        pred_name=None,
        pred_trace=None
    ) -> Prediction:
        try:
            retrieved_ids = prediction.sources
            
            nugget_data = example.nugget_data if hasattr(example, 'nugget_data') else []
            
            retrieved_ids_str = [str(id) for id in retrieved_ids[:k]] if retrieved_ids else []
            
            covered = []
            uncovered = []
            
            for nugget in nugget_data:
                nugget_relevant_ids = [str(id) for id in nugget.get('relevant_corpus_ids', [])]
                nugget_text = nugget.get('text', '')
                
                if any(doc_id in retrieved_ids_str for doc_id in nugget_relevant_ids):
                    covered.append(nugget_text)
                else:
                    uncovered.append(nugget_text)
            
            coverage_score = len(covered) / len(nugget_data) if nugget_data else 0.0
            
            feedback = f"Nuggets covered: {covered}\nNuggets not covered: {uncovered}"
            
            return Prediction(
                score=coverage_score,
                feedback=feedback
            )
            
        except Exception as e:
            return Prediction(
                score=0.0,
                feedback=str(e)
            )
    
    return coverage_metric_with_feedback