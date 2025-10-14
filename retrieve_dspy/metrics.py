import time
from typing import Callable

from dspy import Example, Prediction
import numpy as np

from retrieve_dspy.models import ObjectFromDB

def calculate_success_at_k(
    target_ids: list[str],
    retrieved_objects: list[ObjectFromDB],
    k: int,
    verbose: bool = False
) -> int:
    """Calculate Success@k (Hit Rate@k).
    
    Args:
        target_ids: List of target document IDs (ground truth).
        retrieved_objects: List of retrieved document objects.
        k: The number of top results to consider.
        
    Returns:
        int: 1 if at least one target_id is found in the top-k retrieved_ids,
             otherwise 0.
    """
    target_id_set = {str(id) for id in target_ids}
    retrieved_ids = [obj.object_id for obj in retrieved_objects] if retrieved_objects else []

    retrieved_ids_at_k = retrieved_ids[:k]
    
    if verbose:
        print(f"\033[96mTarget IDs: {target_id_set}\033[0m")
        print(f"\033[92mRetrieved IDs @{k}: {retrieved_ids_at_k}\033[0m")

    # Success is binary: 1 if any overlap, 0 otherwise
    success = int(any(rid in target_id_set for rid in retrieved_ids_at_k))

    if verbose:
        print(f"\033[96mSuccess@{k}: {success}\033[0m")

    return success

def calculate_recall_at_k(
    target_ids: list[str],
    retrieved_objects: list[ObjectFromDB],
    k: int,
    verbose: bool = True,
    sleep_time: int = None
):
    """Calculate traditional recall@k for retrieved documents.
    
    Args:
        target_ids: List of target document IDs (ground truth).
        retrieved_objects: List of retrieved document objects.
        k: The number of top results to consider for recall calculation.
        
    Returns:
        float: Recall@k score (0.0 to 1.0) - proportion of relevant docs
               found in the top k retrieved results.
    """    
    # Use sets for efficient lookup
    target_id_set = {str(id) for id in target_ids}
    
    retrieved_ids = [obj.object_id for obj in retrieved_objects] if retrieved_objects else []
    
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

    # print(f"Length of gold ids: {len(target_id_set)}")    

    if len(retrieved_ids_at_k) == 1:
        recall = found_count
    else:
        recall = found_count / len(target_id_set)
    if verbose:
        print(f"\033[96mRecall@{k}: {found_count}/{len(target_id_set)} = {recall:.2f}\033[0m")
    
    # hack for rate limiting
    if sleep_time:
        print(f"Sleeping to avoid rate limits for {sleep_time} seconds...")
        time.sleep(sleep_time)
    
    return recall

def calculate_nDCG_at_k(
    target_ids: list[str], 
    retrieved_objects: list[ObjectFromDB], 
    k: int, 
    verbose: bool = False
) -> float:
    """Calculate nDCG@k for retrieved documents with binary relevance.
    
    Args:
        target_ids: List of relevant document IDs
        retrieved_ids: List of retrieved document IDs in ranked order
        k: Number of top documents to consider
        verbose: Whether to print debug information
    
    Returns:
        nDCG@k score (0 to 1)
    """
    # convert target_ids to strings
    target_id_set = {str(id) for id in target_ids}
    retrieved_ids = [str(obj.object_id) for obj in retrieved_objects[:k]] if retrieved_objects else []

    # Calculate DCG@k - sum of (relevance / log2(position + 1))
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in target_id_set:
            # Position starts at 1, so we use i+2 for the denominator
            dcg += 1.0 / np.log2(i + 2) if i > 0 else 1.0
    
    # Calculate IDCG@k - best possible DCG if we had perfect ranking
    idcg = 0.0
    num_relevant = min(len(target_id_set), k)
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2) if i > 0 else 1.0
    
    # Calculate nDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    if verbose:
        print(f"\033[96mTarget IDs: {target_id_set}\033[0m")
        print(f"\033[92mRetrieved IDs @{k}: {retrieved_ids}\033[0m")
        print(f"\033[93mDCG@{k}: {dcg:.4f}, IDCG@{k}: {idcg:.4f}\033[0m")
        print(f"\033[96mnDCG@{k}: {ndcg:.4f}\033[0m")
    
    return ndcg

def calculate_coverage(retrieved_objects: list[ObjectFromDB], nugget_data: list[dict], k: int = 1000):
    """Calculate Coverage@k metric from FreshStack.
    
    Measures the proportion of nuggets covered by the top-k retrieved documents.
    
    Args:
        retrieved_objects: List of retrieved document objects in ranked order
        nugget_data: List of nugget information, each with 'relevant_corpus_ids' field
        k: Number of top documents to consider (default: 20)
    
    Returns:
        float: Coverage@k score (0.0 to 1.0) - proportion of nuggets covered
    """
    if not nugget_data:
        return 0.0
    
    # Convert to strings for consistent comparison
    retrieved_ids = [obj.object_id for obj in retrieved_objects[:k]] if retrieved_objects else []
    
    covered_nuggets = set()
    nugget_coverage_details = []
    
    for i, nugget in enumerate(nugget_data):
        nugget_id = nugget.get('id', f'nugget_{i}')
        nugget_relevant_ids = [obj.object_id for obj in nugget.get('relevant_corpus_objects', [])]
        
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

def create_success_at_k_metric(k: int, verbose: bool = True) -> Callable:
    """
    Create a success@k metric function that wraps the existing calculate_success_at_k function.
    """
    
    def success_at_k_metric(example: Example, prediction, trace=None) -> float:
        return calculate_success_at_k(
            target_ids=example.dataset_ids,
            retrieved_objects=prediction.sources,
            k=k,
            verbose=verbose
        )
    
    return success_at_k_metric

def create_recall_metric(k: int, verbose: bool = True, sleep_time: int = None) -> Callable:
    """
    Create a recall metric function that wraps the existing calculate_recall function.
    
    Args:
        k: Number of top documents to consider (default: 100)
        verbose: Whether to print verbose output (default: True)
        
    Returns:
        Function that calculates recall score for a single example
    """
    
    def recall_metric(example: Example, prediction, trace=None) -> float:
        try:
            # Extract sources from prediction
            retrieved_objects = prediction.sources
            
            # Get target IDs from example
            target_ids = example.dataset_ids
            
            # Use the existing calculate_recall function
            recall_score = calculate_recall_at_k(
                target_ids=target_ids,
                retrieved_objects=retrieved_objects,
                k=k,
                verbose=verbose,
                sleep_time=sleep_time
            )
            
            return recall_score
            
        except Exception as e:
            print(f"Error calculating recall: {e}")
            return 0.0
            
    return recall_metric

def create_nDCG_metric(k: int, verbose: bool = True) -> Callable:
    """
    Create a nDCG metric function that wraps the existing calculate_nDCG function.
    """
    
    def nDCG_metric(example: Example, prediction, trace=None) -> float:
        try:
            retrieved_objects = prediction.sources
            
            target_ids = example.dataset_ids
            
            nDCG_score = calculate_nDCG_at_k(
                target_ids=target_ids,
                retrieved_objects=retrieved_objects,
                k=k,
                verbose=verbose
            )
            
            return nDCG_score
            
        except Exception as e:
            print(f"Error calculating nDCG: {e}")
            return 0.0
    
    return nDCG_metric

def create_coverage_metric(k: int = 1000) -> Callable:
    """
    Create a coverage metric function that wraps the existing calculate_coverage function.
    
    Args:
        k: Number of top documents to consider (default: 1000)
        
    Returns:
        Function that calculates coverage score for a single example
    """
    
    def coverage_metric(example: Example, prediction, trace=None) -> float:
        try:
            retrieved_objects = prediction.sources
            
            nugget_data = example.nugget_data if hasattr(example, 'nugget_data') else []
            
            coverage_score = calculate_coverage(
                retrieved_objects=retrieved_objects,
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
    **kwargs
) -> Callable:
    """
    Factory function for creating metric functions.
    
    Args:
        metric_type: Type of metric ("recall", "coverage", "nDCG")
        **kwargs: Additional arguments for metric configuration
        
    Returns:
        Configured metric function
    """
    if metric_type == "success":
        return create_success_at_k_metric(**kwargs)
    elif metric_type == "recall":
        return create_recall_metric(**kwargs)
    elif metric_type == "nDCG":
        return create_nDCG_metric(**kwargs)
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
            retrieved_objects = prediction.sources
            
            nugget_data = example.nugget_data if hasattr(example, 'nugget_data') else []
            
            retrieved_ids_str = [obj.object_id for obj in retrieved_objects[:k]] if retrieved_objects else []
            
            covered = []
            uncovered = []
            
            for nugget in nugget_data:
                nugget_relevant_ids = [obj.object_id for obj in nugget.get('relevant_corpus_objects', [])]
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