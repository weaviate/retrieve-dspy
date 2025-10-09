import numpy as np
import yaml
import time
from pathlib import Path

import retrieve_dspy
from retrieve_dspy.metrics import create_metric
from retrieve_dspy.datasets.in_memory import in_memory_dataset_loader, prepare_random_subset
from retrieve_dspy.clients import get_weaviate_client, get_voyage_client, get_and_connect_weaviate_async_client, get_voyage_async_client

from retriever_builder import RetrieverBuilder


def load_config(config_path="./benchmark-run/eval-config.yml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_clients():
    """Initialize all required clients."""
    weaviate_client = get_weaviate_client()
    voyage_client = get_voyage_client()
    
    # Only initialize async clients if needed (they're not used by most retrievers)
    weaviate_async_client = None
    voyage_async_client = None
    
    return weaviate_client, weaviate_async_client, voyage_client, voyage_async_client


def create_metrics_dict(metrics_config):
    """Create metrics dictionary from configuration."""
    metrics = {}
    
    for metric_config in metrics_config:
        metric_name = metric_config["name"]
        metrics[metric_name] = create_metric(
            metric_type=metric_config["type"],
            k=metric_config["k"],
            verbose=False  # Keep individual metrics quiet
        )
    
    return metrics


def load_dataset(dataset_config):
    """Load the specified dataset."""
    dataset_name = dataset_config["name"]
    
    if dataset_name == "enron":
        # Handle enron dataset loading if you have a specific loader
        raise NotImplementedError("Enron dataset loading needs to be implemented")
    else:
        # Handle BEIR datasets and others
        _, queries = in_memory_dataset_loader(dataset_name=dataset_name)
        return queries


def print_trial_results(trial, num_trials, primary_score, offline_scores):
    """Print results for a single trial."""
    print(f"\nTrial {trial + 1}/{num_trials} Results:")
    print(f"Primary score: {primary_score:.3f}")
    
    for metric_name, score in offline_scores.items():
        print(f"\033[96m{metric_name}\033[0m: \033[92m{score:.3f}\033[0m")


def print_final_results(scores, offline_scores_across_trials, metrics):
    """Print final aggregated results across all trials."""
    print("\n" + "="*60)
    print("PRIMARY METRIC RESULTS ACROSS TRIALS:")
    print("="*60)
    
    scores = np.array(scores)
    print(f"Individual scores: {[f'{score:.3f}' for score in scores]}")
    print(f"Min score: {scores.min():.3f}")
    print(f"Max score: {scores.max():.3f}") 
    print(f"\033[92mMean score: {scores.mean():.3f}\033[0m")
    print(f"Std dev: {scores.std():.3f}")

    print("\n" + "="*60)
    print("ALL METRICS RESULTS ACROSS TRIALS:")
    print("="*60)

    for metric_name in metrics.keys():
        metric_scores = np.array(offline_scores_across_trials[metric_name])
        print(f"\n\033[96m{metric_name}:\033[0m")
        print(f"  Individual scores: {[f'{score:.3f}' for score in metric_scores]}")
        print(f"  Min score:  {metric_scores.min():.3f}")
        print(f"  Max score:  {metric_scores.max():.3f}")
        print(f"  \033[92mMean score: {metric_scores.mean():.3f}\033[0m")
        print(f"  Std dev:    {metric_scores.std():.3f}")


def main():
    # Load configuration
    config = load_config()
    
    # Setup clients
    weaviate_client, weaviate_async_client, voyage_client, voyage_async_client = setup_clients()
    
    # Initialize retriever builder
    builder = RetrieverBuilder(weaviate_client, weaviate_async_client, voyage_client, voyage_async_client)
    
    # Build retriever from config
    print(f"Building retriever with config: {config['retriever']}")
    rag_pipeline = builder.build_retriever(
        retriever_config=config["retriever"],
        dataset_config=config["dataset"],
        lm_config=config.get("language_models")
    )
    
    # Debug: Check if pipeline was created successfully
    if rag_pipeline is None:
        raise ValueError("Failed to create RAG pipeline - returned None")
    
    print(f"Successfully created {type(rag_pipeline).__name__} pipeline")
    
    # Debug: Check if pipeline has required methods
    if not hasattr(rag_pipeline, '__call__'):
        raise ValueError(f"RAG pipeline {type(rag_pipeline).__name__} is not callable")
    
    # Load dataset
    queries = load_dataset(config["dataset"])
    
    # Create metrics
    metrics = create_metrics_dict(config["metrics"])
    
    # Primary metric (first one in the list, or recall@1 by default)
    primary_metric = create_metric(
        metric_type="recall",
        k=1,
        verbose=True
    )
    
    # Initialize tracking variables
    eval_config = config["evaluation"]
    num_trials = eval_config["num_trials"]
    scores = []
    offline_scores_across_trials = {metric_name: [] for metric_name in metrics.keys()}
    used_qs = None  # TODO: Leave this when introducing fine-tuned retrievers
    
    print(f"Running evaluation with {config['retriever']['type']} retriever")
    print(f"Dataset: {config['dataset']['name']} (Collection: {config['dataset']['collection_name']})")
    print(f"Trials: {num_trials}, Samples per trial: {eval_config['num_samples']}")
    print(f"Retriever config: retrieved_k={config['retriever'].get('retrieved_k', 'N/A')}")
    
    # Run evaluation trials
    for trial in range(num_trials):
        print(f"\nRunning trial {trial + 1}/{num_trials}")
        
        # Prepare test set
        testset = prepare_random_subset(
            queries=queries,
            num_samples=eval_config["num_samples"],
            seed=eval_config["seed"],
            samples_used_in_training=used_qs,
        )
        
        # Create evaluator
        evaluator = retrieve_dspy.utils.get_evaluator(
            testset=testset,
            metric=primary_metric,
        )
        
        # Run evaluation
        dspy_evaluator_kwargs = {
            "num_threads": eval_config["num_threads"]
        }
        
        evaluator_result = evaluator(rag_pipeline, **dspy_evaluator_kwargs)
        primary_score = evaluator_result.score
        scores.append(primary_score)
        all_results = evaluator_result.results
        
        # Calculate offline metrics
        print("Calculating offline metrics...")
        offline_scores = retrieve_dspy.utils.offline_recall_evaluator(
            results=all_results,
            metrics=metrics
        )
        
        # Store results
        for key, value in offline_scores.items():
            offline_scores_across_trials[key].append(value)
        
        # Print trial results
        print_trial_results(trial, num_trials, primary_score, offline_scores)
        
        # Optional sleep to avoid rate limits (uncomment if needed)
        # print("Sleeping to avoid rate limits...")
        # time.sleep(60)
    
    # Print final results
    print_final_results(scores, offline_scores_across_trials, metrics)


if __name__ == "__main__":
    main()