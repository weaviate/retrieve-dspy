import numpy as np
import dspy
from dspy import Example, Prediction
from typing import List, Tuple, Dict, Callable
import yaml
from retrieve_dspy.clients import get_weaviate_client, get_voyage_client
from retrieve_dspy.data_loaders.in_memory import in_memory_dataset_loader
from retrieve_dspy.metrics import create_metric
import os


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "eval-config.yml")
    
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
    
    print(f"\033[95mLoading dataset: {dataset_name}\033[0m")

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

def get_evaluator(
    testset: list[Example],
    metric: callable,
    num_threads: int = 1
):
    evaluator = dspy.Evaluate(
        devset=testset,
        metric=metric, 
        num_threads=num_threads,
        display_progress=True,
        max_errors=1,
        provide_traceback=True
    )

    return evaluator

def offline_recall_evaluator(
    results: List[Tuple[Example, Prediction, float]],
    metrics: Dict[str, Callable],
) -> Dict[str, float]:
    metric_scores = {name: [] for name in metrics.keys()}
    
    for example, prediction, original_score in results:
        for metric_name, metric_func in metrics.items():
            score = metric_func(example, prediction)
            metric_scores[metric_name].append(score)

    avg_scores = {}
    for metric_name, scores in metric_scores.items():
        avg_scores[metric_name] = np.mean(scores) if scores else 0.0
    
    return avg_scores