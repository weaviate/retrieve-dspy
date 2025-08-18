import json
import os
from typing import Iterable, Set, List, Tuple, Callable, Dict

import numpy as np

import dspy
from dspy import Example, Prediction

def get_evaluator(
    testset: list[Example],
    metric: callable
):
    evaluator = dspy.Evaluate(
        devset=testset,
        metric=metric, 
        num_threads=1,
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

# Used for saving training samples and ensuring we are not testing with training samples

def _iter_example_questions(examples: Iterable[Example]) -> Iterable[str]:
    for ex in examples:
        # DSPy Example supports dict-like access
        q = ex.get("question") if hasattr(ex, "get") else ex["question"]
        if q is not None:
            yield q

def save_training_questions(train_examples: List[Example], path: str) -> dict:
    already = load_training_questions(path)
    batch_unique = set(_iter_example_questions(train_examples))

    to_add = [q for q in batch_unique if q not in already]

    if to_add:
        with open(path, "a", encoding="utf-8") as f:
            for q in to_add:
                f.write(json.dumps({"question": q}) + "\n")

    return {
        "path": path,
        "added": len(to_add),
        "total_in_file": len(already) + len(to_add),
    }

def load_training_questions(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()

    questions: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                q = obj.get("question")
                if isinstance(q, str):
                    questions.add(q)
            except json.JSONDecodeError:
                # Fallback for legacy plain-text lines
                questions.add(line)

    return questions