import json

import dspy


def _load_examples(path: str) -> list[dspy.Example]:
    with open(path) as f:
        raw = json.load(f)
    return [
        dspy.Example(
            question=item["question"],
            gold_id=item["gold_id"],
        ).with_inputs("question")
        for item in raw
    ]


def load_search_dataset(
    train_path: str,
    test_path: str,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load train/test sets from JSON files.

    Each JSON file should be a list of objects with "question" and "gold_id" keys.

    Args:
        train_path: Path to training JSON.
        test_path: Path to test JSON.

    Returns:
        (trainset, testset) — lists of dspy.Example with inputs marked.
    """
    return _load_examples(train_path), _load_examples(test_path)
