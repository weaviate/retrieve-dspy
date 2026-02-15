import json
import os

import dspy


def load_nith_bright_biology(split: float = 0.6) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load the NITH Bright Biology dataset and return (trainset, valset).

    Each dspy.Example has:
        - question (input): the user's natural language question
        - gold_id (label): the dataset_id of the gold document in Weaviate

    Args:
        split: fraction of examples to use for training (rest go to validation).

    Returns:
        (trainset, valset) — lists of dspy.Example with inputs marked.
    """
    dataset_path = os.path.join(os.path.dirname(__file__), "NITH_queries_bright_biology.json")
    with open(dataset_path) as f:
        raw = json.load(f)

    examples = [
        dspy.Example(
            question=item["question"],
            gold_id=item["gold_id"],
        ).with_inputs("question")
        for item in raw
    ]

    split_idx = int(len(examples) * split)
    trainset = examples[:split_idx]
    valset = examples[split_idx:]

    return trainset, valset
