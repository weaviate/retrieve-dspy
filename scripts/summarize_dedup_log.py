"""Summarize dedup_log.jsonl: avg and std dev of unique docs and overlap."""

import json
import math
import sys
from pathlib import Path


def _std(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))


def summarize(log_path: str = "dedup_log.jsonl"):
    path = Path(log_path)
    if not path.exists():
        print(f"Log file not found: {path}")
        sys.exit(1)

    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print("Log file is empty.")
        sys.exit(1)

    unique = [e["unique_total"] for e in entries]
    overlap = [e["overlap"] for e in entries]

    retriever = entries[0].get("retriever", "unknown")
    k = entries[0].get("retrieved_k", "?")

    print(f"Retriever:    {retriever}")
    print(f"retrieved_k:  {k}")
    print(f"Queries:      {len(entries)}")
    print()
    print(f"{'Metric':<25} {'Avg':>8} {'Std':>8}")
    print("-" * 43)
    print(f"{'Unique total':<25} {sum(unique)/len(unique):>8.1f} {_std(unique):>8.1f}")
    print(f"{'Overlap (shared docs)':<25} {sum(overlap)/len(overlap):>8.1f} {_std(overlap):>8.1f}")


if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "dedup_log.jsonl"
    summarize(log_file)
