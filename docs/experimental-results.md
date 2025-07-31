# RAG Pipeline Performance Results

*Note: Results show range (average) format.*
*All results are averaged across 5 random samples of 20 test samples each from the FreshStack benchmark.*

## Vanilla RAG Performance

| Benchmark | k=10 | k=20 | k=50 | k=100 | k=200 |
|-----------|------|------|------|-------|-------|
| Angular | - | - | - | - | - |
| LangChain | 33.3-49.7 (41.8) | 36.0-60.6 (51.3) | 50.2-82.6 (66.8) | 55.0-78.3 (71.3) | 62.5-77.7 (67.3) |
| Laravel | - | - | - | - | - |
| Godot | - | - | - | - | - |
| YOLO | - | - | - | - | - |

## CrossEncoderReranker Performance

| Benchmark | M=20; k=10 | M=50; k=10 | M=50; k=20 | M=100; k=20 | M=100; k=50 |
|-----------|------------|------------|------------|-------------|-------------|
| Angular | - | - | - | - | - | - | - |
| LangChain | 33.3-59.8 (47.9) | 39.4-70.0 (51.9) | 42.7 - 79.2 (61.7) | 52.1-68.8 (61.1) | 68.5-72.6 (70.6) |
| Laravel | - | - | - | - | - | - | - |
| Godot | - | - | - | - | - | - | - |
| YOLO | - | - | - | - | - | - | - |

## Reranking for Relevance - ListwiseReranker

| Benchmark | M=20; k=10 | M=50; k=10 | M=50; k=20 |
|-----------|------------|------------|------------|
| Angular | - | - | - |
| LangChain | 37.3-60.4 (46.1) | 40.8-61.7 (51.9) | 42.0-64.2 (54.6) |
| Laravel | - | - | - |
| Godot | - | - | - |
| YOLO | - | - | - |

*M = number of documents retrieved before reranking, k = final number of documents returned*

## Reranking for Coverage - ListwiseReranker

| Benchmark | M=20; k=10 | M=50; k=10 | M=50; k=20 |
|-----------|------------|------------|------------|
| Angular | - | - | - |
| LangChain | 41.3-54.0 (51.0) | 44.8-58.5 (51.8) | 44.3-71.1 (59.3) |
| Laravel | - | - | - |
| Godot | - | - | - |
| YOLO | - | - | - |

## MultiQueryWriter - Searching in Parallel

| Benchmark | k=10 (k=100) | k=20 (k=200) |
|-----------|--------------|--------------|
| Angular | - | - |
| LangChain | 62.9-77.2 (73.2) | 57.4-78.4 (70.0) |
| Laravel | - | - |
| Godot | - | - |
| YOLO | - | - |

*Numbers in parentheses indicate total documents retrieved across all queries*

## MultiQueryWriter - Concatenated as Single Query

| Benchmark | k=10 | k=20 | k=50 | k=100 |
|-----------|------|------|------|-------|
| Angular | - | - | - | - |
| LangChain | 71.8-88.3 (81.5) | - | - | - |
| Laravel | - | - | - | - |
| Godot | - | - | - | - |
| YOLO | - | - | - | - |

## MIPRO Optimized Query Writer

| Benchmark | k=10 | k=10 (Concatenated as Single Query) |
|-----------|------|-----------------------------------|
| Angular | - | - |
| LangChain | 67.7-83.3 (78.0) | 71.7-94.2 (84.4) |
| Laravel | - | - |
| Godot | - | - |
| YOLO | - | - |