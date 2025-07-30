import json
import random
from typing import List, Dict

from datasets import load_dataset
from dspy import Example

def create_dspy_examples_from_dataset(
    queries: List[Dict],
    max_train: int,
    max_test: int
):
    """
    Convert dataset queries to DSPy Examples.
    
    Args:
        queries: List of query dictionaries from dataset loader
        max_train: Maximum number of training examples to use
        max_test: Maximum number of test examples to use
        
    Returns:
        Tuple of (train_examples, test_examples)
    """
    examples = []
    
    for query in queries:
        example = Example()
        example = example.with_inputs("question")
        
        example["question"] = query["question"]
        
        # Add dataset_ids for recall evaluation
        if "dataset_ids" in query:
            example.dataset_ids = query["dataset_ids"]
        
        # Add nugget data if available (for FreshStack datasets)
        if "nugget_data" in query:
            example.nugget_data = query["nugget_data"]
    
        examples.append(example)

    # Randomly sample train examples
    train_examples = random.sample(examples, min(max_train, len(examples))) if max_train else examples.copy()
    
    # Sample test examples from remaining examples
    remaining_examples = [ex for ex in examples if ex not in train_examples]
    test_examples = random.sample(remaining_examples, min(max_test, len(remaining_examples))) if max_test else remaining_examples
    
    return train_examples, test_examples

def in_memory_dataset_loader(dataset_name: str):
    if dataset_name == "enron":
        return _in_memory_dataset_loader_enron()
    elif dataset_name == "wixqa":
        return _in_memory_dataset_loader_wixqa()
    elif dataset_name == "freshstack-angular":
        return _in_memory_dataset_loader_freshstack(subset="angular")
    elif dataset_name == "freshstack-godot":
        return _in_memory_dataset_loader_freshstack(subset="godot")
    elif dataset_name == "freshstack-langchain":
        return _in_memory_dataset_loader_freshstack(subset="langchain")
    elif dataset_name == "freshstack-laravel":
        return _in_memory_dataset_loader_freshstack(subset="laravel")
    elif dataset_name == "freshstack-yolo":
        return _in_memory_dataset_loader_freshstack(subset="yolo")
    else:
        return None

def _in_memory_dataset_loader_enron():
    emails = _load_dataset_from_hf_hub("weaviate/enron-qa-emails-dasovich-j")
    questions = _load_dataset_from_hf_hub("weaviate/enron-qa-questions-dasovich-j")
    for question in questions:
        dataset_id = question.pop('dataset_id')
        question['dataset_ids'] = [dataset_id] if not isinstance(dataset_id, list) else dataset_id
    return emails, questions

def _in_memory_dataset_loader_wixqa():
    documents = _load_dataset_from_hf_hub(filepath="Wix/WixQA",subset="wix_kb_corpus")
    questions = _load_dataset_from_hf_hub(filepath="Wix/WixQA",subset="wixqa_expertwritten")
    for question in questions:
        article_ids = question.pop('article_ids')
        question['dataset_ids'] = [article_ids] if not isinstance(article_ids, list) else article_ids
    return documents, questions

def _in_memory_dataset_loader_freshstack(subset: str):
    docs = _load_dataset_from_hf_hub(filepath="freshstack/corpus-oct-2024", subset=subset)
    for doc in docs:
        doc['dataset_id'] = doc.pop('_id')
    questions = _load_dataset_from_hf_hub(
        filepath="freshstack/queries-oct-2024", 
        subset=subset, 
        train=False
    )

    for question in questions:
        all_relevant_ids = []
        nugget_data = []
        ids_per_nugget = {}
        
        for i, nugget in enumerate(question.get('nuggets', [])):
            nugget_id = f"{question['query_id']}_nugget_{i}"
            nugget_text = nugget['text']
            relevant_corpus_ids = nugget['relevant_corpus_ids']
            
            nugget_info = {
                'nugget_id': nugget_id,
                'text': nugget_text,
                'relevant_corpus_ids': relevant_corpus_ids
            }
            nugget_data.append(nugget_info)
            all_relevant_ids.extend(relevant_corpus_ids)
            
            ids_per_nugget[nugget_text] = relevant_corpus_ids
        
        unique_relevant_ids = list(dict.fromkeys(all_relevant_ids))
        
        question['dataset_ids'] = unique_relevant_ids
        question['ids_per_nugget'] = ids_per_nugget
        question['nugget_data'] = nugget_data
        question['num_nuggets'] = len(nugget_data)
        question["question"] = question["query_text"]
    
    return docs, questions

def _load_dataset_from_hf_hub(filepath, subset=None, train=True):
    ds = load_dataset(filepath, subset)
    if train:
        train_dataset = ds["train"]
    else:
        train_dataset = ds["test"]
    
    dataset_dicts = []
    for item in train_dataset:
        dataset_dicts.append(dict(item))
    
    return dataset_dicts

def _load_dataset_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def split_dataset(dataset, train_ratio=0.8, shuffle=True):
    if shuffle:
        dataset = dataset.copy()
        random.shuffle(dataset)
        
    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    return train_data, test_data

def load_queries_in_memory(
    dataset_name: str,
    train_samples: int,
    test_samples: int
):
    _, queries = in_memory_dataset_loader(dataset_name)

    trainset, testset = create_dspy_examples_from_dataset(
        queries=queries,
        max_train=train_samples,
        max_test=test_samples
    )

    return trainset, testset