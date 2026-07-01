"""Measure retrieval quality with Precision@K and Recall@K.

This script reads the evaluation dataset produced by build_eval_dataset.py,
issues retrieval requests against the local RAG API, and reports how well the
returned chunk IDs match the expected chunk IDs.
"""

import json
import requests

from eval_config import BASE_URL, PROJECT_ID, CHUNKS_PER_QUESTION, RETRIEVE_K

def load_eval_dataset(path: str) -> list:
    with open(path, 'r') as file:
        return json.load(file)

def compute_precision_at_k(retrieved_ids: list, expected_ids: list) -> float:
    return len(set(retrieved_ids) & set(expected_ids)) / len(retrieved_ids) if retrieved_ids else 0.0

def compute_recall_at_k(retrieved_ids: list, expected_ids: list) -> float:
    return len(set(retrieved_ids) & set(expected_ids)) / len(expected_ids) if expected_ids else 0.0

def get_retrieved_chunk_ids(question: str, project_id: str, k: int) -> list:
    url = f"{BASE_URL}/api/v1/nlp/index/search/{project_id}"
    response = requests.post(url, json={"text": question, "limit": k})
    data = response.json()
    return [result["chunk_id"] for result in data["results"]]

def run_eval(eval_path: str, project_id: str, k: int):
    eval_data = load_eval_dataset(eval_path)
    
    precision_scores = []
    recall_scores = []
    
    for item in eval_data:
        retrieved_ids = get_retrieved_chunk_ids(item["question"], project_id, k)
        expected_ids = item["expected_chunk_ids"]
        
        precision_scores.append(compute_precision_at_k(retrieved_ids, expected_ids))
        recall_scores.append(compute_recall_at_k(retrieved_ids, expected_ids))
    
    print(f"Precision@{k}: {sum(precision_scores) / len(precision_scores):.3f}")
    print(f"Recall@{k}:    {sum(recall_scores) / len(recall_scores):.3f}")

if __name__ == "__main__":
    run_eval("eval_data.json", project_id=PROJECT_ID, k=RETRIEVE_K)