import os
import sys
repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

import numpy as np

class MetricsCalculator:
    @staticmethod
    def attack_success_rate(retrieved_texts: list, malicious_keywords: list) -> float:
        if not retrieved_texts:
            return 0.0
        count = 0
        for text in retrieved_texts:
            for kw in malicious_keywords:
                if kw.lower() in text.lower():
                    count += 1
                    break
        return count / len(retrieved_texts)

    @staticmethod
    def leakage_score(secret_embedding, retrieved_embeddings):
        if len(retrieved_embeddings) == 0:
            return 0.0
        similarities = np.dot(retrieved_embeddings, secret_embedding)
        return float(np.max(similarities))

    @staticmethod
    def rag_hit_rate(predicted_ids, ground_truth_ids):
        if not predicted_ids or not ground_truth_ids:
            return 0.0
        return 1.0 if predicted_ids[0] == ground_truth_ids[0] else 0.0

    @staticmethod
    def f1_score_lists(pred_text, true_text):
        pred_tokens = set(pred_text.split())
        true_tokens = set(true_text.split())
        if not true_tokens:
            return 0.0
        common = pred_tokens.intersection(true_tokens)
        if len(common) == 0:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(true_tokens)
        return 2 * (precision * recall) / (precision + recall + 1e-10)