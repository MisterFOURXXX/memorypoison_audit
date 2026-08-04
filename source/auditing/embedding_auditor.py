import os
import sys
repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

class EmbeddingAuditor:
    @staticmethod
    def compute_drift(embeddings_window_1: np.ndarray, embeddings_window_2: np.ndarray) -> float:
        if len(embeddings_window_1) == 0 or len(embeddings_window_2) == 0:
            return 0.0
        mean1 = np.mean(embeddings_window_1, axis=0)
        mean2 = np.mean(embeddings_window_2, axis=0)
        return 1 - cosine_similarity([mean1], [mean2])[0][0]

    @staticmethod
    def compute_entropy(embeddings: np.ndarray) -> float:
        if embeddings.size == 0:
            return 0.0
        flat = embeddings.flatten()
        hist, _ = np.histogram(flat, bins=10, density=True)
        hist = hist / (hist.sum() + 1e-10)
        return entropy(hist)

    @staticmethod
    def neighborhood_density(embeddings: np.ndarray, k: int = 5) -> float:
        if len(embeddings) < 2:
            return 0.0
        sim_matrix = cosine_similarity(embeddings)
        total = 0.0
        count = 0
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                total += sim_matrix[i][j]
                count += 1
        return total / count if count > 0 else 0.0