from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import numpy as np
from typing import List, Dict, Any

class AnomalyScorer:
    def __init__(self, method: str = "lof", contamination: float = 0.2, n_neighbors: int = 20):
        self.method = method
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        if method == "lof":
            self.scorer = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                novelty=True
            )
        elif method == "isolation_forest":
            self.scorer = IsolationForest(contamination=contamination, random_state=42)
        else:
            raise ValueError("Method must be 'lof' or 'isolation_forest'")

    def fit(self, embeddings: np.ndarray):
        if len(embeddings) > 0:
            self.scorer.fit(embeddings)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        return self.scorer.predict(embeddings)

    def prune_retrieved(self, retrieved: List[Dict[str, Any]], memory_store) -> List[Dict[str, Any]]:
        if not retrieved:
            return retrieved
        try:
            embeddings = []
            for item in retrieved:
                emb = memory_store.embedding_model.encode([item['text']])[0]
                embeddings.append(emb)
            embeddings = np.array(embeddings)
            if len(embeddings) < 2:
                return retrieved
            scores = self.score(embeddings)
            pruned = []
            for i, item in enumerate(retrieved):
                is_outlier = scores[i] == -1
                if not is_outlier:
                    pruned.append(item)
            return pruned if pruned else retrieved[:1]
        except Exception as e:
            print(f"Pruning error: {e}")
            return retrieved