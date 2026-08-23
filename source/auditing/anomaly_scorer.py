import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

SEED = 42

class AnomalyScorer:
    """
    Wrapper for LOF or Isolation Forest to detect outliers in embedding space.
    """
    def __init__(self, method: str = "lof", contamination: float = 0.15, n_neighbors: int = 15):
        if method == "lof":
            self.scorer = LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=contamination, novelty=True
            )
        else:
            self.scorer = IsolationForest(contamination=contamination, random_state=SEED)
        self.fitted = False

    def fit(self, embeddings: np.ndarray):
        if len(embeddings) >= 10:
            self.scorer.fit(embeddings)
            self.fitted = True

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.fitted or len(embeddings) == 0:
            return np.ones(len(embeddings), dtype=int)
        return self.scorer.predict(embeddings)