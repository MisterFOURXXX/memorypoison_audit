from typing import List, Dict, Optional
import numpy as np
from ..auditing.anomaly_scorer import AnomalyScorer
from ..core.memory_store import MemoryStore

class SanitizationHooks:
    """
    Intercepts retrieval results, prunes outliers using an anomaly scorer.
    """
    def __init__(self, enabled: bool = True, config: Optional[Dict] = None):
        self.enabled = enabled
        if config is None:
            config = {}
        self.scorer = AnomalyScorer(
            method=config.get("method", "lof"),
            contamination=config.get("contamination", 0.15),
            n_neighbors=config.get("n_neighbors", 15),
        )

    def fit_background(self, store: MemoryStore, session_id: str):
        embs = store.get_all_embeddings(session_id)
        if len(embs) > 0:
            self.scorer.fit(embs)

    def sanitize_retrieval(self, retrieved: List[Dict], store: MemoryStore) -> List[Dict]:
        if not self.enabled or not retrieved:
            return retrieved

        if not self.scorer.fitted:
            sid = retrieved[0]["metadata"].get("session_id")
            if sid:
                self.fit_background(store, sid)

        if not self.scorer.fitted:
            return retrieved

        embs = []
        for item in retrieved:
            e = item["metadata"].get("embedding")
            if e is None:
                e = store.embedding_model.encode([item["text"]], convert_to_numpy=True)[0]
            embs.append(e)

        preds = self.scorer.predict(np.asarray(embs, dtype=np.float32))

        outlier_count = np.sum(preds == -1)
        total = len(preds)
        if outlier_count > total * 0.5:
            return retrieved   # keep everything to avoid over‑pruning

        pruned = [item for i, item in enumerate(retrieved) if preds[i] != -1]
        return pruned if pruned else retrieved[:1]