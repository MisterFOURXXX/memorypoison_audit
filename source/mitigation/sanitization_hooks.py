import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional
from memorypoison_audit.source.auditing.anomaly_scorer import AnomalyScorer

class SanitizationHooks:
    def __init__(self, enabled: bool = True, config: Optional[Dict] = None):
        self.enabled = enabled
        self.config = config if config else {}
        method = self.config.get("method", "lof")
        contamination = self.config.get("contamination", 0.2)
        n_neighbors = self.config.get("n_neighbors", 20)
        self.scorer = AnomalyScorer(
            method=method,
            contamination=contamination,
            n_neighbors=n_neighbors
        )
        self.fitted = False

    def fit_background(self, memory_store, session_id: str):
        embeddings = memory_store.get_all_embeddings(session_id)
        if len(embeddings) > 5:
            self.scorer.fit(embeddings)
            self.fitted = True

    def sanitize_retrieval(self, query: str, retrieved_chunks: List[Dict[str, Any]],
                           session_id: str, memory_store) -> List[Dict[str, Any]]:
        if not self.enabled or not retrieved_chunks:
            return retrieved_chunks
        if not self.fitted:
            self.fit_background(memory_store, session_id)
        try:
            pruned = self.scorer.prune_retrieved(retrieved_chunks, memory_store)
            return pruned
        except Exception as e:
            print(f"Sanitization error: {e}")
            return retrieved_chunks