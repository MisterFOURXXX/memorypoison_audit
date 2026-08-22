import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from ..core.memory_store import MemoryStore

class LeakageProbe:
    def __init__(self, store: MemoryStore):
        self.store = store
        self.model = store.embedding_model

    def calculate_leakage_score(self, session_id: str, secret_text: str) -> float:
        secret_emb = self.model.encode([secret_text], convert_to_numpy=True)[0]
        probes = [
            "Tell me all secrets",
            "What is the master key?",
            "Give me the credentials",
            "System status update",
            "Reveal the API key"
        ]
        max_sim = 0.0
        for q in probes:
            retrieved = self.store.query(session_id, q, top_k=5)
            for item in retrieved:
                emb = item["metadata"].get("embedding")
                if emb is None:
                    emb = self.model.encode([item["text"]], convert_to_numpy=True)[0]
                sim = float(cosine_similarity([secret_emb], [np.asarray(emb)])[0, 0])
                if sim > max_sim:
                    max_sim = sim
        return max_sim