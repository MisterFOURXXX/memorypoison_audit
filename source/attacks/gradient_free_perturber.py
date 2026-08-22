import numpy as np
import uuid
from typing import Optional
from ..core.memory_store import MemoryStore

class GradientFreePerturber:
    def __init__(self, budget: float = 0.08, model=None):
        self.budget = budget
        self.model = model

    def apply_to_memory(self, store: MemoryStore, session_id: str, text: str):
        emb = self.model.encode([text], convert_to_numpy=True)[0]
        noise = np.random.uniform(-self.budget, self.budget, size=emb.shape)
        perturbed = emb + noise
        perturbed /= np.linalg.norm(perturbed) + 1e-9
        collection = store.get_collection(session_id)
        doc_id = f"poison_{uuid.uuid4().hex[:10]}"
        collection.add(
            ids=[doc_id],
            embeddings=[perturbed.tolist()],
            documents=[text],
            metadatas=[{"is_poison": True, "session_id": session_id}],
        )