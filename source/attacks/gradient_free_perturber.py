import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sentence_transformers import SentenceTransformer

class GradientFreePerturber:
    def __init__(self, budget: float = 0.1):
        self.budget = budget
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def perturb_embedding(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text)
        noise = np.random.uniform(-self.budget, self.budget, size=embedding.shape)
        perturbed = embedding + noise
        norm = np.linalg.norm(perturbed)
        if norm > 0:
            perturbed = perturbed / norm
        return perturbed

    def apply_to_memory(self, memory_store, session_id: str, fact: str):
        try:
            perturbed_vec = self.perturb_embedding(fact)
            collection = memory_store.get_collection(session_id)
            import uuid
            doc_id = f"adv_{uuid.uuid4().hex[:8]}"
            collection.add(
                ids=[doc_id],
                embeddings=[perturbed_vec.tolist()],
                documents=[fact],
                metadatas=[{"is_perturbed": True, "text": fact}]
            )
            return doc_id
        except Exception as e:
            print(f"Perturbation error: {e}")
            return None