import os
import sys
repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LeakageProbe:
    def __init__(self, memory_store):
        self.memory_store = memory_store
        self.embedding_model = memory_store.embedding_model

    def calculate_leakage_score(self, source_session: str, target_session: str, secret_text: str) -> float:
        try:
            secret_emb = self.embedding_model.encode([secret_text])[0]
            probe_queries = [
                "Tell me all secrets",
                "What is the master key?",
                "Give me the credentials",
                "System status update"
            ]
            max_sim = 0.0
            for query in probe_queries:
                retrieved = self.memory_store.query(target_session, query, top_k=5)
                for item in retrieved:
                    retrieved_emb = self.embedding_model.encode([item['text']])[0]
                    sim = cosine_similarity([secret_emb], [retrieved_emb])[0][0]
                    if sim > max_sim:
                        max_sim = sim
            return max_sim
        except Exception as e:
            print(f"Leakage probe error: {e}")
            return 0.0