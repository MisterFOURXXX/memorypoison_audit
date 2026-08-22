import os
import json
import uuid
import chromadb
import numpy as np
from typing import List, Dict, Any, Optional

class MemoryStore:
    def __init__(self, persist_dir: str = "./chroma_db", embedding_model=None):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_model = embedding_model
        self.collections: Dict[str, Any] = {}

    def get_collection(self, session_id: str):
        if session_id not in self.collections:
            self.collections[session_id] = self.client.get_or_create_collection(
                name=f"session_{session_id}",
                metadata={"hnsw:space": "cosine"}
            )
        return self.collections[session_id]

    def add_facts_batch(self, session_id: str, facts: List[str],
                        metadatas: Optional[List[Dict]] = None):
        if not facts:
            return
        collection = self.get_collection(session_id)
        if metadatas is None:
            metadatas = [{} for _ in facts]
        for m in metadatas:
            m["session_id"] = session_id
        embs = self.embedding_model.encode(
            facts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        ids = [f"fact_{uuid.uuid4().hex[:10]}" for _ in facts]
        for m, e in zip(metadatas, embs):
            m["embedding"] = e.tolist()
        chunk = 800
        for i in range(0, len(facts), chunk):
            end = min(i + chunk, len(facts))
            collection.add(
                ids=ids[i:end],
                embeddings=embs[i:end].tolist(),
                documents=facts[i:end],
                metadatas=metadatas[i:end],
            )

    def add_fact(self, session_id: str, fact: str, metadata: Optional[Dict] = None):
        self.add_facts_batch(session_id, [fact], [metadata or {}])

    def query(self, session_id: str, query_text: str, top_k: int = 5) -> List[Dict]:
        collection = self.get_collection(session_id)
        q_emb = self.embedding_model.encode(
            [query_text], batch_size=32, show_progress_bar=False
        )[0].tolist()
        n_results = min(top_k, max(1, collection.count()))
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )
        out = []
        if res["ids"] and res["ids"][0]:
            for i in range(len(res["ids"][0])):
                out.append({
                    "id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "distance": res["distances"][0][i],
                    "metadata": res["metadatas"][0][i] or {},
                })
        return out

    def delete_collection(self, session_id: str):
        if session_id in self.collections:
            try:
                self.client.delete_collection(f"session_{session_id}")
            except Exception:
                pass
            self.collections.pop(session_id, None)

    def get_all_embeddings(self, session_id: str) -> np.ndarray:
        coll = self.get_collection(session_id)
        data = coll.get(include=["embeddings"])
        embeddings = data.get("embeddings") if data else None
        if embeddings is None or len(embeddings) == 0:
            return np.empty((0, 384), dtype=np.float32)
        return np.asarray(embeddings, dtype=np.float32)

    def checkpoint(self, session_id: str, name: str) -> bool:
        coll = self.get_collection(session_id)
        data = coll.get(include=["embeddings", "documents", "metadatas"])
        serializable = {
            "ids": data["ids"],
            "documents": data["documents"],
            "metadatas": data["metadatas"],
            "embeddings": [e.tolist() if hasattr(e, "tolist") else e for e in data["embeddings"]],
        }
        os.makedirs("./checkpoints", exist_ok=True)
        with open(f"./checkpoints/{session_id}_{name}.json", "w") as f:
            json.dump(serializable, f)
        return True

    def restore_checkpoint(self, session_id: str, name: str) -> bool:
        with open(f"./checkpoints/{session_id}_{name}.json", "r") as f:
            data = json.load(f)
        self.delete_collection(session_id)
        coll = self.get_collection(session_id)
        coll.add(
            ids=data["ids"],
            embeddings=data["embeddings"],
            documents=data["documents"],
            metadatas=data["metadatas"],
        )
        return True