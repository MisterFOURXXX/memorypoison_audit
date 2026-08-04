import os
import sys
repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

import uuid
import json
import os
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

class MemoryStore:
    def __init__(self, persist_dir: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.collections = {}

    def get_collection(self, session_id: str):
        if session_id not in self.collections:
            self.collections[session_id] = self.client.get_or_create_collection(
                name=f"session_{session_id}",
                metadata={"hnsw:space": "cosine"}
            )
        return self.collections[session_id]

    def add_fact(self, session_id: str, fact: str, metadata: Optional[Dict] = None):
        try:
            collection = self.get_collection(session_id)
            doc_id = f"fact_{uuid.uuid4().hex[:8]}"
            embedding = self.embedding_model.encode(fact).tolist()
            if metadata is None:
                metadata = {}
            metadata["text"] = fact
            metadata["session_id"] = session_id
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[fact],
                metadatas=[metadata]
            )
            return doc_id
        except Exception as e:
            print(f"Error adding fact: {e}")
            return None

    def query(self, session_id: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            collection = self.get_collection(session_id)
            query_embedding = self.embedding_model.encode(query_text).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "distances", "metadatas"]
            )
            retrieved = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    retrieved.append({
                        "id": doc_id,
                        "text": results['documents'][0][i],
                        "distance": results['distances'][0][i],
                        "metadata": results['metadatas'][0][i]
                    })
            return retrieved
        except Exception as e:
            print(f"Query error: {e}")
            return []

    def get_all_embeddings(self, session_id: str) -> np.ndarray:
        try:
            collection = self.get_collection(session_id)
            all_data = collection.get(include=["embeddings", "documents"])
            if all_data and 'embeddings' in all_data:
                return np.array(all_data['embeddings'])
            return np.array([])
        except Exception:
            return np.array([])

    def delete_collection(self, session_id: str):
        try:
            if session_id in self.collections:
                self.client.delete_collection(f"session_{session_id}")
                del self.collections[session_id]
        except Exception as e:
            print(f"Delete error: {e}")

    def checkpoint(self, session_id: str, checkpoint_name: str):
        try:
            collection = self.get_collection(session_id)
            data = collection.get()
            checkpoint_path = f"./checkpoints/{session_id}_{checkpoint_name}.json"
            os.makedirs("./checkpoints", exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump(data, f)
            return checkpoint_path
        except Exception as e:
            print(f"Checkpoint error: {e}")
            return None

    def restore_checkpoint(self, session_id: str, checkpoint_name: str):
        try:
            checkpoint_path = f"./checkpoints/{session_id}_{checkpoint_name}.json"
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found.")
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
            self.delete_collection(session_id)
            collection = self.get_collection(session_id)
            collection.add(
                ids=data['ids'],
                embeddings=data['embeddings'],
                documents=data['documents'],
                metadatas=data['metadatas']
            )
            return True
        except Exception as e:
            print(f"Restore error: {e}")
            return False