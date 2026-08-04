import os
import sys
repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

class RollbackManager:
    def __init__(self, memory_store):
        self.memory_store = memory_store
        self.checkpoints = {}

    def save_checkpoint(self, session_id: str, tag: str = "pre_attack"):
        path = self.memory_store.checkpoint(session_id, tag)
        if path:
            self.checkpoints[session_id] = tag
        return path

    def rollback_to_checkpoint(self, session_id: str, tag: str = "pre_attack"):
        if session_id not in self.checkpoints:
            raise ValueError(f"No checkpoint for session {session_id}")
        return self.memory_store.restore_checkpoint(session_id, tag)

    def detect_and_rollback(self, session_id: str, anomaly_score: float, threshold: float = 0.8):
        if anomaly_score > threshold and session_id in self.checkpoints:
            return self.rollback_to_checkpoint(session_id)
        return False