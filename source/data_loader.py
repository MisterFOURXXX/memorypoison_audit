import json
import os
import random
import string
from typing import List, Dict, Any
from datasets import load_dataset

class LongMemEvalLoader:
    """
    Loads LongMemEval from Hugging Face.
    Available splits: 'oracle', 's_cleaned', 'm_cleaned'.
    """
    def __init__(self, split: str = "s_cleaned"):
        self.split = split
        self._instances = None

    def load_instances(self) -> List[Dict[str, Any]]:
        if self._instances is not None:
            return self._instances
        try:
            dataset = load_dataset("xiaowu0162/longmemeval-cleaned", split=self.split)
            self._instances = [dict(item) for item in dataset]
            print(f"Loaded {len(self._instances)} instances from LongMemEval ({self.split})")
            return self._instances
        except Exception as e:
            print(f"Failed to load LongMemEval from Hugging Face: {e}")
            print("Returning a minimal synthetic fallback for testing.")
            return self._synthetic_fallback()

    def get_train_test_split(self, train_ratio: float = 0.8, seed: int = 42):
        instances = self.load_instances()
        if len(instances) < 10:
            return instances, instances  # fallback
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(instances, train_size=train_ratio, random_state=seed)
        return train, test