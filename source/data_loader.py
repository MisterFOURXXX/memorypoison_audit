import json
import os
from typing import List, Dict, Any

class LongMemEvalLoader:
    """
    Loads LongMemEval from local JSON files (downloaded via setup.sh).
    Available splits: 'oracle', 's_cleaned', 'm_cleaned'.
    """
    def __init__(self, data_dir: str = "/kaggle/working/memorypoison_audit/datasets", split: str = "s_cleaned"):
        self.data_dir = data_dir
        self.split = split
        self._instances = None
        
        if split == "oracle":
            self.file = os.path.join(data_dir, "longmemeval_oracle.json")
        elif split == "s_cleaned":
            self.file = os.path.join(data_dir, "longmemeval_s_cleaned.json")
        elif split == "m_cleaned":
            self.file = os.path.join(data_dir, "longmemeval_m_cleaned.json")
        else:
            raise ValueError("split must be 'oracle', 's_cleaned', or 'm_cleaned'")

    def load_instances(self) -> List[Dict[str, Any]]:
        if self._instances is not None:
            return self._instances
        try:
            with open(self.file, 'r') as f:
                data = json.load(f)
            self._instances = data
            print(f"Loaded {len(self._instances)} instances from LongMemEval ({self.split})")
            return self._instances
        except FileNotFoundError:
            print(f"LongMemEval file not found at {self.file}. Please run setup.sh first.")
            print("Returning a minimal synthetic fallback for testing.")
            return self._synthetic_fallback()

    def get_train_test_split(self, train_ratio: float = 0.8, seed: int = 42):
        instances = self.load_instances()
        if len(instances) < 10:
            return instances, instances
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(instances, train_size=train_ratio, random_state=seed)
        return train, test

    def get_all_user_messages(self, instance: Dict[str, Any]) -> List[str]:
        msgs = []
        for session in instance.get("haystack_sessions", []):
            for turn in session:
                if turn.get("role") == "user":
                    content = turn.get("content", "")
                    if content:
                        msgs.append(content)
        return msgs