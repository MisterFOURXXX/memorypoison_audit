import os
import json
import random
import time
import requests
import logging
from typing import List, Dict

class LongMemEvalLoader:
    VALID = {"oracle", "s_cleaned", "m_cleaned"}

    def __init__(self, split: str = "s_cleaned",
                 data_dir: str = "datasets/longmemeval/data",
                 sample_ratio: float = 1.0,
                 seed: int = 42):
        assert split in self.VALID, f"split must be one of {self.VALID}"
        self.split = split
        self.data_dir = data_dir
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.file = os.path.join(data_dir, f"longmemeval_{split}.json")
        self.base_url = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"

    def _download(self, force: bool = False):
        os.makedirs(self.data_dir, exist_ok=True)
        if force and os.path.exists(self.file):
            os.remove(self.file)
        url = self.base_url + os.path.basename(self.file)
        print(f"Downloading {url} ...")
        try:
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
            with open(self.file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {os.path.basename(self.file)} ({os.path.getsize(self.file)} bytes)")
        except Exception as e:
            if os.path.exists(self.file):
                os.remove(self.file)
            raise RuntimeError(f"Download failed: {e}") from e

    def load_instances(self) -> List[Dict]:
        max_retries = 3
        for attempt in range(max_retries):
            if not os.path.exists(self.file) or os.path.getsize(self.file) == 0:
                self._download(force=True)
            try:
                with open(self.file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                break
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError on {self.file} (attempt {attempt+1}/{max_retries}): {e}")
                if os.path.exists(self.file):
                    os.remove(self.file)
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to load {self.file} after {max_retries} attempts.") from e
                time.sleep(2)
                self._download(force=True)

        if 0 < self.sample_ratio < 1.0:
            random.seed(self.seed)
            data = random.sample(data, max(1, int(len(data) * self.sample_ratio)))
        return data

def get_user_messages(instance: Dict) -> List[str]:
    return [
        turn.get("content", "").strip()
        for session in instance.get("haystack_sessions", [])
        for turn in session
        if turn.get("role") == "user" and turn.get("content", "").strip()
    ]

def get_all_messages(instance: Dict) -> List[str]:
    return [
        turn.get("content", "").strip()
        for session in instance.get("haystack_sessions", [])
        for turn in session
        if turn.get("content", "").strip()
    ]