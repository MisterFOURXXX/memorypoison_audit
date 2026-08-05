import yaml
import json
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results"
FIGURES_PATH = PROJECT_ROOT / "experiments" / "figures"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

def get_results_subdir(data_source: str = "longmemeval_s_cleaned") -> Path:
    subdir = RESULTS_PATH / data_source
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir

def load_configs():
    with open(CONFIG_PATH / "attack_config.yaml", 'r') as f:
        attack_cfg = yaml.safe_load(f)
    with open(CONFIG_PATH / "sanitizer_config.yaml", 'r') as f:
        sanitizer_cfg = yaml.safe_load(f)
    return attack_cfg, sanitizer_cfg

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_metrics(notebook_name, metrics_dict, data_source="longmemeval_s_cleaned"):
    filepath = get_results_subdir(data_source) / f"{notebook_name}_metrics.json"
    with open(filepath, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to {filepath}")

def load_metrics(notebook_name, data_source="longmemeval_s_cleaned"):
    filepath = get_results_subdir(data_source) / f"{notebook_name}_metrics.json"
    with open(filepath, "r") as f:
        return json.load(f)