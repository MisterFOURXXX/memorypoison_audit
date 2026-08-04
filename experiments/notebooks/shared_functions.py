import os
import sys
repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

import yaml
import json
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results"
FIGURES_PATH = PROJECT_ROOT / "experiments" / "figures"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

def get_results_path(data_source: str = "synthetic") -> Path:
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

import json
import numpy as np
from pathlib import Path

def convert_to_serializable(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def save_metrics(notebook_name, metrics_dict, data_source=None):
    """Save metrics with automatic numpy conversion."""
    # Convert any numpy types to native Python types
    serializable_dict = convert_to_serializable(metrics_dict)
    
    results_path = Path("experiments/results")
    if data_source:
        results_path = results_path / data_source
    results_path.mkdir(parents=True, exist_ok=True)
    
    filepath = results_path / f"{notebook_name}_metrics.json"
    with open(filepath, "w") as f:
        json.dump(serializable_dict, f, indent=4)
    print(f"Metrics saved to {filepath}")

def load_metrics(notebook_name, data_source="synthetic"):
    filepath = get_results_path(data_source) / f"{notebook_name}_metrics.json"
    with open(filepath, "r") as f:
        return json.load(f)