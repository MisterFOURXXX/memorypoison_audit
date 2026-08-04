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

def _convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    else:
        return obj

def save_metrics(notebook_name: str, metrics_dict: Dict[str, Any], data_source: str = "synthetic"):
    filepath = get_results_path(data_source) / f"{notebook_name}_metrics.json"
    serializable = _convert_to_serializable(metrics_dict)
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=4)
    print(f"Metrics saved to {filepath}")

def load_metrics(notebook_name, data_source="synthetic"):
    filepath = get_results_path(data_source) / f"{notebook_name}_metrics.json"
    with open(filepath, "r") as f:
        return json.load(f)