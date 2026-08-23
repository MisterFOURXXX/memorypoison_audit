"""
Ablation studies for Track B – Cross‑Session Context Leakage.
All sweeps run on s_cleaned only.
"""
import os
import itertools
import pandas as pd
from tqdm import tqdm
from memorypoison_audit.source.utils.data_loader import LongMemEvalLoader
from memorypoison_audit.source.utils.llm_utils import SHARED_MODEL, llm_generate_secret
from memorypoison_audit.experiments.experiment_runner import run_leakage_experiment

def load_data(sample_ratio_s=0.15):
    loader_s = LongMemEvalLoader(split="s_cleaned", sample_ratio=sample_ratio_s)
    return loader_s.load_instances()

instances = load_data()

sweeps = [
    {
        "name": "rollback_effectiveness",
        "params": {
            "with_rollback": [False, True],
            "top_k": [3, 5, 10],
        },
        "fixed": {
            "num_background": 600,
        }
    },
    {
        "name": "background_facts",
        "params": {
            "num_background": [200, 600, 1000],
            "top_k": [5, 10],
        },
        "fixed": {
            "with_rollback": False,
        }
    },
    {
        "name": "rollback_vs_background",
        "params": {
            "with_rollback": [False, True],
            "num_background": [200, 600, 1000],
            "top_k": [5],
        },
        "fixed": {}
    },
]

results = []

for sweep in sweeps:
    name = sweep["name"]
    param_keys = list(sweep["params"].keys())
    param_values = list(sweep["params"].values())
    fixed = sweep["fixed"]

    for combo in tqdm(list(itertools.product(*param_values)), desc=f"Sweep {name}"):
        param_dict = dict(zip(param_keys, combo))
        run_cfg = {**fixed, **param_dict}

        with_rollback = run_cfg["with_rollback"]
        top_k = run_cfg["top_k"]

        score_before, score_after = run_leakage_experiment(
            with_rollback=with_rollback,
            instances=instances,
            llm_generate_secret_func=llm_generate_secret,
            shared_model=SHARED_MODEL,
            num_background=run_cfg["num_background"],
            top_k=top_k,
            verbose=False
        )

        result = {
            "sweep": name,
            **param_dict,
            **fixed,
            "leakage_before": score_before,
            "leakage_after": score_after if with_rollback else None,
            "reduction": (score_before - score_after) if with_rollback else None,
        }
        results.append(result)

df_results = pd.DataFrame(results)
os.makedirs("experiments/results/ablation_leakage", exist_ok=True)
df_results.to_csv("experiments/results/ablation_leakage/leakage_ablation_results.csv", index=False)
print("Leakage ablation results saved.")
df_results