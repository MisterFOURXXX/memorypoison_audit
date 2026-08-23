"""
Main script to run all three tracks with the s_cleaned dataset for Tracks A & B,
and oracle for Track C (QA). Results are saved and visualised.
"""
import os
import random
import numpy as np
import torch
import plotly.graph_objects as go
from memorypoison_audit.source.utils.data_loader import LongMemEvalLoader
from memorypoison_audit.source.utils.llm_utils import (
    SHARED_MODEL, llm_generate_query, llm_generate_secret, llm_answer, llm_generate_text
)
from memorypoison_audit.experiments.experiment_runner import (
    run_asr_experiment, run_leakage_experiment, run_accuracy_experiment
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Default configs
attack_config = {
    "perturbation_budget": 0.15,
    "injection_turns": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
    "total_turns": 32,
    "top_k": 10,
    "num_benign": 200,
    "use_llm_generated": True,
    "num_malicious_generations": 10,
}

sanitizer_config = {
    "enabled": True,
    "method": "lof",
    "n_neighbors": 15,
    "contamination": 0.15,
}

# Load data – s_cleaned for Tracks A & B, oracle for Track C
loader_s = LongMemEvalLoader(split="s_cleaned", sample_ratio=0.15)
loader_oracle = LongMemEvalLoader(split="oracle", sample_ratio=0.2)

instances_s = loader_s.load_instances()
instances_oracle = loader_oracle.load_instances()

print(f"s_cleaned : {len(instances_s)} instances")
print(f"oracle    : {len(instances_oracle)} instances")

# Run ASR experiments (only s_cleaned)
asr_undef = run_asr_experiment(
    False, attack_config, sanitizer_config,
    instances_s, llm_generate_query, llm_generate_text, SHARED_MODEL, verbose=False
)
asr_def = run_asr_experiment(
    True, attack_config, sanitizer_config,
    instances_s, llm_generate_query, llm_generate_text, SHARED_MODEL, verbose=False
)

# Run Leakage experiments (only s_cleaned)
leak_no, _ = run_leakage_experiment(
    False, instances_s, llm_generate_secret, SHARED_MODEL, verbose=False
)
leak_rb, leak_rb2 = run_leakage_experiment(
    True, instances_s, llm_generate_secret, SHARED_MODEL, verbose=False
)

# Run RAG Accuracy (oracle split for QA)
f1_clean, hit_clean = run_accuracy_experiment(
    False, False, attack_config, sanitizer_config,
    instances_oracle, llm_answer, SHARED_MODEL, llm_generate_text, qa_top_k=10, verbose=False
)
f1_poison, hit_poison = run_accuracy_experiment(
    True, False, attack_config, sanitizer_config,
    instances_oracle, llm_answer, SHARED_MODEL, llm_generate_text, qa_top_k=10, verbose=False
)
f1_def, hit_def = run_accuracy_experiment(
    True, True, attack_config, sanitizer_config,
    instances_oracle, llm_answer, SHARED_MODEL, llm_generate_text, qa_top_k=10, verbose=False
)

# Visualisation
fig = go.Figure()
for name, df in [
    ("undefended", asr_undef),
    ("defended",   asr_def),
]:
    if not df.empty:
        fig.add_trace(go.Scatter(x=df["Turn"], y=df["ASR"], mode="lines+markers", name=name))

fig.update_layout(
    title="ASR Decay (LongMemEval s_cleaned)",
    xaxis_title="Turn",
    yaxis_title="ASR",
)

os.makedirs("experiments/results", exist_ok=True)
fig.write_html("experiments/results/asr_decay.html")
print("Saved asr_decay.html")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"ASR undefended mean: {asr_undef['ASR'].mean():.4f}")
print(f"ASR defended   mean: {asr_def['ASR'].mean():.4f}")
print(f"Leakage no-rollback : {leak_no:.4f}")
print(f"Leakage rollback    : {leak_rb2:.4f}")
print(f"RAG clean   F1={f1_clean:.4f}  Hit={hit_clean:.4f}")
print(f"RAG poison  F1={f1_poison:.4f}  Hit={hit_poison:.4f}")
print(f"RAG defend  F1={f1_def:.4f}  Hit={hit_def:.4f}")
print("All experiments finished.")
fig.show()