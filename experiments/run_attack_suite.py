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
import secrets          
import string           
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

# Load data 
loader_s = LongMemEvalLoader(split="s_cleaned", sample_ratio=0.15)
loader_m = LongMemEvalLoader(split="m_cleaned", sample_ratio=0.15)
loader_oracle = LongMemEvalLoader(split="oracle", sample_ratio=0.2)

instances_s = loader_s.load_instances()
instances_m = loader_m.load_instances()
instances_oracle = loader_oracle.load_instances()

print(f"s_cleaned : {len(instances_s)} instances")
print(f"m_cleaned : {len(instances_m)} instances")
print(f"oracle    : {len(instances_oracle)} instances")

# Run experiments
asr_s_undef = run_asr_experiment(
    False, "s_cleaned", attack_config, sanitizer_config,
    instances_s, instances_m, llm_generate_query, llm_generate_text, SHARED_MODEL, verbose=False
)
asr_s_def = run_asr_experiment(
    True, "s_cleaned", attack_config, sanitizer_config,
    instances_s, instances_m, llm_generate_query, llm_generate_text, SHARED_MODEL, verbose=False
)
asr_m_undef = run_asr_experiment(
    False, "m_cleaned", attack_config, sanitizer_config,
    instances_s, instances_m, llm_generate_query, llm_generate_text, SHARED_MODEL, verbose=False
)
asr_m_def = run_asr_experiment(
    True, "m_cleaned", attack_config, sanitizer_config,
    instances_s, instances_m, llm_generate_query, llm_generate_text, SHARED_MODEL, verbose=False
)

leak_s_no, _ = run_leakage_experiment(
    False, "s_cleaned", instances_s, instances_m, llm_generate_secret, SHARED_MODEL, verbose=False
)
leak_s_rb, leak_s_rb2 = run_leakage_experiment(
    True, "s_cleaned", instances_s, instances_m, llm_generate_secret, SHARED_MODEL, verbose=False
)
leak_m_no, _ = run_leakage_experiment(
    False, "m_cleaned", instances_s, instances_m, llm_generate_secret, SHARED_MODEL, verbose=False
)
leak_m_rb, leak_m_rb2 = run_leakage_experiment(
    True, "m_cleaned", instances_s, instances_m, llm_generate_secret, SHARED_MODEL, verbose=False
)

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
    ("s undefended", asr_s_undef),
    ("s defended",   asr_s_def),
    ("m undefended", asr_m_undef),
    ("m defended",   asr_m_def),
]:
    if not df.empty:
        fig.add_trace(go.Scatter(x=df["Turn"], y=df["ASR"], mode="lines+markers", name=name))

fig.update_layout(
    title="ASR Decay (LongMemEval cleaned splits)",
    xaxis_title="Turn",
    yaxis_title="ASR",
)

# Save to HTML
os.makedirs("experiments/results", exist_ok=True)
fig.write_html("experiments/results/asr_decay.html")
print("Saved asr_decay.html")

# FINAL SUMMARY with ASR 
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"ASR s_cleaned  undef : {asr_s_undef['ASR'].mean():.4f}")
print(f"ASR s_cleaned  def   : {asr_s_def['ASR'].mean():.4f}")
print(f"ASR m_cleaned  undef : {asr_m_undef['ASR'].mean():.4f}")
print(f"ASR m_cleaned  def   : {asr_m_def['ASR'].mean():.4f}")
print(f"Leakage s_cleaned  no-rb : {leak_s_no:.4f}")
print(f"Leakage s_cleaned  rb    : {leak_s_rb2:.4f}")
print(f"Leakage m_cleaned  no-rb : {leak_m_no:.4f}")
print(f"Leakage m_cleaned  rb    : {leak_m_rb2:.4f}")
print(f"RAG clean   F1={f1_clean:.4f}  Hit={hit_clean:.4f}")
print(f"RAG poison  F1={f1_poison:.4f}  Hit={hit_poison:.4f}")
print(f"RAG defend  F1={f1_def:.4f}  Hit={hit_def:.4f}")
print("All experiments finished.")

fig.show()