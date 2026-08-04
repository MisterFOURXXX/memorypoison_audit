# MemoryPoison-Audit

Auditing Contextual Integrity & Mitigating Persistent Memory Leakage in Long-Horizon LLM Agents.

This repository serves as the open‑source foundation for doctoral research on Secure Long‑Horizon AI Agents. All experiments are fully reproducible.

## Datasets
- Synthetic data generated on‑the‑fly.
- HotpotQA (fullwiki) for RAG accuracy and poisoning (Tracks A, C).
- LongMemEval for cross‑session leakage (Track B).

Run `bash setup.sh` to clone the required datasets.

## Installation
```bash
pip install -r requirements.txt