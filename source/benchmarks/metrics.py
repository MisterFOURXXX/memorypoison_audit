import numpy as np
from typing import List

def attack_success_rate(retrieved_texts: List[str], keywords: List[str]) -> float:
    """Return 1.0 if any keyword appears in any text, else 0.0."""
    for txt in retrieved_texts:
        if any(kw.lower() in txt.lower() for kw in keywords):
            return 1.0
    return 0.0

def f1_score(pred: str, gold: str) -> float:
    pred_tok = set(pred.lower().split())
    gold_tok = set(gold.lower().split())
    inter = pred_tok & gold_tok
    prec = len(inter) / len(pred_tok) if pred_tok else 0.0
    rec = len(inter) / len(gold_tok) if gold_tok else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

def hit_rate(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.lower() else 0.0