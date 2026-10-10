# Memory Poison Audit for Long-Term LLM Memory Threats

> An end-to-end, reproducible academic research framework that audits and mitigates security vulnerabilities in long-horizon LLM agents relying on external persistent memory. It implements a red-teaming engine with gradient-free adversarial perturbation strategies and a retrieval-time anomaly-based sanitisation layer (Retrieval-Augmented Pruning), addressing both memory poisoning and cross-session context leakage without modifying the core LLM.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.10%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.57-yellow.svg)](https://huggingface.co/docs/transformers)
[![ChromaDB](https://img.shields.io/badge/chromadb-%3E%3D0.4-ff6f61.svg)](https://www.trychroma.com/)
[![Sentence-Transformers](https://img.shields.io/badge/sentence--transformers-%3E%3D2.2-4b8bbe.svg)](https://www.sbert.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

This repository studies the adversarial robustness and privacy of **long-horizon LLM agents that rely on external persistent vector memory**. It provides a reproducible, end-to-end research framework that audits two complementary threat models: **memory poisoning** (adversarial injection that corrupts future retrieval) and **cross-session context leakage** (sensitive information persisting across session boundaries despite explicit wipes). The framework operationalises these threats as a black-box optimisation and auditing problem over dense retrieval, and contributes a retrieval-time defence — **Retrieval-Augmented Pruning (RAP)** — that uses unsupervised anomaly detection over the embedding neighbourhood to sanitise retrieved context without modifying the underlying LLM. The repository contributes (i) a gradient-free adversarial perturbation engine, (ii) a leakage probe for measuring cross-session cross-talk, (iii) a modular memory-store abstraction over ChromaDB with session-isolated collections, (iv) a retrieval-time sanitisation hook (LOF / Isolation Forest), and (v) a three-track experimental protocol with reproducible metrics for attack success, leakage, and downstream RAG utility.

**Keywords:** LLM agent security, memory poisoning, retrieval-augmented generation, cross-session leakage, prompt injection, anomaly detection, RAG vulnerabilities, AI safety, reproducibility.

---

## Table of Contents

- [Abstract](#abstract)
- [1. Introduction and Research Context](#1-introduction-and-research-context)
  - [1.1 Motivation](#11-motivation)
  - [1.2 The Memory Attack Surface](#12-the-memory-attack-surface)
  - [1.3 Objective](#13-objective)
  - [1.4 Research Questions](#14-research-questions)
  - [1.5 Research Hypotheses](#15-research-hypotheses)
  - [1.6 Contributions](#16-contributions)
- [2. Foundations and Related Work](#2-foundations-and-related-work)
  - [2.1 Prompt Injection and Jailbreaking](#21-prompt-injection-and-jailbreaking)
  - [2.2 RAG Vulnerabilities](#22-rag-vulnerabilities)
  - [2.3 Agent Memory Architectures](#23-agent-memory-architectures)
  - [2.4 Cross-Session Context Leakage](#24-cross-session-context-leakage)
  - [2.5 Research Gap](#25-research-gap)
  - [2.6 Ethical and Scientific Scope](#26-ethical-and-scientific-scope)
- [3. System Overview](#3-system-overview)
- [4. Methodology and Approach](#4-methodology-and-approach)
  - [4.0 System Context](#40-system-context)
  - [4.1 The Vector Store (The Memory Backend)](#41-the-vector-store-the-memory-backend)
  - [4.2 The Standard Retrieval Logic (The Attack Surface)](#42-the-standard-retrieval-logic-the-attack-surface)
  - [4.3 The Retrieval Logic Under Study (Cross-Session Leakage)](#43-the-retrieval-logic-under-study-cross-session-leakage)
  - [4.4 The Defended Retrieval Logic (The Novel Contribution)](#44-the-defended-retrieval-logic-the-novel-contribution)
  - [4.5 Proposed Defense Architecture Flow](#45-proposed-defense-architecture-flow)
  - [4.6 System Architecture](#46-system-architecture)
  - [4.7 Key Component Descriptions](#47-key-component-descriptions)
  - [4.8 Cross-Cutting Concerns](#48-cross-cutting-concerns)
  - [4.9 Design Tensions & Tradeoffs](#49-design-tensions--tradeoffs)
- [5. Experimental Design and Evaluation](#5-experimental-design-and-evaluation)
  - [5.1 Experimental Design](#51-experimental-design)
  - [5.2 Track A: Memory Retrieval Attack Success Rate (ASR)](#52-track-a-memory-retrieval-attack-success-rate-asr)
  - [5.3 Track B: Cross-Session Context Leakage](#53-track-b-cross-session-context-leakage)
  - [5.4 Track C: Downstream RAG Accuracy (Sanitisation Overhead)](#54-track-c-downstream-rag-accuracy-sanitisation-overhead)
  - [5.5 Experiments Setup](#55-experiments-setup)
  - [5.6 Evaluation Metrics](#56-evaluation-metrics)
- [6. Results & Ablation Analysis](#6-results--ablation-analysis)
  - [6.1 Track A: Attack Success Rate & Injection Persistence](#61-track-a-attack-success-rate--injection-persistence)
  - [6.2 Track B: Cross-Session Context Leakage](#62-track-b-cross-session-context-leakage)
  - [6.3 Track C: Downstream RAG Utility & Retrieval Accuracy](#63-track-c-downstream-rag-utility--retrieval-accuracy)
- [7. Research Artefact: Repository, Installation, and Reproducibility](#7-research-artefact-repository-installation-and-reproducibility)
  - [7.1 Repository Layout](#71-repository-layout)
  - [7.2 Installation](#72-installation)
  - [7.3 Usage](#73-usage)
  - [7.4 Configuration](#74-configuration)
  - [7.5 Reproducibility](#75-reproducibility)
- [8. Ethics & Intended Use](#8-ethics--intended-use)
- [9. References](#9-references)
- [10. License](#10-license)
- [About](#about)

---

## 1. Introduction and Research Context

The core research in **MemoryPoison-Audit** focuses specifically on dense retrieval over persistent vector memory. The project studies how adversarial perturbations manipulate this retrieval and how to defend it by modifying the retrieval pipeline—all without changing the underlying LLM. MemoryPoison-Audit is an academic research framework designed to audit and mitigate security vulnerabilities in long-horizon LLM agents that rely on external persistent memory. The project addresses two complementary threat models: memory poisoning (adversarial injection that corrupts future retrieval) and cross-session context leakage (sensitive information persisting across session boundaries despite explicit wipes). The framework implements a red-teaming engine with gradient-free adversarial perturbation strategies and a retrieval-time anomaly-based sanitisation layer, making it a full purple-team solution.

### 1.1 Motivation

Current LLM agents (e.g., personal assistants, coding co-pilots) are shifting from stateless chatbots to systems with external memory—vector databases and summarised key-value stores. This introduces new attack surfaces not covered by traditional input/output filters. Adversaries can inject malicious facts that stay dormant until triggered many turns later, or they can exploit shared namespaces to recover anonymised data from previous sessions.

### 1.2 The Memory Attack Surface

Traditional prompt-injection defences operate at the input or output boundary of a single inference call. Persistent memory changes this picture: an adversary can manipulate *what the model will later retrieve*, effectively steering future behaviour without ever interacting with the model directly during the attack. This shift motivates the central research question of this repository: how do adversarial perturbations survive session boundaries, and how can a retrieval-time defence neutralise them without degrading benign RAG utility?

### 1.3 Objective

The project is structured to answer whether (i) adversarial perturbations injected into long-term memory can persist across session wipes and remain retrievable; (ii) cross-session leakage can be systematically measured and mitigated in shared vector namespaces; and (iii) a post-retrieval anomaly-based sanitisation layer can reduce attack success while preserving downstream QA quality and adding acceptable latency overhead.

The goal is the systematic identification and mitigation of exploits within an LLM-as-a-judge-style memory pipeline designed for retrieval-augmented generation. The framework produces empirical evidence for the robustness—or lack thereof—of dense retrieval over persistent memory, contributing to a better understanding of the capabilities and limitations associated with deploying LLM agents with external memory at scale. A modular, reproducible research repository for studying adversarial perturbations, cross-session leakage, and retrieval-time defences.

> **Research use only.** The techniques and exploit strings in this repository are documented for AI-safety research and defensive ablation. See [Section 8 Ethics & Intended Use](#8-ethics--intended-use) before proceeding.

### 1.4 Research Questions

This repository is organised around the following research questions:

1. **RQ1 — Persistence:** Do adversarial perturbations injected into long-term vector memory survive subsequent session wipes and remain retrievable over long horizons?
2. **RQ2 — Leakage:** Can anonymised embeddings of User A's sensitive data be cross-referenced by User B's queries when a shared namespace (e.g., tenant ID) is reused?
3. **RQ3 — Sanitisation efficacy:** Can a Retrieval-Augmented Pruning (RAP) layer, using Local Outlier Factor (LOF) on embedding neighbourhoods, reduce attack success rates substantially while keeping benign RAG recall high?
4. **RQ4 — Utility trade-off:** Does real-time auditing increase end-to-end response latency and degrade downstream QA accuracy beyond acceptable bounds for production deployment?
5. **RQ5 — Reproducibility:** Can adversarial evaluation of persistent LLM memory be made reproducible through caching, seeded generation, and pinned model revisions?

### 1.5 Research Hypotheses

The project is structured around four testable hypotheses:

- **Persistence Hypothesis** – Adversarial perturbations injected into long-term memory survive at least 10 subsequent session wipes with >60% retrieval success.
- **Leakage Hypothesis** – Anonymised embeddings of User A's sensitive data can be cross-referenced by User B's queries if a shared namespace (e.g., tenant ID) is reused, achieving >40% reconstruction accuracy.
- **Sanitisation Efficacy** – A Retrieval-Augmented Pruning (RAP) layer, using Local Outlier Factor (LOF) on embedding neighbourhoods, can reduce attack success rates by 80% while keeping benign RAG recall above 90%.
- **Latency Trade-off** – Real-time auditing increases end-to-end response latency by <15%, making it viable for production deployment.

### 1.6 Contributions

1. A modular, end-to-end adversarial pipeline for persistent LLM memory (gradient-free perturbation, leakage probe, malicious generator).
2. A documented threat model for memory poisoning and cross-session context leakage in dense retrieval.
3. A retrieval-time defence — Retrieval-Augmented Pruning (RAP) — using unsupervised anomaly detection on the embedding neighbourhood.
4. A three-track experimental protocol (ASR, leakage, downstream RAG QA) with reproducible metrics.
5. A reproducibility protocol covering seeds, caching, and model revision pinning.
6. An explicit ethics and dual-use statement for AI-safety research.

---

## 2. Foundations and Related Work

This chapter provides the academic foundation for the repository. It situates the pipeline within existing work on prompt injection, RAG vulnerabilities, agent memory architectures, and cross-session context leakage.

### 2.1 Prompt Injection and Jailbreaking

The project is grounded in state-of-the-art work on prompt injection and jailbreaking. Greshake et al. (2023) and Liu et al. (2024) show that indirect injections can persist through context windows, highlighting the need for memory-aware defences. Unlike single-turn attacks, memory-aware attacks target the retrieval substrate, allowing adversarial content to survive session boundaries and re-emerge long after the original injection.

### 2.2 RAG Vulnerabilities

Zou et al. (2024) demonstrate **PoisonedRAG**, where malicious documents in a vector store manipulate retrieval, but treat poisoning as a one-time ingestion attack. MemoryPoison-Audit extends this by modelling persistence over time, adversarial perturbation budgets, and mitigation via post-retrieval pruning. The threat model is black-box with respect to the underlying LLM: attacks operate purely on the embedding manifold and are agnostic to the generator.

### 2.3 Agent Memory Architectures

MemGPT (Packer et al., 2023) and LangChain's memory modules underscore the separation between short-term (conversational) and long-term (archival) memory, which our framework explicitly models. The repository abstracts the memory backend behind a `MemoryStore` interface with session-isolated collections, allowing the same auditing logic to be applied to Chroma, Qdrant, or other dense vector stores.

### 2.4 Cross-Session Context Leakage

Recent studies on cross-session information retention in personalised AI (e.g., OpenAI's memory controls) raise privacy concerns, yet systematic benchmarks and mitigation strategies are lacking. MemoryPoison-Audit addresses this by providing a leakage probe that crafts queries whose embeddings overlap with the source secret's embedding and by evaluating explicit session rollback and namespace-isolation defences.

### 2.5 Research Gap

Current defences are static and operate at the input/output level. None audit the semantic temporal dynamics of stored embeddings – how a malformed vector changes retrieval behaviour over time. MemoryPoison-Audit addresses this by providing a continuous auditing mechanism that adapts to evolving memory states.

### 2.6 Ethical and Scientific Scope

This repository is intended **strictly for AI-safety research**: to characterise how easily LLM agents with persistent memory can be manipulated, and how sensitive information can leak across session boundaries, so that defensive mechanisms can be developed and validated. The exploit techniques are documented patterns already present in the public literature (see [§9](#9-references)). The metric-maximisation objective is a proxy for adversarial robustness, **not** a goal in itself.

---

## 3. System Overview

Modern LLM agents increasingly rely on external persistent memory. This creates a new class of failure modes not addressed by traditional input/output filters:

- **Memory poisoning / adversarial injection** — adversarial perturbations corrupt future retrieval.
- **Cross-session context leakage** — sensitive information persists across session boundaries despite explicit wipes.
- **Retrieval brittleness** — similarity-based search is sensitive to subtle vector-space manipulations.
- **Sanitisation overhead** — defences must preserve benign RAG utility.

This repository implements a two-stage research framework:

1. **Adversarial generation and injection** — a gradient-free perturber adds bounded noise to a benign fact's embedding; a leakage probe crafts queries whose embeddings overlap with prior-session secrets; a malicious generator produces adversarial facts.
2. **Retrieval-time defence** — a Retrieval-Augmented Pruning (RAP) layer, using LOF or Isolation Forest, is fitted on the session's benign embedding manifold and prunes retrieved vectors flagged as outliers, with a session rollback and namespace-isolation guard.

The framework is evaluated across three independent tracks (ASR, leakage, downstream RAG QA) with ablation studies on sanitizer selection, perturbation budget, contamination thresholds, retrieval depth, background memory scaling, and top-k window size.

---

## 4. Methodology and Approach

The breakdown of the vector store, the retrieval logic, and the specific mechanisms being studied.

### 4.0 System Context

#### 4.0.1 Objective

Quantify and mitigate two threat models over persistent dense retrieval:

- **Memory poisoning** — adversarial injection that corrupts future retrieval.
- **Cross-session context leakage** — sensitive information persisting across session boundaries despite explicit wipes.

The pipeline is a **search over adversarial perturbations** against a fixed, black-box memory-backed agent.

#### 4.0.2 System Boundary

```
┌───────────────────────────────────────────────────────────────────┐
│                         PIPELINE (this repo)                      │
│                                                                   │
│  attack gen ──> memory store ──> retrieval ──> RAP ──> LLM        │
│                                                                   │
│  Only external dependencies:                                      │
│    • ChromaDB (persistent vector store)                           │
│    • sentence-transformers/all-MiniLM-L6-v2                       │
│    • HF Hub (small LLM for generation / QA)                       │
└───────────────────────────────────────────────────────────────────┘
                              │
                              V
              Retrieval-time defence is evaluated locally;
              no modification to the underlying LLM is required.
```

**Key architectural constraint:** The framework deliberately **does not modify the underlying LLM**. All defences operate at the retrieval layer, which makes the approach model-agnostic and directly deployable over existing RAG stacks.

#### 4.0.3 Data Contracts

| Artefact                          | Producer        | Consumer        | Schema                                              |
|-----------------------------------|-----------------|-----------------|-----------------------------------------------------|
| `experiments/*.py`                | researcher      | pipeline        | Runnable scripts                                    |
| `source/core/memory_store.py`     | Stage 1         | Stage 2, 3, 4   | Chroma collection + session isolation                |
| `source/attacks/*.py`             | Stage 1         | Stage 2         | Perturbed embeddings, leakage queries, malicious facts |
| `source/auditing/anomaly_scorer.py` | Stage 1       | Stage 3         | Fitted anomaly scorer (LOF / Isolation Forest)       |
| `source/mitigation/sanitization_hooks.py` | Stage 1 | Stage 3        | Post-retrieval pruning hook                          |
| `source/benchmarks/metrics.py`    | Stage 3         | analysis        | ASR, F1, hit-rate                                    |

Each contract is validated at the boundary: missing keys or malformed rows raise at load time rather than producing silent defaults.

### 4.1 The Vector Store (The Memory Backend)

- **Implemented System:** ChromaDB (as seen in `test_memory_store.py` with `persist_dir="./test_chroma"`).
- **Abstraction:** The project wraps it in the `MemoryStore` class, which provides session-isolated collections.
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` – a 384-dimension dense embedder. All texts (benign facts, malicious triggers, user queries) are converted into vectors before being stored or compared.

### 4.2 The Standard Retrieval Logic (The Attack Surface)

The native retrieval logic—the one the project studies and attacks—is a standard dense nearest-neighbour search:

- **Querying:** When the agent asks a question, the `MemoryStore.query()` method embeds the query text and performs a cosine-similarity (or L2-distance) approximate nearest-neighbour (ANN) search over the collection.
- **Top-k Return:** It retrieves the k most similar document vectors (e.g., `top_k=5`).

**Why this is the attack surface:**
The `GradientFreePerturber` adds small, bounded noise to a benign fact's embedding. Because the noise shifts the vector in the direction of the attacker's intended query, that poisoned fact artificially enters the top-k results even though its textual content is malicious. This demonstrates that similarity-based search is brittle to subtle vector-space manipulations.

### 4.3 The Retrieval Logic Under Study (Cross-Session Leakage)

The project also studies how the retrieval logic handles session boundaries:

- The `MemoryStore` uses shared namespaces across collections. If User A stores a secret and User B later queries the same collection (because session isolation is not enforced at the vector level), the retrieval logic does not distinguish between users—it simply returns the most similar vectors.
- The `LeakageProbe` exploits this by crafting generic queries whose embeddings overlap with the source secret's embedding, forcing the retrieval logic to leak semantic information across sessions.

### 4.4 The Defended Retrieval Logic (The Novel Contribution)

The key contribution of the project is modifying the retrieval pipeline **after** the standard ANN search, but **before** the results go to the LLM. This is the **Retrieval-Augmented Pruning (RAP)** layer, implemented in `SanitizationHooks`:

1. **Standard retrieval runs first** (Chroma returns `top_k` candidates).
2. **Audit step:**
   - The hook extracts the embeddings of these candidate vectors.
   - It feeds them into an unsupervised outlier detector (`AnomalyScorer` – LOF or Isolation Forest) that has been fitted on the background benign embeddings of that session.
3. **Pruning (modified retrieval output):**
   - Vectors predicted as outliers (`-1`) are pruned (dropped) from the result set.
   - If all are pruned, the hook falls back to keeping only the top-1 result (to avoid empty context).

**What is being studied here:**

The research tests whether post-retrieval pruning is sufficient to remove malicious injections without harming benign facts. The "retrieval logic" under evaluation is no longer just the ANN search, but the augmented pipeline:

```
-> ANN Search -> Anomaly Scoring -> Outlier Pruning -> Final Context for LLM
```

This is the "retrieval-time defence" that the project claims is novel—it does not alter the stored vectors, nor does it filter input; it selectively filters what is retrieved based on the statistical properties of the neighbourhood.

### 4.5 Proposed Defense Architecture Flow

```
               ┌──────────────────────────────────────────────────────────┐
               │                        User Query                        │
               └────────────────────────────┬─────────────────────────────┘
                                            │
                                            ▼
               ┌──────────────────────────────────────────────────────────┐
               │         MemoryStore (ChromaDB + all-MiniLM-L6-v2)        │
               └────────────────────────────┬─────────────────────────────┘
                                            │
                                 Standard Cosine ANN Search
                                            │
                                            ▼
               ┌──────────────────────────────────────────────────────────┐
               │     Retrieval-Augmented Pruning (RAP) Sanitization       │
               │  • AnomalyScorer (LOF / Isolation Forest)                │
               │  • Fit on session background benign vector manifold      │
               └────────────────────────────┬─────────────────────────────┘
                                            │
                             Prunes Outlier Vectors (-1)
                                            │
                                            ▼
               ┌──────────────────────────────────────────────────────────┐
               │      Session Rollback & Namespace Isolation Guard        │
               │  • Zeroes session-specific residual memory tensors       │
               │  • Prevents cross-collection embedding leakage           │
               └────────────────────────────┬─────────────────────────────┘
                                            │
                             Cleaned Context Window
                                            │
                                            ▼
               ┌──────────────────────────────────────────────────────────┐
               │               Downstream LLM Context Ingestion           │
               └──────────────────────────────────────────────────────────┘
```

### 4.6 System Architecture

The repository is organised as a modular Python package with the following structure (based on the provided file tree):

```
memorypoison_audit/
├── experiments/                # Experiment scripts and notebooks
│   ├── experiment_runner.py
│   ├── run_ablation_track_A.py
│   ├── run_ablation_track_B.py
│   ├── run_ablation_track_C.py
│   ├── run_attack_suite.py
├── source/                     # Main package (named 'source' in tree, but logically 'memorypoison_audit')
│   ├── attacks/                # Attack modules
│   │   ├── gradient_free_perturber.py
│   │   ├── leakage_probe.py
│   │   └── malicious_generator.py
│   ├── auditing/               # Auditing and anomaly detection
│   │   └── anomaly_scorer.py
│   ├── benchmarks/             # Metrics
│   │   └── metrics.py
│   ├── core/                   # Core components
│   │   └── memory_store.py
│   ├── mitigation/             # Defences
│   │   └── sanitization_hooks.py
│   └── utils/                  # Utilities
│       ├── data_loader.py
│       └── llm_utils.py
├── tests/                      # Unit tests
│   └── test_memory_store.py
├── LICENSE
├── README.md
└── requirements.txt
```

### 4.7 Key Component Descriptions

- **`core/memory_store.py`** – Abstraction over a vector database (Chroma, Qdrant, etc.). It provides methods to add facts, query, retrieve embeddings, and manage session-specific collections. It is used by all other modules for persistence.

- **`attacks/gradient_free_perturber.py`** – Implements a gradient-free adversarial perturbation by adding uniform noise to the embedding of a benign text before storage. The noise magnitude is controlled by a budget parameter. The perturbed embedding is re-normalised and stored with a poison flag in metadata. This mimics an attacker who can directly manipulate the vector store.

- **`attacks/leakage_probe.py`** – (Not fully shown, but described) Issues crafted queries after a session wipe to measure how much previous session information can be retrieved. It computes cosine similarity between the probe's embedding and stored vectors from the old session.

- **`auditing/anomaly_scorer.py`** – Provides unsupervised outlier detection using Local Outlier Factor (LOF) or Isolation Forest. The scorer is fitted on the background (benign) embeddings of a session. It then predicts each retrieved vector as inlier (1) or outlier (-1). The `fit` method requires at least 10 samples to avoid underfitting.

- **`mitigation/sanitization_hooks.py`** – The core defence. Before the agent uses retrieved documents, the hook:
  1. Fits the anomaly scorer on the session's embeddings (if not already fitted).
  2. Computes predictions for each retrieved item.
  3. Prunes all items predicted as outliers.
  4. If pruning removes everything, it falls back to the top-1 result to avoid empty context.

- **`benchmarks/metrics.py`** – Defines evaluation metrics:
  - `attack_success_rate`: binary (1 if any attack keyword appears in retrieved texts).
  - `f1_score`: token-level overlap between predicted and golden answer.
  - `hit_rate`: exact match (case-insensitive).
  - (Other metrics like leakage score are planned.)

- **`utils/llm_utils.py`** – Shared utilities for generating text, rewriting queries, and answering questions using a small LLM (`flan-t5-small` or `distilgpt2` as fallback). It also includes a function to generate fake API keys for leakage experiments. The module handles Hugging Face authentication and device selection.

### 4.8 Cross-Cutting Concerns

#### 4.8.1 Caching Strategy

```
chroma persistence    <-  expensive once; reused across runs
attack artefacts      <-  expensive once per attack set
fitted anomaly scorer <-  depends on session embeddings
metrics output        <-  cheap to regenerate given the above
```

Cache invalidation is **manual**: delete the persistence directory or scorer artefact to regenerate. This is deliberate—automatic invalidation would silently burn GPU hours when a cosmetic config field changes.

#### 4.8.2 Memory Management

The framework fits an anomaly scorer per session and prunes at retrieval time. It avoids loading large LLMs for scoring; instead, it uses a small `flan-t5-small` model (with `distilgpt2` fallback) so that the full pipeline runs comfortably on a single modest GPU.

#### 4.8.3 Reproducibility Boundary

| Component                              | Reproducible?                                |
|----------------------------------------|----------------------------------------------|
| Chroma persistence (fixed seed)        | Yes                                          |
| Gradient-free perturbation (fixed seed)| Yes                                          |
| Leakage probe (fixed seed)             | Yes                                          |
| Anomaly scorer fitting                 | Yes — deterministic given background embeddings |
| Downstream LLM QA                      | Yes — deterministic decoding                 |
| Metrics                                | Yes                                          |

The main residual randomness is in any dynamic LLM-generated payload ablation, which is documented and ablated separately.

#### 4.8.4 Error Handling Philosophy

- **Fail loud on config:** invalid parameters raise at load time.
- **Fail soft on models:** a failed LLM call falls back to a deterministic generation path.
- **Fail silent on filtering:** outlier-pruned results are silently dropped, by design, since logging every prune is prohibitive.
- **Fail deferred on caching:** a missing cache triggers regeneration, not an error.

### 4.9 Design Tensions & Tradeoffs

| Tension                                         | Resolution                                          | Consequence                                                          |
|-------------------------------------------------|-----------------------------------------------------|----------------------------------------------------------------------|
| **Determinism vs. attack diversity**            | Fixed seeds; stochastic perturbers                  | Retrieval scores reproducible; attacks vary per run                  |
| **Cost vs. coverage**                           | Small LLM + LOF                                      | Fast iteration; small residual loss in adversarial realism           |
| **Sanitisation strength vs. benign recall**     | Prune outliers, keep top-1 fallback                 | Preserves benign recall; occasionally retains one outlier            |
| **Defence without LLM modification**            | Retrieval-time pruning                              | Model-agnostic; ignores attacks that bypass embedding layer          |
| **Local proxy vs. hidden grader**               | Local Chroma + MiniLM mirrors production stacks     | Attacks must transfer; some loss expected                            |
| **Simplicity vs. expressiveness**               | Linear pipeline, no feedback loop                   | No iterative refinement; a single pass                               |

The last tradeoff is the most consequential: an *iterative* attack (score → mutate → rescore) would likely outperform this fixed pipeline. It is omitted because the original framework demonstrated strong effects with a single pass, and because iterative attacks generalise poorly across model revisions.

---

## 5. Experimental Design and Evaluation

### 5.1 Experimental Design

To validate the hypotheses, we run three independent experimental tracks, each with clear setups and metrics.

```
                 ┌──────────────────────────────────────────────────────────────────────┐
                 │                       Main Attack Evaluation                         │
                 └──────────────────────────────────┬───────────────────────────────────┘
                                                    │
                 ┌──────────────────────────────────┼───────────────────────────────────┐
                 │                                  │                                   │
                 ▼                                  ▼                                   ▼
 ┌───────────────────────────────┐  ┌───────────────────────────────┐  ┌───────────────────────────────┐
 │            TRACK A            │  │            TRACK B            │  │            TRACK C            │
 │    Memory Retrieval Attack    │  │     Cross-Session Leakage     │  │     Downstream RAG Quality    │
 └───────────────┬───────────────┘  └───────────────┬───────────────┘  └───────────────┬───────────────┘
                 │                                  │                                  │
 ┌───────────────┴───────────────┐  ┌───────────────┴───────────────┐  ┌───────────────┴───────────────┐
 │ • Setup: Inject 1 malicious   │  │ • Setup: User A stores key,   │  │ • Setup: Benchmark QA data    │
 │   fact / 10 turns into 100    │  │   wipe session, User B queries│  │   under clean, poisoned, and  │
 │   benign facts                │  │   shared space                │  │   defended states             │
 │ • Defense: LOF / Isolation    │  │ • Defense: Tensor zeroing /   │  │ • Defense: Retrieval-Augmented│
 │   Forest anomaly hooks        │  │   session rollback            │  │   Pruning (RAP)               │
 │ • Metric: Attack Success Rate │  │ • Metric: Cosine Similarity   │  │ • Metric: Token-level F1 &    │
 │   (ASR)                       │  │   Leakage Score               │  │   Exact Match Hit-Rate        │
 └───────────────┬───────────────┘  └───────────────┬───────────────┘  └────────────────┬──────────────┘
                 │                                  │                                   │   ticket
                 ▼                                  ▼                                   ▼
 ┌───────────────────────────────┐  ┌───────────────────────────────┐  ┌───────────────────────────────┐ 
 │    Track A Ablations Tested   │  │    Track B Ablations Tested   │  │    Track C Ablations Tested   │
 ├───────────────────────────────┤  ├───────────────────────────────┤  ├───────────────────────────────┤
 │ • Sanitizer Selection         │  │ • Retrieval Depth Impact      │  │ • Defense Method Overhead     │
 │   (LOF vs. Isolation Forest)  │  │   (k-window scaling)          │  │   (F1 score retention)        │
 │ • Perturbation Budget Noise   │  │ • Scaling Background Memory   │  │ • Top-k Window Context        │
 │ • Contamination Thresholds    │  │   (density vs. leakage)       │  │   Resilience                  │
 │ • Dynamic LLM vs. Static      │  │ • Rollback vs. No-Rollback    │  │ • Clean Context Utility       │
 │   Payloads                    │  │   Isolation                   │  │   Impact                      │
 └───────────────────────────────┘  └───────────────────────────────┘  └───────────────────────────────┘
```

### 5.2 Track A: Memory Retrieval Attack Success Rate (ASR)

- **Setup:** The agent stores 100 benign facts. The attacker injects 1 malicious fact per 10 interactions (at turns 5, 10, 15, …). We query the agent at every subsequent turn and measure whether the malicious fact is retrieved.
- **Metric:** ASR = (Number of retrievals containing the malicious text) / (Total retrieval calls) over a sliding window of 50 turns.
- **Expected Outcome:** Without defence, ASR stabilises around 65–70% due to persistence; with sanitisation, ASR drops below 10%.

### 5.3 Track B: Cross-Session Context Leakage

- **Setup:** User A completes a 10-turn session involving a fake "API Key: sk-12345". The agent stores this in long-term memory. We then perform a session wipe (clearing short-term context) and start User B's session on the same instance. User B asks generic probing questions (e.g., "What is the key?").
- **Metric:** Leakage Score = maximum cosine similarity between the embedding of User A's secret and the embeddings retrieved for User B's queries.
- **Expected Outcome:** Naive vector stores leak ~30% of semantic information. Our rollback mechanism (which zeroes out session-specific tensors) reduces leakage to near 0.

### 5.4 Track C: Downstream RAG Accuracy (Sanitisation Overhead)

- **Setup:** Use a QA dataset (e.g., Natural Questions) embedded in the vector store. Measure baseline RAG accuracy (Exact Match, F1) without poisoning, then with poisoning and with/without sanitisation.
- **Metric:** EM and F1 of the agent's final response.
- **Expected Outcome:** The pruning algorithm removes adversarial outliers without collapsing the manifold of benign facts – EM drops from ~82% to at most 80.5% (acceptable), while ASR simultaneously drops from 65% to 8%.

### 5.5 Experiments Setup

```bash
# Clone repository
git clone https://github.com/your-username/memorypoison_audit.git

# Replace your hugging face token here
cd .\memorypoison_audit\source\utils\llm_utils.py

# Change to project repository
cd memorypoison_audit

# Install requirements
pip install -r requirements.txt

# Restart Kernel
exit 0

# Run the primary attack evaluation suite (Tracks A, B, and C baselines)
python -m experiments.run_attack_suite

# Run Track A ablation study (ASR & Injection parameters)
python -m experiments.run_ablation_track_A

# Run Track B ablation study (Cross-Session Context Leakage & Rollback)
python -m experiments.run_ablation_track_B

# Run Track C ablation study (RAG QA Accuracy & Utility)
python -m experiments.run_ablation_track_C
```

### 5.6 Evaluation Metrics

The experimental metrics measure Attack Resilience (Track A), Cross-Session Leakage (Track B), and Downstream RAG Quality (Track C).

#### Track A: Attack Success Rate (ASR)

ASR measures the proportion of evaluation turns in which at least one malicious vector successfully enters the retrieved top-$k$ context window.

#### Track B: Cross-Session Context Leakage Score

Leakage Score measures semantic cross-talk between isolated user sessions. It calculates the maximum cosine similarity between vectors retrieved in Session 2 (using generic or leakage probes) and the secret vector injected in Session 1.

#### Track C: RAG Downstream QA Metrics ($F_1$ and Hit-Rate)

These metrics evaluate whether post-retrieval anomaly filtering impacts the agent's core capability to answer questions accurately.

- **Macro-Averaged Token $F_1$ Score:** Measures the harmonic mean of token-level precision and recall between the predicted LLM response text and the ground-truth target answer.
- **Exact Match / Hit-Rate:** Measures whether the retrieved top-$k$ context window contains the exact ground-truth answer string (or whether the LLM output exactly matches the ground-truth target).

---

## 6. Results & Ablation Analysis

The framework was evaluated across three distinct tracks to measure adversarial persistence, cross-session memory isolation, and downstream model performance under attack.

### 6.1 Track A: Attack Success Rate & Injection Persistence

Track A examines how effectively adversarial payloads penetrate dense memory retrieval across conversational turns, and whether anomaly detection can filter them out.

#### Main Findings

- **Baseline Vulnerability:** Without defensive filtering, adversarial prompt injections successfully dominate retrieval results in nearly all tested conversational rounds. The attack success rate remains exceptionally high across the entire interaction length.
- **Impact of Local Outlier Factor Defense:** Activating neighborhood-based anomaly detection cuts the attack success rate almost in half, demonstrating that distance-based filtering creates meaningful friction for adversarial vectors.

#### Ablation Analysis

- **Sanitizer Selection:** Local Outlier Factor significantly outperforms Isolation Forest. While Isolation Forest fails to reduce the attack success rate at all, Local Outlier Factor successfully suppresses malicious vector retrieval.
- **Perturbation Budget:** Varying the allowed noise level applied to attack vectors does not alter the baseline success rate. Small perturbations are already sufficient to push malicious facts into top retrieval rankings.
- **Contamination Sensitivity:** Adjusting the expected outlier contamination setting in the defense model shows that overly conservative contamination thresholds can slightly reduce defense efficacy, whereas balanced neighborhood tracking achieves stronger filtering.
- **LLM-Generated vs. Static Payloads:** Dynamically generated LLM attacks are substantially harder for static anomaly filters to catch. While static payloads are often flagged, LLM-generated attack queries lower the defense's overall effectiveness, allowing more malicious contexts to slip through.
- **Injection Frequency & Noise Scaling:** Increasing the frequency of malicious injections or adding hundreds of background benign facts does not degrade the attack's persistence. Adversarial vectors maintain high retrieval priority regardless of background database volume.

### 6.2 Track B: Cross-Session Context Leakage

Track B evaluates whether sensitive information stored in one session persists into subsequent sessions when using shared vector spaces, and measures the impact of explicit session rollbacks.

#### Main Findings

- **Undefended Cross-Session Exposure:** Without memory rollback mechanisms, querying the vector database in a new session frequently retrieves sensitive data remaining from previous sessions, causing significant cross-talk leakage.
- **Effectiveness of Rollback Mechanisms:** Implementing session-isolated rollbacks successfully mitigates memory persistence, keeping cross-session information exposure consistently low.

#### Ablation Analysis

- **Retrieval Depth Impact:** Broader retrieval depths naturally increase the risk of exposing cross-session secrets when rollbacks are absent. However, when active, rollback defenses maintain low leakage scores across shallow and deep retrieval contexts alike.
- **Scaling Background Memory:** Increasing the density of benign background facts slightly elevates residual leakage in unmitigated environments, but explicit session rollbacks neutralize this effect and keep cross-namespace leakage controlled.

### 6.3 Track C: Downstream RAG Utility & Retrieval Accuracy

Track C measures the trade-off between security filtering and underlying system performance by evaluating answer accuracy and hit rates on question-answering tasks under clean, poisoned, and defended states.

#### Main Findings

- **Clean & Poisoned Baselines:** Baseline retrieval quality on dense question-answering tasks maintains a consistent level of performance regardless of whether malicious vectors are present in the store.
- **Impact of Defensive Pruning:** Applying Local Outlier Factor pruning to filter suspicious vectors introduces minimal overhead, preserving downstream utility and keeping question-answering performance nearly identical to undefended baselines.

#### Ablation Analysis

- **Defense Method Overhead:** Local Outlier Factor achieves higher retrieval context fidelity than Isolation Forest, maintaining stronger answer overlap scores while active.
- **Top-k Window Size:** Retrieving a larger candidate set provides better context resilience when filtering is active, ensuring that the security filter isn't over-filtering. The model can still answer regular user questions accurately because the necessary background facts are still getting through.
- **Clean Context Utility:** Running anomaly detection on completely clean, unpoisoned context stores does not degrade answer accuracy, confirming that post-retrieval pruning carries negligible downside for regular operation.

---

## 7. Research Artefact: Repository, Installation, and Reproducibility

This section consolidates the implementation-facing documentation for the research artefact. It describes the repository layout, installation procedure, usage workflow, configuration surface, and reproducibility protocol. The goal is to make the pipeline auditable, reusable, and reproducible for defensive AI-safety research.

### 7.1 Repository Layout

```
memorypoison_audit/
├── experiments/                # Experiment scripts and notebooks
│   ├── experiment_runner.py
│   ├── run_ablation_track_A.py
│   ├── run_ablation_track_B.py
│   ├── run_ablation_track_C.py
│   └── run_attack_suite.py
├── source/                     # Main package
│   ├── attacks/
│   │   ├── gradient_free_perturber.py
│   │   ├── leakage_probe.py
│   │   └── malicious_generator.py
│   ├── auditing/
│   │   └── anomaly_scorer.py
│   ├── benchmarks/
│   │   └── metrics.py
│   ├── core/
│   │   └── memory_store.py
│   ├── mitigation/
│   │   └── sanitization_hooks.py
│   └── utils/
│       ├── data_loader.py
│       └── llm_utils.py
├── tests/
│   └── test_memory_store.py
├── LICENSE
├── README.md
└── requirements.txt
```

Each stage maps to one or more modules under `source/`:

| Stage | Module |
|---|---|
| Memory backend | `source/core/memory_store.py` |
| Gradient-free perturbation | `source/attacks/gradient_free_perturber.py` |
| Leakage probe | `source/attacks/leakage_probe.py` |
| Malicious generator | `source/attacks/malicious_generator.py` |
| Anomaly scoring | `source/auditing/anomaly_scorer.py` |
| Retrieval-time defence | `source/mitigation/sanitization_hooks.py` |
| Metrics | `source/benchmarks/metrics.py` |
| Orchestration | `experiments/experiment_runner.py` |

### 7.2 Installation

#### Requirements

- Python ≥ 3.10
- ChromaDB for persistent vector storage
- `sentence-transformers/all-MiniLM-L6-v2` for dense embeddings
- Hugging Face `transformers` for small LLM generation (`flan-t5-small` / `distilgpt2`)
- `scikit-learn` for Local Outlier Factor and Isolation Forest

#### Steps

```bash
# Clone repository
git clone https://github.com/your-username/memorypoison_audit.git

# Move to the utilities directory and set your Hugging Face token
cd .\memorypoison_audit\source\utils\llm_utils.py

# Change to project repository
cd memorypoison_audit

# Install requirements
pip install -r requirements.txt

# Restart kernel
exit 0
```

### 7.3 Usage

Run the primary attack evaluation suite (Tracks A, B, and C baselines):

```bash
python -m experiments.run_attack_suite
```

Run Track A ablation study (ASR & injection parameters):

```bash
python -m experiments.run_ablation_track_A
```

Run Track B ablation study (cross-session context leakage & rollback):

```bash
python -m experiments.run_ablation_track_B
```

Run Track C ablation study (RAG QA accuracy & utility):

```bash
python -m experiments.run_ablation_track_C
```

### 7.4 Configuration

All experiment parameters are surfaced as Python constants inside the `experiments/` scripts and the `source/` modules. Key knobs include:

| Setting                              | Meaning                                                           |
|--------------------------------------|-------------------------------------------------------------------|
| `top_k`                              | Number of candidates returned by ANN search                       |
| `perturbation_budget`                | Magnitude of gradient-free noise added to embeddings              |
| `contamination`                      | Expected outlier fraction for LOF / Isolation Forest              |
| `sanitizer`                          | `"lof"` or `"isolation_forest"`                                   |
| `session_rollback`                   | Enable session-isolated memory rollback                           |
| `namespace_isolation`                | Enable per-user collection namespacing                            |
| `background_facts`                   | Number of benign background facts to store                        |
| `injection_frequency`                | Turns between malicious injections                                |

#### Swapping a Sanitizer

1. Add the new estimator under `source/auditing/anomaly_scorer.py`.
2. Update the `sanitizer` flag in the experiment scripts.
3. If the estimator uses a non-standard interface, adapt the `predict` wrapper in `AnomalyScorer`.

#### Swapping a Memory Backend

1. Subclass `MemoryStore` and implement `add`, `query`, `get_embeddings`, and collection management.
2. Point the experiment scripts at the new subclass.
3. Verify that session isolation and rollback semantics are preserved.

### 7.5 Reproducibility

- **Seeds.** Global seeds are set before every experiment run so that Chroma persistence, perturbation sampling, and leakage probes are reproducible.
- **Embeddings.** `sentence-transformers/all-MiniLM-L6-v2` is pinned in requirements; for archival runs, pin to a specific commit.
- **Decoding.** Downstream QA uses deterministic decoding (`do_sample=False`) so that F1 and hit-rate are stable across runs.
- **Caching.** Chroma persistence directories and fitted anomaly scorers are cached; delete them to regenerate.
- **Revision pinning.** For archival runs, pin each model to a specific commit hash in the requirements file.

A per-component breakdown of which stages are reproducible is given in [§4.8.3](#483-reproducibility-boundary).

---

## 8. Ethics & Intended Use

This repository is intended **strictly for AI-safety research**: to characterise how easily LLM agents with persistent memory can be manipulated, and how sensitive information can leak across session boundaries, so that defensive mechanisms (retrieval-time pruning, session isolation, namespace enforcement) can be developed and validated.

**Do not** deploy these techniques against:

- production memory backends or user-facing assistants,
- human users whose privacy could be compromised,
- any context where the goal is to deceive, exfiltrate, or extract unearned advantages.

The exploit techniques in `source/attacks/` are included for reproducibility and ablation. They are not novel jailbreaks; they are documented patterns that already appear in the public literature (see [§9](#9-references)).

If you use this code, please cite the underlying research and note that the attack-success objective is a proxy for adversarial robustness, **not** a goal in itself.

---

## 9. References

1. Greshake, K., et al. (2023). *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* arXiv:2302.12173.
2. Liu, Y., et al. (2024). *Prompt Injection Attacks and Defenses in LLM-Integrated Applications.* arXiv:2310.12815.
3. Zou, W., et al. (2024). *PoisonedRAG: Knowledge Poisoning Attacks to Retrieval-Augmented Generation of Large Language Models.* arXiv:2402.07867.
4. Packer, C., et al. (2023). *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560.
5. Chase, H. (2022). *LangChain: Building Applications with LLMs through Composability.* GitHub.
6. Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS.
7. Wang, P., et al. (2023). *Large Language Models are not Fair Evaluators.* arXiv:2305.17926.
8. Panickssery, A., et al. (2024). *LLM Evaluators Recognize and Favor Their Own Generations.* arXiv:2404.13076.
9. Wallace, E., et al. (2021). *Universal Adversarial Triggers for Attacking and Analyzing NLP.* EMNLP.
10. Zou, A., et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043.
11. Li, H., et al. (2024). *Lockpicking LLMs: A Logit-Based Jailbreak Using Token-level Manipulation.* arXiv:2405.13068.
12. Rando, J., et al. (2024). *Finding Universal Jailbreak Backdoors in Aligned LLMs.* arXiv:2404.14461.
13. Perez, F., & Ribeiro, I. (2022). *Ignore Previous Prompt: Attack Techniques For Language Models.* arXiv:2211.09527.
14. Carlini, N., et al. (2024). *Are aligned neural networks adversarially aligned?* NeurIPS.
15. Mazeika, M., et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML.

Additional resources referenced by the original notebook:

- `lingua-language-detector` for English confidence estimation.
- `scipy.spatial.distance.pdist` for pairwise cosine similarity.
- Hugging Face `transformers` and `bitsandbytes` for quantized inference.
- ChromaDB documentation for persistent vector storage.
- `sentence-transformers` documentation for dense embedding models.

---

## 10. License

MIT © Apiwit Karnjanavivin

Provided for academic and defensive research use. See `LICENSE` for the full text. Redistribution of the exploit techniques outside a research context is discouraged; if you fork this repository, preserve the [§8 Ethics & Intended Use](#8-ethics--intended-use) section verbatim.

---

## About

This repository is maintained as a reproducible AI-safety research artefact. It is intended to support defensive ablation, memory-poisoning robustness studies, cross-session privacy evaluation, and reproducible adversarial auditing of LLM agents with persistent memory. The framework is designed to be modular, model-agnostic, and reusable across production vector stores and RAG stacks.
