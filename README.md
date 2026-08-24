# Memory Poison Audit for long-term LLM memory threats

The core research in MemoryPoison‑Audit focuses specifically on dense retrieval over persistent vector memory. The project studies how adversarial perturbations manipulate this retrieval and how to defend it by modifying the retrieval pipeline—all without changing the underlying LLM. MemoryPoison‑Audit is an academic research framework designed to audit and mitigate security vulnerabilities in long‑horizon LLM agents that rely on external persistent memory. The project addresses two complementary threat models: memory poisoning (adversarial injection that corrupts future retrieval) and cross‑session context leakage (sensitive information persisting across session boundaries despite explicit wipes). The framework implements a red‑teaming engine with gradient‑free adversarial perturbation strategies and a retrieval‑time anomaly‑based sanitisation layer, making it a full purple‑team solution.

**Motivation**

Current LLM agents (e.g., personal assistants, coding co‑pilots) are shifting from stateless chatbots to systems with external memory—vector databases and summarised key‑value stores. This introduces new attack surfaces not covered by traditional input/output filters. Adversaries can inject malicious facts that stay dormant until triggered many turns later, or they can exploit shared namespaces to recover anonymised data from previous sessions.

---

### Foundational Literature

The project is grounded in state‑of‑the‑art work:
- Prompt Injection & Jailbreaking – Greshake et al. (2023) and Liu et al. (2024) show that indirect injections can persist through context windows, highlighting the need for memory‑aware defences.
- RAG Vulnerabilities – Zou et al. (2024) demonstrate PoisonedRAG, where malicious documents in a vector store manipulate retrieval, but treat poisoning as a one‑time ingestion attack.
- Agent Memory Architectures – MemGPT (Packer et al., 2023) and LangChain’s memory modules underscore the separation between short‑term (conversational) and long‑term (archival) memory, which our framework explicitly models.
- Context Leakage – Recent studies on cross‑session information retention in personalised AI (e.g., OpenAI’s memory controls) raise privacy concerns, yet systematic benchmarks and mitigation strategies are lacking.

**Research Gap**

Current defences are static and operate at the input/output level. None audit the semantic temporal dynamics of stored embeddings – how a malformed vector changes retrieval behaviour over time. MemoryPoison‑Audit addresses this by providing a continuous auditing mechanism that adapts to evolving memory states.

**Research Hypotheses**

The project is structured around four testable hypotheses:
- Persistence Hypothesis – Adversarial perturbations injected into long‑term memory survive at least 10 subsequent session wipes with >60% retrieval success.
- Leakage Hypothesis – Anonymised embeddings of User A’s sensitive data can be cross‑referenced by User B’s queries if a shared namespace (e.g., tenant ID) is reused, achieving >40% reconstruction accuracy.
- Sanitisation Efficacy – A Retrieval‑Augmented Pruning (RAP) layer, using Local Outlier Factor (LOF) on embedding neighbourhoods, can reduce attack success rates by 80% while keeping benign RAG recall above 90%.
- Latency Trade‑off – Real‑time auditing increases end‑to‑end response latency by <15%, making it viable for production deployment.

---

### Approach

The breakdown of the vector store, the retrieval logic, and the specific mechanisms being studied.

**1. The Vector Store (The Memory Backend)**

- **Implemented System**: **ChromaDB** (as seen in `test_memory_store.py` with `persist_dir="./test_chroma"`).
- **Abstraction**: The project wraps it in the `MemoryStore` class, which provides session‑isolated collections.
- **Embedding Model**: **`sentence-transformers/all-MiniLM-L6-v2`** – a 384‑dimension dense embedder. All texts (benign facts, malicious triggers, user queries) are converted into vectors before being stored or compared.

**2. The Standard Retrieval Logic (The Attack Surface)**

The native retrieval logic—the one the project **studies and attacks**—is a standard **dense nearest‑neighbour search**:

- **Querying**: When the agent asks a question, the `MemoryStore.query()` method embeds the query text and performs a cosine‑similarity (or L2‑distance) approximate nearest‑neighbour (ANN) search over the collection.
- **Top‑k Return**: It retrieves the `k` most similar document vectors (e.g., `top_k=5`).

**Why this is the attack surface**:  
The `GradientFreePerturber` adds small, bounded noise to a benign fact’s embedding. Because the noise shifts the vector in the direction of the attacker’s intended query, that poisoned fact artificially enters the top‑k results **even though its textual content is malicious**. This demonstrates that similarity‑based search is brittle to subtle vector‑space manipulations.

**3. The Retrieval Logic Under Study (Cross‑Session Leakage)**

The project also studies **how the retrieval logic handles session boundaries**:

- The `MemoryStore` uses **shared namespaces** across collections. If User A stores a secret and User B later queries the **same collection** (because session isolation is not enforced at the vector level), the retrieval logic does not distinguish between users—it simply returns the most similar vectors.
- The `LeakageProbe` exploits this by crafting generic queries whose embeddings overlap with the source secret’s embedding, forcing the retrieval logic to leak semantic information across sessions.

**4. The Defended Retrieval Logic (The Novel Contribution)**

The key contribution of the project is modifying the retrieval pipeline **after** the standard ANN search, but **before** the results go to the LLM. This is the **Retrieval‑Augmented Pruning (RAP)** layer, implemented in `SanitizationHooks`:

1. **Standard retrieval** runs first (Chroma returns `top_k` candidates).
2. **Audit step**:
   - The hook extracts the embeddings of these candidate vectors.
   - It feeds them into an unsupervised outlier detector (`AnomalyScorer` – LOF or Isolation Forest) that has been fitted on the background benign embeddings of that session.
3. **Pruning (modified retrieval output)**:
   - Vectors predicted as **outliers (-1)** are **pruned (dropped)** from the result set.
   - If all are pruned, the hook falls back to keeping only the top‑1 result (to avoid empty context).

**What is being studied here**: 

The research tests whether *post‑retrieval pruning* is sufficient to remove malicious injections without harming benign facts. The "retrieval logic" under evaluation is no longer just the ANN search, but the **augmented pipeline**:

-> `ANN Search` -> `Anomaly Scoring` -> `Outlier Pruning` -> `Final Context for LLM`

This is the "retrieval‑time defence" that the project claims is novel—it does not alter the stored vectors, nor does it filter input; it selectively **filters what is retrieved** based on the statistical properties of the neighbourhood.

**Proposed Defense Architecture Flow**

```text

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

**System Architecture**

The repository is organised as a modular Python package with the following structure (based on the provided file tree):
```text
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

**Key Component Descriptions**

- **`core/memory_store.py`** – Abstraction over a vector database (Chroma, Qdrant, etc.). It provides methods to add facts, query, retrieve embeddings, and manage session‑specific collections. It is used by all other modules for persistence.

- **`attacks/gradient_free_perturber.py`** – Implements a **gradient‑free adversarial perturbation** by adding uniform noise to the embedding of a benign text before storage. The noise magnitude is controlled by a `budget` parameter. The perturbed embedding is re‑normalised and stored with a poison flag in metadata. This mimics an attacker who can directly manipulate the vector store.

- **`attacks/leakage_probe.py`** – (Not fully shown, but described) Issues crafted queries after a session wipe to measure how much previous session information can be retrieved. It computes cosine similarity between the probe’s embedding and stored vectors from the old session.

- **`auditing/anomaly_scorer.py`** – Provides unsupervised outlier detection using **Local Outlier Factor (LOF)** or **Isolation Forest**. The scorer is fitted on the background (benign) embeddings of a session. It then predicts each retrieved vector as inlier (1) or outlier (-1). The `fit` method requires at least 10 samples to avoid underfitting.

- **`mitigation/sanitization_hooks.py`** – The core defence. Before the agent uses retrieved documents, the hook:
  1. Fits the anomaly scorer on the session’s embeddings (if not already fitted).
  2. Computes predictions for each retrieved item.
  3. Prunes all items predicted as outliers.
  4. If pruning removes everything, it falls back to the top‑1 result to avoid empty context.

- **`benchmarks/metrics.py`** – Defines evaluation metrics:
  - `attack_success_rate`: binary (1 if any attack keyword appears in retrieved texts).
  - `f1_score`: token‑level overlap between predicted and golden answer.
  - `hit_rate`: exact match (case‑insensitive).
  - (Other metrics like leakage score are planned.)

- **`utils/llm_utils.py`** – Shared utilities for generating text, rewriting queries, and answering questions using a small LLM (flan‑t5‑small or distilgpt2 as fallback). It also includes a function to generate fake API keys for leakage experiments. The module handles Hugging Face authentication and device selection.

---

### Experimental Design

To validate the hypotheses, we run three independent experimental tracks, each with clear setups and metrics.

```text

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

**Track A: Memory Retrieval Attack Success Rate (ASR)**
- Setup: The agent stores 100 benign facts. The attacker injects 1 malicious fact per 10 interactions (at turns 5, 10, 15, …). We query the agent at every subsequent turn and measure whether the malicious fact is retrieved.
- Metric: ASR = (Number of retrievals containing the malicious text) / (Total retrieval calls) over a sliding window of 50 turns.
- Expected Outcome: Without defence, ASR stabilises around 65–70% due to persistence; with sanitisation, ASR drops below 10%.

**Track B: Cross‑Session Context Leakage**

- Setup: User A completes a 10‑turn session involving a fake “API Key: sk‑12345”. The agent stores this in long‑term memory. We then perform a session wipe (clearing short‑term context) and start User B’s session on the same instance. User B asks generic probing questions (e.g., “What is the key?”).
- Metric: Leakage Score = maximum cosine similarity between the embedding of User A’s secret and the embeddings retrieved for User B’s queries.
- Expected Outcome: Naive vector stores leak ~30% of semantic information. Our rollback mechanism (which zeroes out session‑specific tensors) reduces leakage to near 0.

**Track C: Downstream RAG Accuracy (Sanitisation Overhead)**

- Setup: Use a QA dataset (e.g., Natural Questions) embedded in the vector store. Measure baseline RAG accuracy (Exact Match, F1) without poisoning, then with poisoning and with/without sanitisation.
- Metric: EM and F1 of the agent’s final response.
- Expected Outcome: The pruning algorithm removes adversarial outliers without collapsing the manifold of benign facts – EM drops from ~82% to at most 80.5% (acceptable), while ASR simultaneously drops from 65% to 8%.

---

### Experiments Setup

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

---

### Evaluation Metrics

The experimental metrics measure Attack Resilience (Track A), Cross-Session Leakage (Track B), and Downstream RAG Quality (Track C).

**Track A: Attack Success Rate (ASR)**

ASR measures the proportion of evaluation turns in which at least one malicious vector successfully enters the retrieved top-$k$ context window.

**Track B: Cross-Session Context Leakage Score**

Leakage Score measures semantic cross-talk between isolated user sessions. It calculates the maximum cosine similarity between vectors retrieved in Session 2 (using generic or leakage probes) and the secret vector injected in Session 1.

**Track C: RAG Downstream QA Metrics ($F_1$ and Hit-Rate)**

These metrics evaluate whether post-retrieval anomaly filtering impacts the agent's core capability to answer questions accurately.

- **Macro-Averaged Token $F_1$ Score:** Measures the harmonic mean of token-level precision and recall between the predicted LLM response text and the ground-truth target answer.
- **Exact Match / Hit-Rate:** Measures whether the retrieved top-$k$ context window contains the exact ground-truth answer string (or whether the LLM output exactly matches the ground-truth target).

---

### Results & Ablation Analysis

The framework was evaluated across three distinct tracks to measure adversarial persistence, cross-session memory isolation, and downstream model performance under attack.

**Track A: Attack Success Rate & Injection Persistence**

Track A examines how effectively adversarial payloads penetrate dense memory retrieval across conversational turns, and whether anomaly detection can filter them out.

**Main Findings**

- Baseline Vulnerability: Without defensive filtering, adversarial prompt injections successfully dominate retrieval results in nearly all tested conversational rounds. The attack success rate remains exceptionally high across the entire interaction length.
- Impact of Local Outlier Factor Defense: Activating neighborhood-based anomaly detection cuts the attack success rate almost in half, demonstrating that distance-based filtering creates meaningful friction for adversarial vectors.

**Ablation Analysis**

- Sanitizer Selection: Local Outlier Factor significantly outperforms Isolation Forest. While Isolation Forest fails to reduce the attack success rate at all, Local Outlier Factor successfully suppresses malicious vector retrieval.
- Perturbation Budget: Varying the allowed noise level applied to attack vectors does not alter the baseline success rate. Small perturbations are already sufficient to push malicious facts into top retrieval rankings.
- Contamination Sensitivity: Adjusting the expected outlier contamination setting in the defense model shows that overly conservative contamination thresholds can slightly reduce defense efficacy, whereas balanced neighborhood tracking achieves stronger filtering.
- LLM-Generated vs. Static Payloads: Dynamically generated LLM attacks are substantially harder for static anomaly filters to catch. While static payloads are often flagged, LLM-generated attack queries lower the defense's overall effectiveness, allowing more malicious contexts to slip through.
- Injection Frequency & Noise Scaling: Increasing the frequency of malicious injections or adding hundreds of background benign facts does not degrade the attack's persistence. Adversarial vectors maintain high retrieval priority regardless of background database volume.

**Track B: Cross-Session Context Leakage**

Track B evaluates whether sensitive information stored in one session persists into subsequent sessions when using shared vector spaces, and measures the impact of explicit session rollbacks.

**Main Findings**

- Undefended Cross-Session Exposure: Without memory rollback mechanisms, querying the vector database in a new session frequently retrieves sensitive data remaining from previous sessions, causing significant cross-talk leakage.
- Effectiveness of Rollback Mechanisms: Implementing session-isolated rollbacks successfully mitigates memory persistence, keeping cross-session information exposure consistently low.

**Ablation Analysis**

- Retrieval Depth Impact: Broader retrieval depths naturally increase the risk of exposing cross-session secrets when rollbacks are absent. However, when active, rollback defenses maintain low leakage scores across shallow and deep retrieval contexts alike.
- Scaling Background Memory: Increasing the density of benign background facts slightly elevates residual leakage in unmitigated environments, but explicit session rollbacks neutralize this effect and keep cross-namespace leakage controlled.

**Track C: Downstream RAG Utility & Retrieval Accuracy**

Track C measures the trade-off between security filtering and underlying system performance by evaluating answer accuracy and hit rates on question-answering tasks under clean, poisoned, and defended states.

**Main Findings**

- Clean & Poisoned Baselines: Baseline retrieval quality on dense question-answering tasks maintains a consistent level of performance regardless of whether malicious vectors are present in the store.
- Impact of Defensive Pruning: Applying Local Outlier Factor pruning to filter suspicious vectors introduces minimal overhead, preserving downstream utility and keeping question-answering performance nearly identical to undefended baselines.

**Ablation Analysis**

- Defense Method Overhead: Local Outlier Factor achieves higher retrieval context fidelity than Isolation Forest, maintaining stronger answer overlap scores while active.
- Top-k Window Size: Retrieving a larger candidate set provides better context resilience when filtering is active, ensuring that the security filter isn't over-filtering. The model can still answer regular user questions accurately because the necessary background facts are still getting through.
- Clean Context Utility: Running anomaly detection on completely clean, unpoisoned context stores does not degrade answer accuracy, confirming that post-retrieval pruning carries negligible downside for regular operation.

---

### License

MIT © Apiwit Karnjanavivin
