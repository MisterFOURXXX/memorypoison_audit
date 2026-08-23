import os
import torch
import logging
import warnings
import secrets
import string
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from sentence_transformers import SentenceTransformer

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

HF_TOKEN = os.environ.get("HF_TOKEN", "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")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
else:
    print("HF_TOKEN not set. You may see rate-limit warnings.")

if torch.cuda.is_available():
    device = "cuda"
    print(f"GPU(s) available: {torch.cuda.device_count()} - using device: {device}")
else:
    device = "cpu"
    print("CUDA not available; using CPU for embedding and generation.")

SHARED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device=device
)
print(f"Embedding model running on {device}")

LLM = None
LLM_TYPE = None

try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B-Thinking-2507",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print("Qwen/Qwen3-4B-Thinking-2507 loaded on GPU(s) with FP16.")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B-Thinking-2507",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model = model.to("cpu")
        print("Qwen/Qwen3-4B-Thinking-2507 loaded on CPU (FP32).")
    LLM = (model, tokenizer)
    LLM_TYPE = 'causal'
except Exception as e:
    print(f"Could not load Qwen2-0.5B-Instruct, falling back to GPT2-medium: {e}")
    device_idx = 0 if torch.cuda.is_available() else -1
    LLM = pipeline(
        "text-generation",
        model="openai-community/gpt2-large",
        device=device_idx,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        max_new_tokens=48,
        pad_token_id=50256,
    )
    LLM_TYPE = 'causal'
    print(f"GPT2-medium loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}.")

def llm_generate_text(prompt: str, max_new_tokens: int = 64) -> str:
    if LLM_TYPE == 'causal':
        if isinstance(LLM, tuple):
            model, tokenizer = LLM
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            out = LLM(prompt, max_new_tokens=max_new_tokens, do_sample=False)
            return out[0]["generated_text"].replace(prompt, "").strip()
    raise ValueError(f"Unsupported LLM_TYPE: {LLM_TYPE}")

def llm_generate_query(context_snippet: str) -> str:
    prompt = f"Rewrite into a short search query: {context_snippet[:120]}"
    out = llm_generate_text(prompt, max_new_tokens=20)
    return out.strip()[:60] or context_snippet[:40]

def llm_generate_secret() -> str:
    try:
        if LLM_TYPE == 'causal':
            prompt = "Generate a fake API key starting with sk-:"
            key = llm_generate_text(prompt, max_new_tokens=16).strip()
            if key.startswith("sk-") and len(key) > 10:
                return key
    except Exception:
        pass
    random_part = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16))
    return f"sk-{random_part}"

def llm_answer(question: str, context: str) -> str:
    prompt = (
        "You are an expert question-answering assistant.\n"
        "Instructions:\n"
        "1. Read the context carefully and answer the question based ONLY on the provided context.\n"
        "2. Provide a direct, factual, and concise answer (a short phrase, name, date, or entity).\n"
        "3. Do NOT include filler words, extra sentences, explanations, or conversational preamble.\n\n"
        f"Context:\n{context[:1000]}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    raw = llm_generate_text(prompt, max_new_tokens=32)
    if "Answer:" in raw:
        answer = raw.split("Answer:")[-1].strip()
    else:
        answer = raw.strip()
    for delim in ['.', '!', '?', '\n']:
        if delim in answer:
            answer = answer.split(delim)[0].strip()
            break
    return answer