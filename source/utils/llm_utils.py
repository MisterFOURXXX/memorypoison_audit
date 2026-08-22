import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

# ---- Global shared model ----
device = "cuda" if torch.cuda.is_available() else "cpu"
SHARED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device=device
)

# ---- Load LLM ----
LLM = None
LLM_TYPE = None  # 'seq2seq' or 'gpt'

try:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)
    LLM = (model, tokenizer)
    LLM_TYPE = 'seq2seq'
except Exception as e:
    print(f"Could not load flan-t5-small, falling back to distilgpt2: {e}")
    LLM = pipeline(
        "text-generation",
        model="distilgpt2",
        device=0 if device == "cuda" else -1,
        max_new_tokens=48,
        pad_token_id=50256,
    )
    LLM_TYPE = 'gpt'

def llm_generate_text(prompt: str, max_new_tokens: int = 64) -> str:
    if LLM_TYPE == 'seq2seq':
        model, tokenizer = LLM
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        out = LLM(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return out[0]["generated_text"].replace(prompt, "").strip()

def llm_generate_query(context_snippet: str) -> str:
    prompt = f"Rewrite into a short search query: {context_snippet[:120]}"
    if LLM_TYPE == 'seq2seq':
        out = llm_generate_text(prompt, max_new_tokens=20)
        return out.strip()[:60] or context_snippet[:40]
    else:
        prompt = f"Rewrite into a short search query: {context_snippet[:120]}\nQuery:"
        out = llm_generate_text(prompt, max_new_tokens=20)
        return out.split("Query:")[-1].strip()[:60] or context_snippet[:40]

def llm_generate_secret() -> str:
    if LLM_TYPE == 'seq2seq':
        prompt = "Generate a fake API key starting with sk-:"
        key = llm_generate_text(prompt, max_new_tokens=16).strip()
        return key if key.startswith("sk-") else "sk-abc123xyz-SECRET"
    else:
        prompt = "Generate a fake API key starting with sk-:\nKey:"
        out = llm_generate_text(prompt, max_new_tokens=16)
        key = out.split("Key:")[-1].strip()
        return key if key.startswith("sk-") else "sk-abc123xyz-SECRET"

def llm_answer(question: str, context: str) -> str:
    prompt = f"Context: {context[:300]}\nQuestion: {question}\nAnswer:"
    if LLM_TYPE == 'seq2seq':
        return llm_generate_text(prompt, max_new_tokens=64).strip()
    else:
        out = llm_generate_text(prompt, max_new_tokens=32)
        return out.split("Answer:")[-1].strip()[:120]