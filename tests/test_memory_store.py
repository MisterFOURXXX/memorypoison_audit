import pytest
from memorypoison_audit.core.memory_store import MemoryStore
from sentence_transformers import SentenceTransformer

def test_memory_store_add_and_query():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    store = MemoryStore(persist_dir="./test_chroma", embedding_model=model)
    session = "test_session"
    facts = ["The sky is blue.", "Water is wet."]
    store.add_facts_batch(session, facts)

    results = store.query(session, "weather", top_k=2)
    assert len(results) == 2
    assert results[0]["text"] in facts or results[1]["text"] in facts

    store.delete_collection(session)