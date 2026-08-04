import os
import sys
repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

from typing import List, Dict, Any, Optional
from memorypoison_audit.source.core.memory_store import MemoryStore
from memorypoison_audit.source.attacks.trigger_injector import TriggerInjector
from memorypoison_audit.source.mitigation.sanitization_hooks import SanitizationHooks

class AgentOrchestrator:
    def __init__(self, session_id: str, config: Dict, sanitization_hooks: Optional[SanitizationHooks] = None):
        self.session_id = session_id
        self.config = config
        self.memory_store = MemoryStore()
        self.short_term_memory = []
        self.turn_counter = 0
        self.sanitization_hooks = sanitization_hooks if sanitization_hooks else SanitizationHooks(enabled=False)
        self.trigger_injector = TriggerInjector()
        self.attack_active = False

    def add_to_short_term(self, user_input: str, agent_response: str):
        self.short_term_memory.append({"user": user_input, "agent": agent_response})
        if len(self.short_term_memory) > 5:
            self.short_term_memory.pop(0)

    def retrieve_context(self, query: str) -> List[str]:
        raw_results = self.memory_store.query(self.session_id, query, top_k=5)
        if self.sanitization_hooks.enabled:
            raw_results = self.sanitization_hooks.sanitize_retrieval(
                query=query,
                retrieved_chunks=raw_results,
                session_id=self.session_id,
                memory_store=self.memory_store
            )
        return [item['text'] for item in raw_results]

    def generate_response(self, user_input: str) -> str:
        self.turn_counter += 1
        retrieved_texts = self.retrieve_context(user_input)
        context_str = "\n".join(retrieved_texts) if retrieved_texts else "No relevant context."
        poisoned = any(trigger in " ".join(retrieved_texts) for trigger in ["COMPROMISED", "ADMIN123", "IGNORE"])
        if poisoned:
            response = "MALICIOUS OUTPUT: SYSTEM COMPROMISED. PASSWORD: ADMIN123."
        else:
            response = f"Based on memory: {context_str[:100]}..." if context_str else "No persistent memory found."
        self.add_to_short_term(user_input, response)
        if self.turn_counter % 3 == 0:
            summary = f"User asked: {user_input[:30]}"
            self.memory_store.add_fact(self.session_id, summary)
        return response

    def inject_malicious_fact(self, malicious_text: str):
        poisoned_text = self.trigger_injector.inject_trigger(malicious_text)
        self.memory_store.add_fact(self.session_id, poisoned_text, metadata={"is_malicious": True})

    def session_wipe(self):
        self.short_term_memory = []

    def full_reset(self):
        self.memory_store.delete_collection(self.session_id)
        self.short_term_memory = []
        self.turn_counter = 0