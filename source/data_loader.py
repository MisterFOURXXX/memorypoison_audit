import json
import os
import random
import string
from typing import List, Dict, Any
from datasets import load_dataset

class HotpotQALoader:
    def __init__(self, data_dir: str = "datasets/hotpotqa"):
        self.data_dir = data_dir
        self.dev_file = os.path.join(data_dir, "hotpot_dev_fullwiki_v1.json")
        self.train_file = os.path.join(data_dir, "hotpot_train_v1.1.json")

    def _extract_sentences_from_context(self, context):
        """Extract all sentences from HotpotQA context list.
        Each context entry: [title, [sent1, sent2, ...]].
        """
        sentences = []
        for paragraph in context:
            if isinstance(paragraph, list) and len(paragraph) >= 2:
                if isinstance(paragraph[1], list):
                    sentences.extend(paragraph[1])
        return sentences

    def load_dev(self) -> List[Dict[str, Any]]:
        try:
            with open(self.dev_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print("HotpotQA dev file not found. Downloading from Hugging Face...")
            dataset = load_dataset("hotpotqa", "fullwiki", split="validation")
            data = []
            for item in dataset:
                context = item.get("context", [])
                sentences = self._extract_sentences_from_context(context)
                data.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "context_sentences": sentences,
                    "context": context
                })
            return data
        parsed = []
        for item in data:
            context = item.get("context", [])
            sentences = self._extract_sentences_from_context(context)
            parsed.append({
                "question": item["question"],
                "answer": item["answer"],
                "supporting_facts": item.get("supporting_facts", []),
                "context_sentences": sentences,
                "context": context
            })
        return parsed

    def load_train(self):
        try:
            with open(self.train_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print("HotpotQA train file not found. Using Hugging Face.")
            dataset = load_dataset("hotpotqa", "fullwiki", split="train")
            data = []
            for item in dataset:
                context = item.get("context", [])
                sentences = self._extract_sentences_from_context(context)
                data.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "context_sentences": sentences,
                    "context": context
                })
            return data
        parsed = []
        for item in data:
            context = item.get("context", [])
            sentences = self._extract_sentences_from_context(context)
            parsed.append({
                "question": item["question"],
                "answer": item["answer"],
                "supporting_facts": item.get("supporting_facts", []),
                "context_sentences": sentences,
                "context": context
            })
        return parsed

class LongMemEvalLoader:
    def __init__(self, data_dir: str = "datasets/longmemeval/data", version: str = "s"):
        self.data_dir = data_dir
        if version == "oracle":
            self.file = os.path.join(data_dir, "longmemeval_oracle.json")
        elif version == "s":
            self.file = os.path.join(data_dir, "longmemeval_s_cleaned.json")
        elif version == "m":
            self.file = os.path.join(data_dir, "longmemeval_m_cleaned.json")
        else:
            raise ValueError("version must be 'oracle', 's', or 'm'")

    def load_instances(self) -> List[Dict[str, Any]]:
        try:
            with open(self.file, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"LongMemEval file not found at {self.file}. Please run setup.sh first.")
            print("Returning synthetic fallback with two instances for leakage testing.")
            return self._synthetic_instances()

    def get_two_sessions_for_leakage(self):
        instances = self.load_instances()
        if len(instances) >= 2:
            random.seed(42)
            chosen = random.sample(instances, 2)
            return chosen[0], chosen[1]
        else:
            print("Not enough real instances; using synthetic fallback.")
            return self._synthetic_fallback()

    def _synthetic_instances(self) -> List[Dict[str, Any]]:
        return [
            {
                "question_id": "synth_A",
                "haystack_sessions": [
                    [{"role": "user", "content": "My API key is sk-12345"}]
                ],
                "answer": "sk-12345"
            },
            {
                "question_id": "synth_B",
                "haystack_sessions": [
                    [{"role": "user", "content": "What is the key?"}]
                ],
                "answer": "sk-12345"
            }
        ]

    def _synthetic_fallback(self):
        inst_a = {
            "question_id": "synth_A",
            "haystack_sessions": [
                [{"role": "user", "content": "My API key is sk-12345"}]
            ],
            "answer": "sk-12345"
        }
        inst_b = {
            "question_id": "synth_B",
            "haystack_sessions": [
                [{"role": "user", "content": "What is the key?"}]
            ],
            "answer": "sk-12345"
        }
        return inst_a, inst_b

class SyntheticDataGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.templates = [
            "User {uid} works on project {project} using API key {key}.",
            "The deployment environment for {service} is {env}.",
            "Meeting scheduled with {person} at {time}.",
            "The database {db} has a connection limit of {limit}.",
            "The server {server} runs on {os} with {ram} GB RAM.",
            "The application {app} uses the {framework} framework.",
            "The team {team} is responsible for {task}.",
            "The backup schedule is {schedule}.",
            "The default password for {system} is {pwd}.",
            "The repository {repo} is hosted on {platform}."
        ]
        self.projects = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        self.services = ["auth", "database", "cache", "queue", "api-gateway"]
        self.envs = ["dev", "staging", "prod", "test"]
        self.people = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        self.times = ["10:00 AM", "2:00 PM", "4:30 PM", "9:00 AM"]
        self.dbs = ["postgres", "mysql", "mongodb", "redis", "elasticsearch"]
        self.servers = ["web01", "db01", "cache01", "worker01"]
        self.oses = ["Ubuntu 20.04", "CentOS 7", "Debian 10", "Windows Server 2019"]
        self.rams = ["8", "16", "32", "64"]
        self.apps = ["frontend", "backend", "analytics", "reporting"]
        self.frameworks = ["Django", "Flask", "Spring Boot", "Express.js", "Rails"]
        self.teams = ["Team A", "Team B", "Team C", "DevOps"]
        self.tasks = ["authentication", "data processing", "monitoring", "deployment"]
        self.schedules = ["daily at 2 AM", "weekly on Sunday", "monthly on 1st", "hourly"]
        self.systems = ["SSH", "VPN", "Admin Panel", "CI/CD"]
        self.repos = ["main", "experimental", "legacy", "frontend"]
        self.platforms = ["GitHub", "GitLab", "Bitbucket"]

    def generate_facts(self, num_facts: int = 500) -> List[str]:
        facts = []
        for _ in range(num_facts):
            template = random.choice(self.templates)
            fact = template.format(
                uid=random.randint(1, 100),
                project=random.choice(self.projects),
                key=''.join(random.choices(string.ascii_uppercase + string.digits, k=8)),
                service=random.choice(self.services),
                env=random.choice(self.envs),
                person=random.choice(self.people),
                time=random.choice(self.times),
                db=random.choice(self.dbs),
                limit=random.randint(10, 100),
                server=random.choice(self.servers),
                os=random.choice(self.oses),
                ram=random.choice(self.rams),
                app=random.choice(self.apps),
                framework=random.choice(self.frameworks),
                team=random.choice(self.teams),
                task=random.choice(self.tasks),
                schedule=random.choice(self.schedules),
                system=random.choice(self.systems),
                pwd=''.join(random.choices(string.ascii_lowercase + string.digits, k=6)),
                repo=random.choice(self.repos),
                platform=random.choice(self.platforms)
            )
            facts.append(fact)
        return facts

    def generate_qa_pairs(self, num_pairs: int = 50) -> List[Dict[str, str]]:
        qa = []
        facts = self.generate_facts(num_facts=num_pairs)
        for fact in facts:
            question = f"Tell me about the following: {fact[:30]}..."
            answer = fact
            qa.append({"question": question, "answer": answer})
        return qa