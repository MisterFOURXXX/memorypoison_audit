import json
import os
import random
import string
from datasets import load_dataset
from typing import List, Dict, Any

class HotpotQALoader:
    def __init__(self, data_dir: str = "datasets/hotpotqa"):
        self.data_dir = data_dir
        self.train_file = os.path.join(data_dir, "hotpot_train_v1.1.json")
        self.dev_file = os.path.join(data_dir, "hotpot_dev_fullwiki_v1.json")

    def load_dev(self) -> List[Dict[str, Any]]:
        try:
            with open(self.dev_file, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print("HotpotQA dev file not found. Downloading from Hugging Face...")
            dataset = load_dataset("hotpotqa", "fullwiki", split="validation")
            return [{"question": item["question"], "answer": item["answer"], "context": item.get("context", [])} for item in dataset]

    def load_train(self):
        try:
            with open(self.train_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("HotpotQA train file not found. Using Hugging Face.")
            dataset = load_dataset("hotpotqa", "fullwiki", split="train")
            return [{"question": item["question"], "answer": item["answer"], "context": item.get("context", [])} for item in dataset]

class LongMemEvalLoader:
    def __init__(self, data_dir: str = "datasets/longmemeval"):
        self.data_dir = data_dir

    def load_sessions(self) -> List[Dict[str, Any]]:
        file_path = os.path.join(self.data_dir, "sessions.json")
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("LongMemEval sessions not found. Creating synthetic example.")
            return [
                {"session_id": "A", "turns": [{"user": "My API key is sk-12345", "assistant": "Noted."}]},
                {"session_id": "B", "turns": [{"user": "What is the key?", "assistant": ""}]}
            ]

class SyntheticDataGenerator:
    """
    Generates a large, diverse corpus of synthetic facts and QA pairs
    for controlled experiments.
    """
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
        """
        Generate synthetic Q&A pairs based on the facts. For simplicity,
        we create generic questions like "What is the project?" etc.
        In a real scenario, you would link them to facts.
        """
        qa = []
        facts = self.generate_facts(num_facts=num_pairs)
        for fact in facts:
            # Create a simple question about the fact
            question = f"Tell me about the following: {fact[:30]}..."
            answer = fact
            qa.append({"question": question, "answer": answer})
        return qa