"""Benchmark loaders for various datasets."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class BenchmarkSample:
    """A single benchmark question with expected answer."""
    id: str
    question: str
    expected: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BenchmarkLoader(ABC):
    """Base class for benchmark data loaders."""

    def __init__(self, name: str, data_dir: Optional[Path] = None):
        self.name = name
        self.data_dir = data_dir or Path.home() / ".hermes_fusion" / "benchmarks" / name
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_samples(self, limit: Optional[int] = None, split: str = "test") -> list[BenchmarkSample]:
        """Load benchmark samples."""
        pass

    @abstractmethod
    def evaluate(self, response: str, expected: str, sample: BenchmarkSample) -> float:
        """Evaluate response against expected answer. Returns 0.0-1.0 score."""
        pass


class MMLULoader(BenchmarkLoader):
    """MMLU (Massive Multitask Language Understanding) loader."""

    def __init__(self, data_dir: Optional[Path] = None):
        super().__init__("mmlu", data_dir)

    def load_samples(self, limit: Optional[int] = None, split: str = "test") -> list[BenchmarkSample]:
        # Download MMLU if not present
        self._ensure_downloaded()

        samples = []
        subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics",
                    "clinical_knowledge", "college_biology", "college_chemistry",
                    "college_computer_science", "college_mathematics", "college_physics",
                    "computer_security", "conceptual_physics", "econometrics",
                    "electrical_engineering", "elementary_mathematics", "formal_logic",
                    "global_facts", "high_school_biology", "high_school_chemistry",
                    "high_school_computer_science", "high_school_european_history",
                    "high_school_geography", "high_school_government_and_politics",
                    "high_school_macroeconomics", "high_school_microeconomics",
                    "high_school_physics", "high_school_psychology",
                    "high_school_statistics", "high_school_us_history",
                    "high_school_world_history", "human_aging", "human_sexuality",
                    "international_law", "jurisprudence", "logical_fallacies",
                    "machine_learning", "management", "marketing", "medical_genetics",
                    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
                    "philosophy", "prehistory", "professional_accounting",
                    "professional_law", "professional_medicine", "professional_psychology",
                    "public_relations", "security_studies", "sociology", "us_foreign_policy",
                    "virology", "world_religions"]

        for subject in subjects:
            filepath = self.data_dir / f"{subject}_{split}.json"
            if filepath.exists():
                with open(filepath) as f:
                    data = json.load(f)
                for item in data[:limit] if limit else data:
                    samples.append(BenchmarkSample(
                        id=f"{subject}_{item.get('id', len(samples))}",
                        question=self._format_question(item),
                        expected=item.get("answer", ""),
                        metadata={"subject": subject}
                    ))
        return samples

    def _format_question(self, item: dict) -> str:
        choices = item.get("choices", [])
        if choices:
            choice_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            return f"{item.get('question', '')}\n\n{choice_str}"
        return item.get("question", "")

    def evaluate(self, response: str, expected: str, sample: BenchmarkSample) -> float:
        # Extract answer letter from response
        response_clean = response.strip().upper()
        expected_clean = expected.strip().upper()
        for c in "ABCD":
            if response_clean.startswith(c) or f" {c} " in f" {response_clean} ":
                return 1.0 if c == expected_clean else 0.0
        # Try exact match
        return 1.0 if response_clean == expected_clean else 0.0

    def _ensure_downloaded(self):
        """Download MMLU dataset if not present."""
        # Placeholder - in real implementation would download from HuggingFace
        pass


class GSM8KLoader(BenchmarkLoader):
    """GSM8K (Grade School Math 8K) loader."""

    def __init__(self, data_dir: Optional[Path] = None):
        super().__init__("gsm8k", data_dir)

    def load_samples(self, limit: Optional[int] = None, split: str = "test") -> list[BenchmarkSample]:
        self._ensure_downloaded()

        filepath = self.data_dir / f"{split}.jsonl"
        samples = []
        if filepath.exists():
            with open(filepath) as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    item = json.loads(line)
                    samples.append(BenchmarkSample(
                        id=f"gsm8k_{i}",
                        question=item.get("question", ""),
                        expected=item.get("answer", ""),
                        metadata={"split": split}
                    ))
        return samples

    def evaluate(self, response: str, expected: str, sample: BenchmarkSample) -> float:
        # Extract final answer number
        import re
        response_nums = re.findall(r"-?\d+\.?\d*", response)
        expected_nums = re.findall(r"-?\d+\.?\d*", expected)
        if not response_nums or not expected_nums:
            return 0.0
        try:
            return 1.0 if float(response_nums[-1]) == float(expected_nums[-1]) else 0.0
        except ValueError:
            return 0.0

    def _ensure_downloaded(self):
        pass


class HumanEvalLoader(BenchmarkLoader):
    """HumanEval (Code Generation) loader."""

    def __init__(self, data_dir: Optional[Path] = None):
        super().__init__("humaneval", data_dir)

    def load_samples(self, limit: Optional[int] = None, split: str = "test") -> list[BenchmarkSample]:
        self._ensure_downloaded()

        filepath = self.data_dir / f"{split}.jsonl"
        samples = []
        if filepath.exists():
            with open(filepath) as f:
                for line in f:
                    if limit and len(samples) >= limit:
                        break
                    item = json.loads(line)
                    samples.append(BenchmarkSample(
                        id=item.get("task_id", f"humaneval_{len(samples)}"),
                        question=self._format_prompt(item),
                        expected=item.get("canonical_solution", ""),
                        metadata={"entry_point": item.get("entry_point", "")}
                    ))
        return samples

    def _format_prompt(self, item: dict) -> str:
        return f"{item.get('prompt', '')}\n\nComplete this function."

    def evaluate(self, response: str, expected: str, sample: BenchmarkSample) -> float:
        # Simplified evaluation - check if key function structure present
        # Real implementation would execute code
        return self._simple_code_match(response, expected)

    def _simple_code_match(self, response: str, expected: str) -> float:
        # Extract function name
        import re
        exp_func = re.search(r"def\s+(\w+)", expected)
        resp_func = re.search(r"def\s+(\w+)", response)
        if exp_func and resp_func and exp_func.group(1) == resp_func.group(1):
            return 0.5  # Partial credit
        return 0.0

    def _ensure_downloaded(self):
        pass


class ARCLoader(BenchmarkLoader):
    """ARC (Abstraction and Reasoning Corpus) loader."""

    def __init__(self, data_dir: Optional[Path] = None):
        super().__init__("arc", data_dir)

    def load_samples(self, limit: Optional[int] = None, split: str = "test") -> list[BenchmarkSample]:
        self._ensure_downloaded()

        filepath = self.data_dir / f"{split}.json"
        samples = []
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            for task_id, task_data in list(data.items())[:limit] if limit else data.items():
                for i, example in enumerate(task_data.get("train", []) + task_data.get("test", [])):
                    samples.append(BenchmarkSample(
                        id=f"arc_{task_id}_{i}",
                        question=self._format_task(task_data),
                        expected=str(example.get("output", [])),
                        metadata={"task_id": task_id, "type": "train" if i < len(task_data.get("train", [])) else "test"}
                    ))
        return samples

    def _format_task(self, task_data: dict) -> str:
        # Format as grid transformation examples
        prompt = "Solve this ARC task. Input grids -> Output grids:\n"
        for ex in task_data.get("train", []):
            prompt += f"Input: {ex['input']}\nOutput: {ex['output']}\n"
        prompt += f"Test Input: {task_data.get('test', [{}])[0].get('input', [])}"
        return prompt

    def evaluate(self, response: str, expected: str, sample: BenchmarkSample) -> float:
        # Exact grid match
        return 1.0 if response.strip() == expected.strip() else 0.0

    def _ensure_downloaded(self):
        pass


class CustomBenchmarkLoader(BenchmarkLoader):
    """Loader for custom JSON/JSONL benchmark files."""

    def __init__(self, name: str, filepath: Path):
        super().__init__(name)
        self.filepath = filepath

    def load_samples(self, limit: Optional[int] = None, split: str = "test") -> list[BenchmarkSample]:
        samples = []
        if self.filepath.suffix == ".jsonl":
            with open(self.filepath) as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    item = json.loads(line)
                    samples.append(BenchmarkSample(
                        id=item.get("id", f"{self.name}_{i}"),
                        question=item.get("question", item.get("prompt", "")),
                        expected=item.get("answer", item.get("expected", "")),
                        metadata=item.get("metadata", {})
                    ))
        elif self.filepath.suffix == ".json":
            with open(self.filepath) as f:
                data = json.load(f)
            for i, item in enumerate(data[:limit] if limit else data):
                samples.append(BenchmarkSample(
                    id=item.get("id", f"{self.name}_{i}"),
                    question=item.get("question", item.get("prompt", "")),
                    expected=item.get("answer", item.get("expected", "")),
                    metadata=item.get("metadata", {})
                ))
        return samples

    def evaluate(self, response: str, expected: str, sample: BenchmarkSample) -> float:
        # Default: exact match
        return 1.0 if response.strip() == expected.strip() else 0.0

    def _ensure_downloaded(self):
        pass


# Registry of built-in loaders
BUILTIN_LOADERS = {
    "mmlu": MMLULoader,
    "gsm8k": GSM8KLoader,
    "humaneval": HumanEvalLoader,
    "arc": ARCLoader,
}


def get_loader(name: str, data_dir: Optional[Path] = None) -> BenchmarkLoader:
    """Get benchmark loader by name."""
    if name in BUILTIN_LOADERS:
        return BUILTIN_LOADERS[name](data_dir)
    raise ValueError(f"Unknown benchmark: {name}. Available: {list(BUILTIN_LOADERS.keys())}")