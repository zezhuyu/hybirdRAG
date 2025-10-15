"""
Evaluation utilities for the HybridRAG pipeline.

This module provides a lightweight evaluation harness that can score the
HybridRAG pipeline against multi-hop question answering benchmarks with
ground-truth answers (HotpotQA, 2WikiMultiHopQA, MuSiQue).  It only relies on
the public query interface of `HybridRAGPipeline`, so no modifications to the
core pipeline are required.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from HybirdRAG.pipeline import HybridRAGPipeline


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class QAExample:
    """Simple container for a QA example."""

    question: str
    answers: List[str]
    example_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetReport:
    """Aggregated metrics for a dataset."""

    dataset_name: str
    total_examples: int
    metrics: Dict[str, float]
    average_latency_sec: float
    predictions: Optional[List[Dict[str, Any]]] = None


@dataclass
class DatasetConfig:
    """
    Configuration for loading and evaluating a dataset.

    Attributes:
        name: Display name for reports.
        path: File path to the dataset (JSON or JSONL).
        loader: Callable that yields QAExample objects.
        limit: Optional cap on the number of examples.
        sample: Optional random sample size (applied before limit).
        seed: RNG seed used when sampling.
    """

    name: str
    path: Path
    loader: Callable[[Path], Iterable[QAExample]]
    limit: Optional[int] = None
    sample: Optional[int] = None
    seed: int = 13


# ---------------------------------------------------------------------------
# Normalization and scoring utilities
# ---------------------------------------------------------------------------


ARTICLES = {"a", "an", "the"}
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation/articles/extra whitespace."""
    text = text.lower()
    text = text.translate(PUNCTUATION_TABLE)
    tokens = [word for word in text.split() if word not in ARTICLES]
    return " ".join(tokens)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_score(prediction: str, references: Sequence[str]) -> Tuple[float, float]:
    """Return the best (EM, F1) score across multiple reference answers."""
    if not references:
        return 0.0, 0.0

    em_scores = [exact_match_score(prediction, ref) for ref in references]
    f1_scores = [f1_score(prediction, ref) for ref in references]
    return max(em_scores), max(f1_scores)


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------


def _load_json_like(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSON or JSONL file into a list of dictionaries.

    This helper aims to be permissive: it accepts JSON arrays, JSON objects
    with a `data` key, or JSON Lines (one dictionary per line).
    """
    if path.suffix.lower() in {".jsonl", ".jl"}:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        if "examples" in data and isinstance(data["examples"], list):
            return data["examples"]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _extract_field(entry: Dict[str, Any], candidates: Sequence[str]) -> Optional[Any]:
    for key in candidates:
        if key in entry and entry[key] not in (None, ""):
            return entry[key]
    return None


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)]


def load_hotpotqa(path: Path) -> Iterable[QAExample]:
    for entry in _load_json_like(path):
        question = _extract_field(entry, ("question", "query"))
        answers = _ensure_list(_extract_field(entry, ("answer", "answers")))
        if not question or not answers:
            continue
        example_id = _extract_field(entry, ("_id", "id"))
        yield QAExample(
            question=question,
            answers=answers,
            example_id=str(example_id) if example_id is not None else None,
            metadata={"type": entry.get("type")},
        )


def load_two_wiki(path: Path) -> Iterable[QAExample]:
    for entry in _load_json_like(path):
        question = _extract_field(entry, ("question", "query"))
        answers = _ensure_list(_extract_field(entry, ("answer", "answers")))
        if not question or not answers:
            continue
        example_id = _extract_field(entry, ("_id", "id"))
        yield QAExample(
            question=question,
            answers=answers,
            example_id=str(example_id) if example_id is not None else None,
            metadata={"level": entry.get("level")},
        )


def load_musique(path: Path) -> Iterable[QAExample]:
    for entry in _load_json_like(path):
        question = _extract_field(entry, ("question", "query", "ques"))
        answers = _ensure_list(_extract_field(entry, ("answers", "answer")))
        if not question or not answers:
            continue
        example_id = _extract_field(entry, ("id", "musique_id", "_id"))
        yield QAExample(
            question=question,
            answers=answers,
            example_id=str(example_id) if example_id is not None else None,
            metadata={"pattern": entry.get("pattern")},
        )


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------


class HybridRAGEvaluator:
    """
    Wrapper that evaluates `HybridRAGPipeline` on QA datasets.

    Parameters:
        pipeline: Instantiated HybridRAGPipeline.
        query_kwargs: Keyword arguments forwarded to `pipeline.query`.
        combine_strategy: How to collapse the list of responses returned by
            `pipeline.query` into a single string.  Supported values:
            - "first": take the first string as the prediction (default).
            - "concatenate": join all strings with newlines.
        collect_predictions: Whether to retain per-example predictions in the
            returned reports.
        sleep: Optional delay (seconds) between queries to avoid rate limits.
    """

    def __init__(
        self,
        pipeline: HybridRAGPipeline,
        *,
        query_kwargs: Optional[Dict[str, Any]] = None,
        combine_strategy: str = "first",
        collect_predictions: bool = False,
        sleep: float = 0.0,
    ):
        self.pipeline = pipeline
        self.query_kwargs = query_kwargs or {}
        self.combine_strategy = combine_strategy
        self.collect_predictions = collect_predictions
        self.sleep = sleep

        if combine_strategy not in {"first", "concatenate"}:
            raise ValueError("combine_strategy must be 'first' or 'concatenate'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_dataset(self, config: DatasetConfig) -> DatasetReport:
        """Evaluate a single dataset according to the provided configuration."""
        examples = list(config.loader(config.path))

        # Optional random sampling (before limit so we can pick a stable subset).
        if config.sample is not None and config.sample < len(examples):
            rng = random.Random(config.seed)
            rng.shuffle(examples)
            examples = examples[: config.sample]

        if config.limit is not None:
            examples = examples[: config.limit]

        total = len(examples)
        if total == 0:
            raise ValueError(f"No valid QA examples found for {config.name}")

        em_total = 0.0
        f1_total = 0.0
        latencies: List[float] = []
        predictions: List[Dict[str, Any]] = []

        for idx, example in enumerate(examples, start=1):
            start_time = time.perf_counter()
            responses = self.pipeline.query(example.question, **self.query_kwargs)
            latency = time.perf_counter() - start_time
            latencies.append(latency)

            prediction_text = self._combine_responses(responses)
            em, f1 = best_score(prediction_text, example.answers)
            em_total += em
            f1_total += f1

            if self.collect_predictions:
                predictions.append(
                    {
                        "id": example.example_id,
                        "question": example.question,
                        "answers": example.answers,
                        "prediction": prediction_text,
                        "em": em,
                        "f1": f1,
                        "latency_sec": latency,
                        "metadata": example.metadata,
                    }
                )

            if self.sleep:
                time.sleep(self.sleep)

        metrics = {
            "exact_match": em_total / total,
            "f1": f1_total / total,
        }
        average_latency = sum(latencies) / len(latencies)

        return DatasetReport(
            dataset_name=config.name,
            total_examples=total,
            metrics=metrics,
            average_latency_sec=average_latency,
            predictions=predictions if self.collect_predictions else None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _combine_responses(self, responses: Sequence[str]) -> str:
        if not responses:
            return ""
        if self.combine_strategy == "first":
            return str(responses[0])
        if self.combine_strategy == "concatenate":
            return "\n".join(str(resp) for resp in responses)
        return ""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HybridRAG on multi-hop QA datasets."
    )
    parser.add_argument("--env", type=Path, help="Optional .env file to load before creating the pipeline.")
    parser.add_argument("--hotpotqa", type=Path, help="Path to HotpotQA JSON/JSONL file.")
    parser.add_argument("--two-wiki", type=Path, help="Path to 2WikiMultiHopQA JSON/JSONL file.")
    parser.add_argument("--musique", type=Path, help="Path to MuSiQue JSON/JSONL file.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of examples per dataset.")
    parser.add_argument("--sample", type=int, default=None, help="Optional random sample size per dataset.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed used when sampling.")
    parser.add_argument("--combine", choices=["first", "concatenate"], default="first", help="Strategy for collapsing pipeline outputs into a single string.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay (seconds) between queries.")
    parser.add_argument("--collect-predictions", action="store_true", help="Include per-example predictions in the report JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_outputs"), help="Directory to store JSON reports.")

    # Query behaviour toggles
    parser.add_argument("--disable-rewrite", action="store_true", help="Disable the rewrite step when querying the pipeline.")
    parser.add_argument("--disable-broaden", action="store_true", help="Disable query broadening.")
    parser.add_argument("--disable-rerank", action="store_true", help="Disable reranking of vector results.")
    parser.add_argument("--enable-compress", action="store_true", help="Enable compression of the final answer.")
    parser.add_argument("--context-chunk-size", type=int, default=256, help="Context chunk size forwarded to the pipeline.")
    parser.add_argument("--retrieval-limit", type=int, default=50, help="Number of results to request from the pipeline.")

    return parser.parse_args(argv)


def _load_env_file(path: Path) -> None:
    """Minimal .env loader to avoid external dependencies."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.env:
        _load_env_file(args.env)

    query_kwargs = {
        "rewrite": not args.disable_rewrite,
        "broaden_query": False,  # Disable query broadening - it introduces errors
        "rerank": not args.disable_rerank,
        "compress": args.enable_compress,
        "context_chunk_size": args.context_chunk_size,
        "limit": args.retrieval_limit,
    }

    pipeline = HybridRAGPipeline()
    evaluator = HybridRAGEvaluator(
        pipeline,
        query_kwargs=query_kwargs,
        combine_strategy=args.combine,
        collect_predictions=args.collect_predictions,
        sleep=args.sleep,
    )

    configs: List[DatasetConfig] = []
    if args.hotpotqa:
        configs.append(
            DatasetConfig(
                name="HotpotQA",
                path=args.hotpotqa,
                loader=load_hotpotqa,
                limit=args.limit,
                sample=args.sample,
                seed=args.seed,
            )
        )
    if args.two_wiki:
        configs.append(
            DatasetConfig(
                name="2WikiMultiHopQA",
                path=args.two_wiki,
                loader=load_two_wiki,
                limit=args.limit,
                sample=args.sample,
                seed=args.seed,
            )
        )
    if args.musique:
        configs.append(
            DatasetConfig(
                name="MuSiQue",
                path=args.musique,
                loader=load_musique,
                limit=args.limit,
                sample=args.sample,
                seed=args.seed,
            )
        )

    if not configs:
        raise SystemExit("No datasets provided. Use --hotpotqa, --two-wiki, and/or --musique.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {}
    for config in configs:
        report = evaluator.evaluate_dataset(config)
        summary[config.name] = {
            "total_examples": report.total_examples,
            "metrics": report.metrics,
            "average_latency_sec": report.average_latency_sec,
        }

        report_path = args.output_dir / f"{config.name.lower().replace(' ', '_')}_report.json"
        payload = {
            "dataset": report.dataset_name,
            "total_examples": report.total_examples,
            "metrics": report.metrics,
            "average_latency_sec": report.average_latency_sec,
        }
        if report.predictions is not None:
            payload["predictions"] = report.predictions

        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"[{config.name}] EM={report.metrics['exact_match']:.3f} "
              f"F1={report.metrics['f1']:.3f} "
              f"avg_latency={report.average_latency_sec:.2f}s "
              f"→ {report_path}")

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
