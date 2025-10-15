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
from comp import MLModelClient
from openai import OpenAI


# ---------------------------------------------------------------------------
# Answer Generation
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """Handles answer generation from retrieved context using LLM."""
    
    def __init__(self, service_host: str = None, openai_host: str = None, openai_api_key: str = None):
        """Initialize the answer generator with LLM service."""
        self.service = MLModelClient(host=service_host or "http://localhost:8000")
        # Use OpenAI client for answer generation
        self.openai_client = OpenAI(
            base_url=openai_host or "http://localhost:11434/v1",
            api_key=openai_api_key or "ollama"
        )
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate a concise answer from retrieved context using LLM with enhanced prompting."""
        try:
            print(f"🔍 DEBUG: generate_answer called with query='{query}' and {len(context)} context items")
            
            if not context:
                return ""
            
            # Filter and clean context to remove irrelevant information
            filtered_context = self._filter_relevant_context(query, context)
            print(f"🔍 DEBUG: Filtered context: {len(filtered_context)} items")
            
            if not filtered_context:
                return ""
            
            # Join context into a single string
            context_text = "\n".join(filtered_context)
            print(f"🔍 DEBUG: Context text length: {len(context_text)} chars")
            
            # Classify question type for better prompting
            question_type = self._classify_question_type(query)
            print(f"🔍 DEBUG: Question type: {question_type}")
            
            # Create enhanced prompt based on question type
            if question_type == "yes_no":
                prompt = f"""Analyze the context step by step to answer this yes/no question.

Question: {query}

Context:
{context_text}

Think through this step by step:
1. What specific information is needed to answer this question?
2. What does the context tell us about this information?
3. Based on the evidence, is the answer yes or no?

Answer with ONLY 'yes' or 'no':"""
                
            elif question_type == "name":
                prompt = f"""Analyze the context to find the specific person's name being asked about.

Question: {query}

Context:
{context_text}

Think through this step by step:
1. What person is the question asking about?
2. What information about this person is in the context?
3. What is the person's name?

Answer with ONLY the name:"""
                
            elif question_type == "title_position":
                prompt = f"""Analyze the context to find the specific job title or position being asked about.

Question: {query}

Context:
{context_text}

Think through this step by step:
1. What title or position is the question asking about?
2. What information about this title/position is in the context?
3. What is the exact title or position?

Answer with ONLY the title/position:"""
                
            else:  # description
                prompt = f"""Analyze the context to provide a precise answer to this question.

Question: {query}

Context:
{context_text}

Think through this step by step:
1. What specific information is the question asking for?
2. What relevant information is available in the context?
3. What is the most accurate answer based on the context?

Provide a brief, direct answer:"""

            print(f"🔍 DEBUG: Using OpenAI client for answer generation")
            print(f"🔍 DEBUG: Prompt length: {len(prompt)} chars")
            
            # Call LLM with enhanced prompt
            response = self.openai_client.chat.completions.create(
                model="gpt-oss:latest",
                messages=[
                    {"role": "system", "content": "You are a precise question-answering assistant. Your job is to extract the exact answer from the provided context. CRITICAL: Always provide your final answer in the 'content' field. For yes/no questions, respond with only 'yes' or 'no'. For names, respond with only the name. For titles/positions, respond with only the title/position. For other questions, provide a brief, direct answer. Do not include explanations, reasoning, or additional text in your response - just the answer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,  # Increased to prevent truncation
                temperature=0.0  # Use 0 for more consistent results
            )
            
            print(f"🔍 DEBUG: LLM response: {response}")
            
            # Extract answer from response
            if response and response.choices:
                content = response.choices[0].message.content
                reasoning = getattr(response.choices[0].message, 'reasoning', None)
                
                print(f"🔍 DEBUG: Raw answer: '{content}'")
                
                if content and content.strip():
                    # Clean and validate the answer
                    cleaned_answer = self._clean_answer(content.strip())
                    print(f"🔍 DEBUG: Cleaned answer: '{cleaned_answer}'")
                    
                    # Validate and potentially correct the answer
                    validated_answer = self._validate_and_correct_answer(cleaned_answer, query, question_type, context_text)
                    
                    return validated_answer
                elif reasoning:
                    # If content is empty but reasoning is available, extract from reasoning
                    print(f"🔍 DEBUG: Using reasoning content: '{reasoning[:100]}...'")
                    extracted_answer = self._extract_answer_from_reasoning(reasoning, query, question_type)
                    print(f"🔍 DEBUG: Extracted from reasoning: '{extracted_answer}'")
                    
                    if extracted_answer and extracted_answer.strip():
                        cleaned_answer = self._clean_answer(extracted_answer)
                        print(f"🔍 DEBUG: Cleaned answer: '{cleaned_answer}'")
                        
                        # Validate and potentially correct the answer
                        validated_answer = self._validate_and_correct_answer(cleaned_answer, query, question_type, context_text)
                        
                        return validated_answer
                    else:
                        print("⚠️ Could not extract answer from reasoning")
                        return ""
                else:
                    print("⚠️ No content or reasoning in LLM response")
                    return ""
            else:
                print("⚠️ No response from LLM")
                return ""
                
        except Exception as e:
            print(f"⚠️ Answer generation failed: {e}")
            return ""
    
    def _classify_question_type(self, query: str) -> str:
        """Classify the question type to use appropriate prompting strategy."""
        query_lower = query.lower()
        
        # Check for yes/no questions - must start with auxiliary verbs and be asking for yes/no
        yes_no_patterns = [
            "were", "was", "did", "does", "do", "are", "is", "have", "has", "had", "can", "could", "would", "should"
        ]
        
        # Check if it's asking for a specific name or title
        if any(word in query_lower for word in ["what position", "what title", "what role", "what job", "what government position", "held", "served as", "was named"]):
            return "title_position"
        elif any(word in query_lower for word in ["who", "what person", "what man", "what woman"]):
            return "name"
        elif query_lower.startswith(tuple(yes_no_patterns)):
            return "yes_no"
        elif any(word in query_lower for word in ["what", "which", "where"]):
            return "description"
        else:
            return "description"
    
    def _filter_relevant_context(self, query: str, context: List[str]) -> List[str]:
        """Filter context to keep only the most relevant pieces."""
        query_terms = set(query.lower().split())
        relevant_context = []
        
        # Remove common stop words from query terms for better matching
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should"}
        query_terms = query_terms - stop_words
        
        for item in context:
            if not item or not item.strip():
                continue
                
            # Check for term overlap
            item_terms = set(item.lower().split())
            overlap = len(query_terms.intersection(item_terms))
            
            # Keep items with some overlap or first few items
            if overlap > 0 or len(relevant_context) < 3:
                relevant_context.append(item)
        
        # Sort by relevance but don't be too restrictive
        relevant_context.sort(key=lambda x: len(set(query.lower().split()).intersection(set(x.lower().split()))), reverse=True)
        
        return relevant_context
    
    def _extract_answer_from_reasoning(self, reasoning: str, query: str, question_type: str) -> str:
        """Extract answer from reasoning text when content is empty."""
        reasoning_lower = reasoning.lower()
        query_lower = query.lower()
        
        # For yes/no questions, look for yes/no in reasoning
        if question_type == "yes_no":
            if "answer is yes" in reasoning_lower or "so answer: yes" in reasoning_lower:
                return "yes"
            elif "answer is no" in reasoning_lower or "so answer: no" in reasoning_lower:
                return "no"
            elif "yes" in reasoning_lower and "no" not in reasoning_lower:
                return "yes"
            elif "no" in reasoning_lower and "yes" not in reasoning_lower:
                return "no"
        
        # Try to extract answers using patterns - enhanced for better extraction
        answer_patterns = [
            r"answer:\s*([^\n\.]+)",
            r"the\s+answer\s+is\s+([^\n\.]+)",
            r"answer\s+is\s+([^\n\.]+)",
            r"therefore\s+([^\n\.]+)",
            r"thus\s+([^\n\.]+)",
            r"so\s+([^\n\.]+?)(?:\.|$)",
            r"so answer:\s*([^\n\.]+)",
            r"the answer:\s*([^\n\.]+)",
            r"answer:\s*([^\n\.]+)",
            r"([A-Z][a-zA-Z\s]{3,50})(?:\s+(?:is|was|are|were)\s+(?:the|a|an))",
            r"(?:is|was|are|were)\s+(?:a|an|the)?\s*([A-Z][a-zA-Z\s]{3,50})(?:\.|,|$)"
        ]
        
        for pattern in answer_patterns:
            import re
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            for answer in matches:
                answer = answer.strip()
                if (len(answer) > 2 and 
                    answer.lower() not in ["context", "information", "data", "none", "unknown", "not found", "not mentioned", "the question", "the context"] and
                    not answer.startswith("but") and
                    not answer.startswith("however") and
                    not answer.startswith("actually") and
                    not answer.startswith("the question") and
                    not answer.startswith("the context")):
                    return answer
        
        # Extract from truncated reasoning - look for partial answers
        if "so answer:" in reasoning_lower:
            # Extract text after "so answer:"
            parts = reasoning.split("so answer:")
            if len(parts) > 1:
                answer_part = parts[1].strip()
                # Clean up the answer part
                answer_part = answer_part.replace('"', '').replace("'", '').strip()
                if answer_part and len(answer_part) < 50:  # Reasonable length
                    return answer_part
        
        # Try to extract from the end of reasoning if truncated
        reasoning_lines = reasoning.split('\n')
        for line in reversed(reasoning_lines[-3:]):  # Check last 3 lines
            line = line.strip()
            if len(line) > 5 and not line.startswith(('but', 'however', 'actually', 'wait', 'let')):
                # Look for potential answers in the last lines
                import re
                potential_answers = re.findall(r"([A-Z][a-zA-Z\s]{2,30})", line)
                for answer in potential_answers:
                    if len(answer) > 2 and len(answer) < 30:
                        return answer.strip()
        
        return ""
    
    def _validate_and_correct_answer(self, answer: str, query: str, question_type: str, context: str) -> str:
        """Validate and potentially self-correct the answer."""
        if not answer or answer.lower() in ["i don't know", "unknown", "not mentioned", "not found"]:
            return answer
        
        # Basic validation checks
        if len(answer) < 2:
            return answer
        
        # For title/position questions, ensure the answer looks like a title/position
        if question_type == "title_position":
            # Check if answer looks like a title/position
            if len(answer) < 3 or answer.lower() in ["title", "position", "job", "role"]:
                # Try to find titles in context
                import re
                title_pattern = r"([A-Z][a-zA-Z\s]{3,30})(?:\s+(?:of|for|in|at|to)\s+[A-Z][a-zA-Z\s]+)?"
                titles = re.findall(title_pattern, context)
                for title in titles:
                    title_lower = title.lower()
                    if any(indicator in title_lower for indicator in ["chief", "secretary", "director", "president", "minister", "ambassador", "governor", "mayor", "officer", "manager", "supervisor", "coordinator", "administrator", "consultant", "advisor"]):
                        return title.strip()
                
                # Try to extract from context using query keywords
                query_lower = query.lower()
                if "government position" in query_lower:
                    gov_pattern = r"([A-Z][a-zA-Z\s]+)(?:\s+(?:of|for|in|at|to)\s+[A-Z][a-zA-Z\s]+)?"
                    gov_matches = re.findall(gov_pattern, context)
                    for match in gov_matches:
                        if len(match) > 5 and any(word in match.lower() for word in ["ambassador", "secretary", "minister", "director", "chief"]):
                            return match.strip()
                
                return answer
        
        return answer
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and validate the generated answer."""
        if not answer:
            return ""
        
        # Remove common prefixes that LLMs sometimes add
        prefixes_to_remove = [
            "the answer is", "answer:", "the answer:", "answer is", 
            "the answer is:", "answer:", "the answer is", "answer is:",
            "based on the context", "according to the context", "from the context"
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        
        # Remove quotes and extra whitespace
        answer = answer.strip('"\' \n\t')
        
        # Remove trailing punctuation if it's not part of the answer
        if answer.endswith('.') and len(answer) > 3:
            answer = answer[:-1]
        
        return answer.strip()


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
        answer_generator: Optional[AnswerGenerator] = None,
    ):
        self.pipeline = pipeline
        self.query_kwargs = query_kwargs or {}
        self.combine_strategy = combine_strategy
        self.collect_predictions = collect_predictions
        self.sleep = sleep
        self.answer_generator = answer_generator

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

            # If answer generator is available, use it to generate answers from retrieved context
            if self.answer_generator and responses:
                try:
                    generated_answer = self.answer_generator.generate_answer(example.question, responses)
                    if generated_answer and generated_answer.strip():
                        prediction_text = generated_answer
                    else:
                        prediction_text = self._combine_responses(responses)
                except Exception as e:
                    print(f"⚠️ Answer generation failed: {e}")
                    prediction_text = self._combine_responses(responses)
            else:
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
    
    # Create answer generator for LLM-based answer generation
    answer_generator = AnswerGenerator()
    
    evaluator = HybridRAGEvaluator(
        pipeline,
        query_kwargs=query_kwargs,
        combine_strategy=args.combine,
        collect_predictions=args.collect_predictions,
        sleep=args.sleep,
        answer_generator=answer_generator,
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
