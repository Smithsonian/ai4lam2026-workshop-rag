"""Reference implementations for the evaluation tutorial module."""

import re
from typing import Literal

import json
import os
import ollama
from pydantic import BaseModel, Field

CHAT_MODEL = "gemma4:e4b"
OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", -1))


class JudgeVerdict(BaseModel):
    """Structured output returned by the LLM evaluator."""

    faithfulness_score: int = Field(
        ge=0, le=10, description="Faithfulness score from 0 to 10"
    )
    relevance_score: int = Field(
        ge=0, le=10, description="Relevance score from 0 to 10"
    )
    overall_verdict: Literal["PASS", "PARTIAL", "FAIL"] = Field(
        description="Overall evaluation verdict"
    )
    reason: str = Field(description="Short explanation of the score")


def normalize_words(text: str) -> set[str]:
    """Lowercase text and return a set of simple word tokens."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def calculate_keyword_overlap(generated: str, expected: str) -> float:
    """Compute the fraction of expected keywords found in the generated answer."""
    expected_words = normalize_words(expected)
    generated_words = normalize_words(generated)

    if not expected_words:
        return 0.0

    overlap = expected_words & generated_words
    return len(overlap) / len(expected_words)


def llm_judge(query: str, context: str, answer: str) -> JudgeVerdict:
    """Use an LLM to score faithfulness and relevance with schema validation."""
    prompt = f"""You are an expert RAG evaluator.

Score the answer on two dimensions:
1. faithfulness: Does the answer stay grounded in the provided context?
2. relevance: Does the answer answer the user's question?

Context: {context}
Question: {query}
Answer: {answer}
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        # Pass the JSON schema directly from the Pydantic model
        format=JudgeVerdict.model_json_schema(),
        messages=[{"role": "user", "content": prompt}],
        keep_alive=OLLAMA_KEEP_ALIVE,
    )

    # Validate and convert directly into the typed Pydantic object
    return JudgeVerdict.model_validate_json(response["message"]["content"])


if __name__ == "__main__":
    samples = [
        {
            "label": "Good answer",
            "query": "What is the habitable zone?",
            "context": (
                "The habitable zone is the region around a star where liquid water "
                "could potentially exist on a planet's surface."
            ),
            "expected": "The habitable zone is the region around a star where liquid water can exist.",
            "answer": "It is the area around a star where liquid water could exist on a planet's surface.",
        },
        {
            "label": "Hallucinated answer",
            "query": "What is the habitable zone?",
            "context": (
                "The habitable zone is the region around a star where liquid water "
                "could potentially exist on a planet's surface."
            ),
            "expected": "The habitable zone is the region around a star where liquid water can exist.",
            "answer": "It is the region where stars are born from cold gas clouds.",
        },
    ]

    for sample in samples:
        overlap_score = calculate_keyword_overlap(sample["answer"], sample["expected"])
        judge_result = llm_judge(sample["query"], sample["context"], sample["answer"])

        print(sample["label"])
        print(f"Overlap score: {overlap_score:.2f}")
        # Convert back to JSON dict for printing
        print(json.dumps(judge_result.model_dump(), indent=2))
        print("-" * 40)
