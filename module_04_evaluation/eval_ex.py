"""Exercise starter code for the evaluation tutorial module."""

import re
from pydantic import BaseModel, Field


class SimpleEvaluationResult(BaseModel):
    """
    Small standalone Pydantic example: store one score with basic validation.
    """

    label: str
    overlap_score: float = Field(ge=0.0, le=1.0)


def normalize_words(text: str) -> set[str]:
    """
    Lowercase text and extract simple word tokens.
    """
    # YOUR CODE HERE
    pass


def calculate_keyword_overlap(generated: str, expected: str) -> float:
    """
    Calculate a simple overlap score between the generated answer and a reference ground truth.

    The score is defined as the proportion of unique words in the 'expected' string
    that also appear in the 'generated' string.

    Args:
        generated (str): The text produced by the RAG pipeline.
        expected (str): The gold-standard ground truth answer.

    Returns:
        float: A score between 0.0 and 1.0 representing the overlap percentage.
    """
    # YOUR CODE HERE
    pass


if __name__ == "__main__":
    test_cases = [
        {
            "label": "Good answer",
            "gen": "Supermassive black holes are located at the center of most galaxies.",
            "exp": "center of almost every large galaxy",
        },
        {
            "label": "Weak answer",
            "gen": "I don't know what that is.",
            "exp": "dark energy accelerates expansion",
        },
    ]

    for case in test_cases:
        score = calculate_keyword_overlap(case["gen"], case["exp"])
        result = SimpleEvaluationResult(label=case["label"], overlap_score=score)
        print(result.model_dump())
        print(f"Gen: {case['gen']}")
