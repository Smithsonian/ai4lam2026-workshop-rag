# Module 4: Evaluating RAG Accuracy

Once you have a pipeline, how do you know if it's actually working? In RAG, we usually care about at least two things:
- **Faithfulness**: Is the answer actually supported by the retrieved context?
- **Relevance**: Does the answer respond to the user question?

This module introduces evaluation in two layers:
1. a **word-overlap baseline**
2. a detailed **LLM-as-a-judge** evaluation that returns structured output

## Prerequisites

This module makes the most sense after you have already run Module 3 and seen example RAG answers.

## Key Concepts
- **Ground Truth**: A set of questions and their known correct answers.
- **Lexical Baseline**: A weak but easy metric based on how many important words overlap with a reference answer.
- **Quantitative Evaluation**: Using metrics output by another LLM as a "judge."
- **Qualitative Evaluation**: Human review of samples based on specific criteria.
- **Structured Output**: Returning JSON from the judge so scores can be logged, filtered, or visualized later.

## Pydantic In Isolation

Before using Pydantic with an LLM, it helps to see the core idea by itself.

Pydantic lets you define the shape of valid data with a normal Python class:

```python
from pydantic import BaseModel, Field

class SimpleScore(BaseModel):
    label: str
    overlap_score: float = Field(ge=0.0, le=1.0)

result = SimpleScore(label="good answer", overlap_score=0.75)
print(result.model_dump())
```

Why this is useful:
- `label` must be a string
- `overlap_score` must be a number between `0.0` and `1.0`
- after validation, you can safely work with a structured Python object instead of loose dictionaries

If the data is invalid, Pydantic raises a validation error:

```python
SimpleScore(label="bad answer", overlap_score=1.5)
```

That fails because `1.5` is outside the allowed range.

In this module, we use the same pattern for LLM evaluation:
- define a schema with fields like `faithfulness_score`
- ask the model to return JSON matching that schema
- validate the model output before using it

## Why Start With Word Overlap?

Before involving another model, it helps to build a baseline that is easy to understand.

The overlap score in this module asks:
- Which unique words appear in the expected answer?
- How many of those same words also appear in the generated answer?

This is useful because it is:
- fast
- local
- easy to explain in class

But it is also limited:
- it does not understand synonyms
- it does not detect subtle hallucinations well
- it can reward answers that copy words without actually answering the question

That is why overlap is a baseline, not a final evaluation strategy.

## Exercises
1. `eval_ex.py`: Write a simple script that checks if certain keywords from the correct answer appear in the generated answer.
2. Optional extension: Wrap the overlap result in a tiny Pydantic model so the score becomes structured data before you involve an LLM.

Suggested command:

```bash
python3 module_04_evaluation/eval_ex.py
```

Expected outcome:
- The first sample should score noticeably higher than the second.
- This is intentionally a weak baseline; it only checks lexical overlap.
- If you try the optional Pydantic extension, you should be able to print a validated Python object or JSON-ready dictionary.

## Solutions
See `eval_sln.py` for a combined demonstration:
- first compute the overlap-word baseline
- then ask an LLM judge to score the answer
- return the judge result as JSON so it is machine-readable

Suggested command:

```bash
python3 module_04_evaluation/eval_sln.py
```

Expected outcome:
- The good answer should receive a higher overlap score than the hallucinated answer.
- The JSON judge output should include separate `faithfulness_score` and `relevance_score` fields.
- The hallucinated answer should score poorly on faithfulness because it is not supported by the provided context.

## Why JSON Output Helps

Free-form judge text is readable, but structured output is more useful in a real workflow.

JSON makes it easier to:
- save evaluation results to a file or dataframe
- compare many answers programmatically
- build dashboards or Streamlit views later
- define pass/fail thresholds in code

For example, a judge record might look like:

```json
{
  "faithfulness_score": 9,
  "relevance_score": 8,
  "overall_verdict": "PASS",
  "reason": "The answer is grounded in the context and directly answers the question."
}
```

In `eval_sln.py`, the `JudgeVerdict` class is just a richer version of the earlier `SimpleScore` idea:
- both inherit from `BaseModel`
- both use typed fields
- both can be converted back into plain dictionaries with `.model_dump()`

Interpretation notes:
- A high overlap score does not necessarily mean the answer is correct.
- A high relevance score does not guarantee factual grounding.
- An LLM judge is useful but can still be wrong.  Therefore human review and task-specific tests are still important.
- In a larger project, you would usually evaluate many question/answer pairs rather than just one or two samples.
