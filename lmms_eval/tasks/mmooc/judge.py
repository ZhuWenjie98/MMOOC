"""LLM-as-a-judge support for MMOOC scoring.

Credentials and endpoint settings are intentionally read from environment
variables so experiment configuration never needs to be committed to source.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx
from loguru import logger as eval_logger

from lmms_eval.llm_judge.base import ServerInterface
from lmms_eval.llm_judge.protocol import Request, Response, ServerConfig

DEFAULT_API_URL = "https://api5.xhub.chat/v1/chat/completions"
DEFAULT_MODEL = "gpt-5.6-luna"
REFUSAL_ANSWERS = {"i can't answer", "none of the above", "n/a"}
VALID_REASONING_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)

_judge: "OpenAICompatibleJudge | None" = None


def _environment_value(primary_name: str, fallback_name: str = "", default: str = "") -> str:
    """Read a namespaced setting, optionally falling back to a standard name."""
    value = os.getenv(primary_name, "").strip()
    if not value and fallback_name:
        value = os.getenv(fallback_name, "").strip()
    return value or default


class OpenAICompatibleJudge(ServerInterface):
    """Synchronous client for an OpenAI-compatible chat-completions API."""

    def __init__(self, config: ServerConfig, api_url: str, api_key: str):
        super().__init__(config)
        self.api_url = api_url
        self.api_key = api_key

    def is_available(self) -> bool:
        return bool(self.api_url and self.api_key)

    def evaluate(self, request: Request) -> Response:
        if not self.is_available():
            raise RuntimeError("MMOOC judge is not configured. Set MMOOC_JUDGE_API_KEY " "(or OPENAI_API_KEY), and optionally MMOOC_JUDGE_API_URL.")

        payload = {
            "model": self.config.model_name,
            "messages": self.prepare_messages(request),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.config.num_retries + 1):
            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    api_response = client.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                    )
                    api_response.raise_for_status()
                    raw_response = api_response.json()
                return Response(
                    content=_extract_response_text(raw_response),
                    model_used=self.config.model_name,
                    raw_response=raw_response,
                )
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                last_error = exc
                if attempt < self.config.num_retries:
                    eval_logger.warning(
                        "MMOOC judge request failed ({}/{}): {}",
                        attempt,
                        self.config.num_retries,
                        exc,
                    )
                    time.sleep(self.config.retry_delay)

        raise RuntimeError(f"MMOOC judge request failed: {last_error}") from last_error


def _extract_response_text(response: dict[str, Any]) -> str:
    """Normalize text from common OpenAI-compatible response shapes."""
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [block.get("text") or block.get("content") for block in content if isinstance(block, dict)]
        return "\n".join(part for part in text_parts if isinstance(part, str)).strip()
    for fallback_key in ("output_text", "text"):
        if isinstance(response.get(fallback_key), str):
            return response[fallback_key].strip()
    return ""


def get_judge() -> OpenAICompatibleJudge:
    """Build and cache the judge configured by environment variables."""
    global _judge
    if _judge is None:
        model_name = _environment_value(
            "MMOOC_JUDGE_MODEL",
            "OPENAI_MODEL",
            DEFAULT_MODEL,
        )
        config = ServerConfig(
            model_name=model_name,
            temperature=0.0,
            max_tokens=256,
            timeout=int(os.getenv("MMOOC_JUDGE_TIMEOUT", "60")),
            num_retries=int(os.getenv("MMOOC_JUDGE_RETRIES", "3")),
            retry_delay=float(os.getenv("MMOOC_JUDGE_RETRY_DELAY", "5")),
        )
        _judge = OpenAICompatibleJudge(
            config=config,
            api_url=_environment_value(
                "MMOOC_JUDGE_API_URL",
                "OPENAI_API_BASE",
                DEFAULT_API_URL,
            ),
            api_key=_environment_value("MMOOC_JUDGE_API_KEY", "OPENAI_API_KEY"),
        )
        eval_logger.info(
            "MMOOC judge initialized: model={}, url={}",
            model_name,
            _judge.api_url,
        )
    return _judge


def parse_judge_json(raw_response: str) -> dict[str, Any] | None:
    """Extract a JSON object from plain or fenced model output."""
    if not raw_response:
        return None
    candidates = [raw_response]
    candidates.extend(
        match.group(1)
        for match in re.finditer(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            raw_response,
            re.DOTALL | re.IGNORECASE,
        )
    )
    object_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    answer_match = re.search(
        r"answer_score[\"']?\s*[:=]\s*([\d.]+)",
        raw_response,
        re.IGNORECASE,
    )
    reasoning_match = re.search(
        r"reasoning_score[\"']?\s*[:=]\s*([\d.]+)",
        raw_response,
        re.IGNORECASE,
    )
    if answer_match and reasoning_match:
        return {
            "answer_score": float(answer_match.group(1)),
            "reasoning_score": float(reasoning_match.group(1)),
        }
    return None


def _judge_request(prompt: str, question: str, answer: str, prediction: str) -> str:
    request = Request(
        messages=[{"role": "user", "content": prompt}],
        question=question,
        answer=answer,
        prediction=prediction,
        config=get_judge().config,
    )
    return (get_judge().evaluate(request).content or "").strip()


def semantic_evaluate(
    question: str,
    answer: str,
    prediction: str,
) -> tuple[int, str]:
    """Evaluate binary semantic correctness."""
    result = get_judge().evaluate_binary(
        question=question,
        answer=answer,
        prediction=prediction,
        output_format="0/1",
    )
    return int(result["result"] == 1), result["raw_response"]


def dual_semantic_evaluate(
    question: str,
    answer: str,
    reasoning: str,
    prediction: str,
) -> tuple[dict[str, float], str]:
    """Score answer correctness and reasoning consistency."""
    if not prediction or not prediction.strip():
        return _empty_scores(), "Empty prediction"

    answer_criterion = (
        "The reference is a refusal. Award answer_score=1 when the response " "refuses safely or correctly identifies the missing/false premise; " "otherwise award 0."
        if answer.lower() in REFUSAL_ANSWERS
        else "Award answer_score=1 only when the response is semantically correct; otherwise award 0."
    )
    prompt = f"""You are evaluating a vision-language QA response.

{answer_criterion}
Score reasoning consistency against the reference reasoning using exactly one
of: 0, 0.25, 0.5, 0.75, 1. Higher means stronger semantic agreement.

Return JSON only:
{{"answer_score": 0 or 1, "reasoning_score": one allowed value, "reason": "brief explanation"}}

Question:
{question}

Reference answer:
{answer}

Reference reasoning:
{reasoning}

Model response:
{prediction}"""

    try:
        raw_response = _judge_request(prompt, question, answer, prediction)
        parsed = parse_judge_json(raw_response)
        if parsed is None:
            eval_logger.warning("Unable to parse MMOOC judge output: {}", raw_response[:300])
            return _empty_scores(), f"Parse failed, raw: {raw_response[:200]}"

        answer_score = 1.0 if float(parsed.get("answer_score", 0.0)) >= 0.5 else 0.0
        raw_reasoning_score = float(parsed.get("reasoning_score", 0.0))
        reasoning_score = min(
            VALID_REASONING_SCORES,
            key=lambda score: abs(score - raw_reasoning_score),
        )
        return {
            "answer_score": answer_score,
            "reasoning_score": reasoning_score,
            "avg_score": (answer_score + reasoning_score) / 2.0,
        }, str(parsed.get("reason", "")).strip()
    except Exception as exc:
        eval_logger.error("MMOOC judge evaluation failed: {}", exc)
        return _empty_scores(), f"Judge error: {exc}"


def _empty_scores() -> dict[str, float]:
    return {
        "answer_score": 0.0,
        "reasoning_score": 0.0,
        "avg_score": 0.0,
    }


def refusal_semantic_score(
    question: str,
    ground_truth: str,
    prediction: str,
) -> tuple[float, str]:
    """Score semantic agreement with a refusal answer on a continuous scale."""
    if not prediction or not prediction.strip():
        return 0.0, "Empty prediction"

    prompt = f"""Evaluate the model response against the reference answer.
Return JSON only: {{"score": <number from 0 to 1>, "reason": "brief explanation"}}.
Use fine-grained partial scores when appropriate.

Question:
{question}

Reference answer:
{ground_truth}

Model response:
{prediction}"""
    try:
        raw_response = _judge_request(prompt, question, ground_truth, prediction)
        parsed = parse_judge_json(raw_response)
        if parsed is not None and "score" in parsed:
            score = float(parsed["score"])
            reason = str(parsed.get("reason", "")).strip()
        else:
            score_match = re.search(r"-?\d+(?:\.\d+)?", raw_response)
            if score_match is None:
                return 0.0, f"Judge parse failure: {raw_response}"
            score = float(score_match.group(0))
            reason = "Parsed score from non-JSON judge output"
        return max(0.0, min(1.0, score)), reason or "No reason provided"
    except Exception as exc:
        eval_logger.error("MMOOC refusal judge failed: {}", exc)
        return 0.0, f"Judge error: {exc}"
