import re
from collections import Counter
from io import BytesIO

from PIL import Image

from loguru import logger as eval_logger


REFUSAL_OPTION = "E"
ANSWER_PATTERN = re.compile(
    r"\b([A-E])\b",
    re.IGNORECASE,
)
PREFIX_PATTERN = re.compile(
    r"\b(?:answer|ans|option|choice|select|is|=>)\s*[:\-]?\s*([A-E])\b",
    re.IGNORECASE,
)


def mmstar_mismacth_process_docs(dataset):
    return dataset


def mmstar_mismacth_doc_to_visual(doc):
    image = doc["image"]
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            image = Image.open(BytesIO(image["bytes"]))
        elif image.get("path") is not None:
            image = Image.open(image["path"])
    return [image.convert("RGB")]


def mmstar_mismacth_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        question = f"{pre_prompt}{question}{post_prompt}"
    return question


def mmstar_mismacth_doc_to_target(doc):
    return REFUSAL_OPTION


def mmstar_mismacth_extract_choice(pred: str) -> str:
    if not pred:
        return ""
    text = pred.strip()
    prefix_match = PREFIX_PATTERN.search(text)
    if prefix_match:
        return prefix_match.group(1).upper()
    candidates = [m.group(1).upper() for m in ANSWER_PATTERN.finditer(text)]
    if not candidates:
        return ""
    counts = Counter(candidates)
    return counts.most_common(1)[0][0]


def mmstar_mismacth_process_results(doc, results):
    pred = results[0]
    choice = mmstar_mismacth_extract_choice(pred)
    correct = 1.0 if choice == REFUSAL_OPTION else 0.0
    if choice == REFUSAL_OPTION:
        judge_response = "Model selected the refusal option E."
    elif choice:
        judge_response = f"Model selected non-refusal option {choice}; expected refusal option E."
    else:
        judge_response = "No valid A-E choice detected; treating as non-refusal."
    return {
        "mmstar_mismacth_refusal_accuracy": {
            "question_id": str(doc.get("index", "")),
            "prediction": pred,
            "extracted_choice": choice,
            "ground_truth": REFUSAL_OPTION,
            "original_answer": doc.get("original_answer", ""),
            "correct": correct,
            "judge_response": judge_response,
        }
    }


def mmstar_mismacth_aggregate_refusal_accuracy(results):
    if not results:
        return 0.0
    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = (correct / total) * 100.0
    eval_logger.info(
        f"MMStar Mismatch Refusal Accuracy: {accuracy:.2f}% ({correct:.0f}/{total})"
    )
    return accuracy
