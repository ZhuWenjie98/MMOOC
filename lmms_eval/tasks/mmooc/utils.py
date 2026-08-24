import re

from loguru import logger as eval_logger

from lmms_eval.tasks.mmooc.data_utils import (
    doc_to_visual,
    filter_and_shuffle_dataset,
)
from lmms_eval.tasks.mmooc.judge import (
    dual_semantic_evaluate,
    refusal_semantic_score,
    semantic_evaluate,
)

YES_NO_PROMPT_SUFFIX = " Please answer yes or no."


def _get_question_id(doc) -> str:
    """Use question_id when available, otherwise fall back to id."""
    return str(doc.get("question_id") or doc.get("id") or "unknown")


def mmooc_ooc_doc_to_visual(doc):
    """Convert an OOC row to the visual format expected by lmms-eval."""
    return doc_to_visual(doc, "OOC")


# =============================================================================
# MMOOC-OOC functions (generic, unified handler for all OOC tasks)
# =============================================================================


def mmooc_ooc_process_docs(dataset):
    """Filter and shuffle the OOC split using YAML and environment settings."""
    return filter_and_shuffle_dataset(
        dataset,
        task_name="OOC",
        yaml_name="mmooc_ooc.yaml",
    )


def mmooc_ooc_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Convert document to text prompt (generic, works for all question types).
    """
    question = doc["question"].strip()

    if lmms_eval_specific_kwargs is not None:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        if post_prompt:
            question = question.replace(YES_NO_PROMPT_SUFFIX, "")
            question = f"{question}{post_prompt}"

    return question


def mmooc_ooc_doc_to_target(doc):
    """Return the answer as-is."""
    return doc["answer"].strip()


def _extract_choice_letter(pred: str) -> str:
    """Extract letter A/B/C/D from model prediction, case-insensitive."""
    pred = pred.strip()
    for prefix in ["(A)", "(B)", "(C)", "(D)", "A)", "B)", "C)", "D)", "A.", "B.", "C.", "D."]:
        if prefix.lower() in pred.lower():
            return prefix[0].upper()
    m = re.search(r"\b([A-D])\b", pred, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if pred.lower() in ["a", "b", "c", "d"]:
        return pred.upper()
    return ""


def mmooc_ooc_mcq_process_results(doc, results):
    """
    Process MCQ predictions.
    For 'I can't answer' ground truth: check if model refuses (no A/B/C/D choice).
    For letter ground truth: compare extracted choice letter.
    """
    pred = results[0].strip()
    gt_ans = doc["answer"].strip()
    question_id = _get_question_id(doc)

    pred_choice = _extract_choice_letter(pred)
    correct = 0
    judge_response = ""

    if gt_ans.lower() in ("i can't answer", "none of the above", "n/a"):
        # Correct if model refuses to pick any option
        if not pred_choice:
            correct = 1
            judge_response = "Refusal correctly detected"
        else:
            judge_response = f"Model chose {pred_choice} but should refuse"
    else:
        # Ground truth is a letter; compare with extracted prediction
        gt_letter = gt_ans.strip().upper()
        if pred_choice and pred_choice == gt_letter:
            correct = 1
            judge_response = f"Correct: {pred_choice}"
        else:
            judge_response = f"Wrong: predicted {pred_choice!r}, expected {gt_letter!r}"

    eval_logger.debug(f"MCQ | pred={pred_choice!r} | gt={gt_ans!r} | correct={correct}")

    return {
        "mmoc_ood_mcq_accuracy": {
            "question_id": question_id,
            "prediction": pred,
            "ground_truth": gt_ans,
            "correct": float(correct),
            "judge_response": judge_response,
        }
    }


def mmooc_ooc_mcq_aggregate_accuracy(results):
    """Aggregate MCQ accuracy."""
    if not results:
        return 0.0
    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = (correct / total) * 100.0
    eval_logger.info(f"MMOOC MCQ Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


# =============================================================================
# YESNO (Yes/No Question)
# =============================================================================


def mmooc_ooc_yesno_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Convert YESNO document to text prompt."""
    question = doc["question"].strip()

    if lmms_eval_specific_kwargs is not None:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        if post_prompt:
            question = question.replace(YES_NO_PROMPT_SUFFIX, "")
            question = f"{question}{post_prompt}"

    return question


def mmooc_ooc_yesno_doc_to_target(doc):
    """Return the answer as-is: either 'I can't answer', 'yes', or 'no'."""
    return doc["answer"].strip()


def _extract_yesno(pred: str) -> str:
    """Extract yes/no from model prediction."""
    pred_lower = pred.lower().strip()
    if pred_lower in ["yes", "no"]:
        return pred_lower
    if re.search(r"\byes\b", pred_lower):
        return "yes"
    if re.search(r"\bno\b", pred_lower):
        return "no"
    if re.search(r"\by\b", pred_lower) and not re.search(r"\bno\b", pred_lower):
        return "yes"
    if re.search(r"\bn\b", pred_lower) and not re.search(r"\byes\b", pred_lower):
        return "no"
    return ""


def _is_empty_prediction(pred: str) -> bool:
    """Treat blank/whitespace-only generations as invalid predictions."""
    return not pred or not pred.strip()


def mmooc_ooc_yesno_process_results(doc, results):
    """
    Process predictions for the MMOOC yesno task.
    Routes to the appropriate handler based on question_type or answer format.
    """
    gt_ans = doc["answer"].strip()
    question_type = doc.get("question_type")
    if question_type is not None:
        question_type = question_type.lower()
    else:
        question_type = ""

    # If question_type is explicitly "yesno", always use yesno processing
    if question_type == "yesno" or "yesno" in question_type:
        # YESNO processing
        pred = results[0].strip()
        gt_ans_lower = gt_ans.lower()
        question_id = _get_question_id(doc)

        if _is_empty_prediction(pred):
            return {
                "mmoc_ood_yesno_accuracy": {
                    "question_id": question_id,
                    "prediction": pred,
                    "ground_truth": gt_ans,
                    "correct": 0.0,
                    "judge_response": "Empty prediction",
                }
            }

        pred_yesno = _extract_yesno(pred)
        correct = 0
        judge_response = ""

        if gt_ans_lower in ("i can't answer", "none of the above", "n/a"):
            # For "I can't answer" GT, check if model correctly says "No" with valid reasoning
            pred_yesno = _extract_yesno(pred)
            if pred_yesno == "no":
                # Model said "No" - check if it's correctly denying a false premise
                # by using semantic scoring to validate the explanation
                score, judge_response = refusal_semantic_score(question=doc["question"].strip(), ground_truth="no", prediction=pred)  # Evaluate against "No" since model correctly denied
                correct = float(score)
            else:
                # Model didn't say "No", use refusal semantic scoring
                score, judge_response = refusal_semantic_score(question=doc["question"].strip(), ground_truth=gt_ans, prediction=pred)
                correct = float(score)
        else:
            if pred_yesno == gt_ans_lower:
                correct = 1
                judge_response = f"Correct: {pred_yesno}"
            else:
                judge_response = f"Wrong: predicted {pred_yesno!r}, expected {gt_ans_lower!r}"

        return {
            "mmoc_ood_yesno_accuracy": {
                "question_id": question_id,
                "prediction": pred,
                "ground_truth": gt_ans,
                "correct": float(correct),
                "judge_response": judge_response,
            }
        }

    # Route based on answer format
    gt_upper = gt_ans.upper()
    if "mcq" in question_type or gt_upper in ("A", "B", "C", "D", "I CAN'T ANSWER"):
        return mmooc_ooc_mcq_process_results(doc, results)
    elif "vqa" in question_type or (gt_ans.lower() not in ("yes", "no", "i can't answer", "none of the above", "n/a") and gt_upper not in ("A", "B", "C", "D", "I CAN'T ANSWER")):
        return mmooc_ooc_vqa_process_results(doc, results)

    # Fallback for unexpected cases - treat as yesno
    pred = results[0].strip()
    question_id = _get_question_id(doc)
    return {
        "mmoc_ood_yesno_accuracy": {
            "question_id": question_id,
            "prediction": pred,
            "ground_truth": gt_ans,
            "correct": 0.0,
            "judge_response": "Unexpected question type",
        }
    }


def mmooc_ooc_yesno_aggregate_accuracy(results):
    """Aggregate OOD YESNO refusal-semantic score in [0, 1]."""
    if not results:
        return 0.0
    total = len(results)
    score_sum = sum(r["correct"] for r in results)
    mean_score = score_sum / total
    eval_logger.info(f"MMOOC YESNO Refusal Semantic Score: {mean_score:.4f} ({score_sum:.4f}/{total})")
    return mean_score


# =============================================================================
# VQA (Open-ended Visual Question Answering)
# =============================================================================


def mmooc_ooc_vqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Convert VQA document to text prompt."""
    question = doc["question"].strip()

    if lmms_eval_specific_kwargs is not None:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        if post_prompt:
            question = question.replace(YES_NO_PROMPT_SUFFIX, "")
            question = f"{question}{post_prompt}"

    return question


def mmooc_ooc_vqa_doc_to_target(doc):
    """Return the ground truth answer as-is."""
    return doc["answer"].strip()


def mmooc_ooc_vqa_process_results(doc, results):
    """
    Process VQA predictions using LLM judge for semantic evaluation.
    For 'I can't answer' ground truth: check if model refuses vs hallucinating.
    """
    pred = results[0].strip()
    gt_ans = doc["answer"].strip()
    question = doc["question"].strip()
    question_id = _get_question_id(doc)

    correct, judge_response = semantic_evaluate(question=question, answer=gt_ans, prediction=pred)

    eval_logger.debug(f"VQA | pred={pred[:50]!r} | gt={gt_ans!r} | correct={correct}")

    return {
        "mmoc_ood_vqa_accuracy": {
            "question_id": question_id,
            "prediction": pred,
            "ground_truth": gt_ans,
            "correct": float(correct),
            "judge_response": judge_response,
        }
    }


def mmooc_ooc_vqa_aggregate_accuracy(results):
    """Aggregate VQA accuracy."""
    if not results:
        return 0.0
    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = (correct / total) * 100.0
    eval_logger.info(f"MMOOC VQA Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def mmooc_ooc_process_results(doc, results):
    """
    Process predictions for the general (parent) MMOOC task.
    Uses dual semantic evaluation: answer similarity + reasoning similarity.
    """
    pred = results[0].strip()
    gt_ans = doc["answer"].strip()
    gt_reasoning = doc.get("reasoning", "").strip()
    question = doc["question"].strip()
    question_id = _get_question_id(doc)

    scores, judge_response = dual_semantic_evaluate(question=question, answer=gt_ans, reasoning=gt_reasoning, prediction=pred)

    return {
        "mmooc_ooc_answer_score": {
            "question_id": question_id,
            "prediction": pred,
            "ground_truth": gt_ans,
            "answer_score": scores["answer_score"],
            "judge_response": judge_response,
        },
        "mmooc_ooc_reasoning_score": {
            "question_id": question_id,
            "prediction": pred,
            "ground_truth_reasoning": gt_reasoning,
            "reasoning_score": scores["reasoning_score"],
            "judge_response": judge_response,
        },
        "mmooc_ooc_accuracy": {
            "question_id": question_id,
            "prediction": pred,
            "ground_truth": gt_ans,
            "ground_truth_reasoning": gt_reasoning,
            "avg_score": scores["avg_score"],
            "judge_response": judge_response,
        },
    }


def mmooc_ooc_aggregate_accuracy(results):
    """Aggregate accuracy for the general MMOOC task using average of answer_score and reasoning_score."""
    if not results:
        return 0.0
    total = len(results)
    avg_sum = sum(r.get("avg_score", 0.0) for r in results)
    overall_acc = (avg_sum / total) * 100.0
    eval_logger.info(f"MMOOC Overall Accuracy: {overall_acc:.2f}% ({avg_sum:.2f}/{total})")
    return overall_acc


def mmooc_ooc_aggregate_semantic_score(results):
    """Aggregate semantic score (reasoning consistency) for the MMOOC-OOC task."""
    if not results:
        return 0.0
    total = len(results)
    reasoning_sum = sum(r.get("reasoning_score", 0.0) for r in results)
    avg_score = (reasoning_sum / total) * 100.0
    eval_logger.info(f"MMOOC-OOC Reasoning Score: {avg_score:.2f}% ({reasoning_sum:.2f}/{total})")
    return avg_score


def mmooc_ooc_aggregate_answer_score(results):
    """Aggregate answer score for the MMOOC-OOC task."""
    if not results:
        return 0.0
    total = len(results)
    answer_sum = sum(r.get("answer_score", 0.0) for r in results)
    avg_score = (answer_sum / total) * 100.0
    eval_logger.info(f"MMOOC-OOC Answer Score: {avg_score:.2f}% ({answer_sum:.2f}/{total})")
    return avg_score


# =============================================================================
# MMOOC-IC functions (generic, unified handler for all IC tasks)
# =============================================================================


def mmooc_ic_doc_to_visual(doc):
    """Convert an IC row to the visual format expected by lmms-eval."""
    return doc_to_visual(doc, "IC")


def mmooc_ic_process_docs(dataset):
    """Filter and shuffle the IC split using YAML and environment settings."""
    return filter_and_shuffle_dataset(
        dataset,
        task_name="IC",
        yaml_name="mmooc_ic.yaml",
        legacy_ooc_type_env_names=("MMOOC_IC_OOCTYPE",),
    )


def mmooc_ic_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if lmms_eval_specific_kwargs is not None:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        if post_prompt:
            question = question.replace(YES_NO_PROMPT_SUFFIX, "")
            question = f"{question}{post_prompt}"
    return question


def mmooc_ic_doc_to_target(doc):
    return doc["answer"].strip()


def mmooc_ic_process_results(doc, results):
    """
    Process predictions for the MMOOC-IC task.
    Uses dual semantic evaluation: answer similarity + reasoning consistency.
    """
    pred = results[0].strip()
    gt_ans = doc["answer"].strip()
    gt_reasoning = doc.get("reasoning", "").strip()
    question = doc["question"].strip()
    question_id = _get_question_id(doc)

    scores, judge_response = dual_semantic_evaluate(question=question, answer=gt_ans, reasoning=gt_reasoning, prediction=pred)

    eval_logger.debug(f"IC | answer_score={scores['answer_score']} | " f"reasoning_score={scores['reasoning_score']} | avg_score={scores['avg_score']}")

    return {
        "mmooc_ic_answer_score": {
            "question_id": question_id,
            "prediction": pred,
            "ground_truth": gt_ans,
            "answer_score": scores["answer_score"],
            "judge_response": judge_response,
        },
        "mmooc_ic_reasoning_score": {
            "question_id": question_id,
            "prediction": pred,
            "ground_truth_reasoning": gt_reasoning,
            "reasoning_score": scores["reasoning_score"],
            "judge_response": judge_response,
        },
        "mmooc_ic_accuracy": {
            "question_id": question_id,
            "prediction": pred,
            "ground_truth": gt_ans,
            "ground_truth_reasoning": gt_reasoning,
            "avg_score": scores["avg_score"],
            "judge_response": judge_response,
        },
    }


def mmooc_ic_aggregate_accuracy(results):
    """Aggregate accuracy for the MMOOC-IC task using average of answer_score and reasoning_score."""
    if not results:
        return 0.0
    total = len(results)
    answer_sum = sum(r.get("answer_score", 0.0) for r in results)
    reasoning_sum = sum(r.get("reasoning_score", 0.0) for r in results)
    avg_sum = sum(r.get("avg_score", 0.0) for r in results)

    answer_acc = (answer_sum / total) * 100.0
    reasoning_acc = (reasoning_sum / total) * 100.0
    overall_acc = (avg_sum / total) * 100.0

    eval_logger.info(f"MMOOC-IC Answer Similarity: {answer_acc:.2f}% ({answer_sum:.2f}/{total})")
    eval_logger.info(f"MMOOC-IC Reasoning Consistency: {reasoning_acc:.2f}% ({reasoning_sum:.2f}/{total})")
    eval_logger.info(f"MMOOC-IC Overall Accuracy: {overall_acc:.2f}% ({avg_sum:.2f}/{total})")
    return overall_acc


def mmooc_ic_aggregate_semantic_score(results):
    """Aggregate semantic score (reasoning consistency) for the MMOOC-IC task."""
    if not results:
        return 0.0
    total = len(results)
    reasoning_sum = sum(r.get("reasoning_score", 0.0) for r in results)
    avg_score = (reasoning_sum / total) * 100.0
    eval_logger.info(f"MMOOC-IC Semantic Score (Reasoning Consistency): {avg_score:.2f}% ({reasoning_sum:.2f}/{total})")
    return avg_score


def mmooc_ic_aggregate_reasoning_score(results):
    """Aggregate reasoning score for the MMOOC-IC task."""
    if not results:
        return 0.0
    total = len(results)
    reasoning_sum = sum(r.get("reasoning_score", 0.0) for r in results)
    avg_score = (reasoning_sum / total) * 100.0
    eval_logger.info(f"MMOOC-IC Reasoning Score: {avg_score:.2f}% ({reasoning_sum:.2f}/{total})")
    return avg_score


def mmooc_ic_aggregate_answer_score(results):
    """Aggregate answer score for the MMOOC-IC task."""
    if not results:
        return 0.0
    total = len(results)
    answer_sum = sum(r.get("answer_score", 0.0) for r in results)
    avg_score = (answer_sum / total) * 100.0
    eval_logger.info(f"MMOOC-IC Answer Score: {avg_score:.2f}% ({answer_sum:.2f}/{total})")
    return avg_score


def mmooc_ic_aggregate_avg_score(results):
    """Aggregate average of answer_score and reasoning_score for the MMOOC-IC task."""
    if not results:
        return 0.0
    total = len(results)
    avg_sum = sum(r.get("avg_score", 0.0) for r in results)
    avg_score = (avg_sum / total) * 100.0
    eval_logger.info(f"MMOOC-IC Avg Score: {avg_score:.2f}% ({avg_sum:.2f}/{total})")
    return avg_score
