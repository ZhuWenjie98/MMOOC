import json
import os
import re
import urllib.error
import urllib.request
from collections import defaultdict
from io import BytesIO

from loguru import logger as eval_logger
from PIL import Image

STRONG_REFUSAL_PATTERNS = [
    r"\bi can\s*['’`]?t\b",
    r"\bi cannot\b",
    r"\bi can\s*not\b",
    r"\bi['’]?m\s+unable\b",
    r"\bi\s+am\s+unable\b",
    r"\bcan['’]?t\s+(answer|determine|identify|see|tell|recognize)\b",
    r"\bcannot\s+(answer|determine|identify|see|tell|recognize)\b",
    r"\bnot\s+(visible|present|shown|depicted|available|in\s+the\s+image)\b",
    r"\bno\s+such\b",
    r"\bunrelated\b",
    r"\birrelevant\b",
    r"\bmismatch(ed)?\b",
    r"\bdoesn['’]?t\s+(match|show|depict|relate|correspond)\b",
    r"\bdoes\s+not\s+(match|show|depict|relate|correspond)\b",
    r"\b(image|photo|picture)\s+(does\s+not|doesn['’]?t)\b",
    r"\b(question|query)\s+(does\s+not|doesn['’]?t)\s+(match|relate|correspond)\b",
    r"\bcannot\s+be\s+(answered|determined)\b",
    r"\bno\s+relevant\b",
    r"\bunable\s+to\s+(answer|determine|identify|see|tell)\b",
]

WEAK_REFUSAL_PATTERNS = [
    r"\bnot\s+in\s+the\b",
    r"\babsence\b",
    r"\bmissing\b",
    r"\bno\s+evidence\b",
    r"\bquestion\s+is\s+not\b",
]

CHINESE_REFUSAL_PATTERNS = [
    r"无法",
    r"不能",
    r"不可见",
    r"不匹配",
    r"无关",
    r"看不到",
    r"没有",
    r"缺席",
]

STRONG_REGEX = re.compile("|".join(STRONG_REFUSAL_PATTERNS), re.IGNORECASE)
WEAK_REGEX = re.compile("|".join(WEAK_REFUSAL_PATTERNS + CHINESE_REFUSAL_PATTERNS), re.IGNORECASE)


def mme_mismatch_process_docs(dataset):
    return dataset


def mme_mismatch_doc_to_visual(doc):
    image = doc["image"]
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            image = Image.open(BytesIO(image["bytes"]))
        elif image.get("path") is not None:
            image = Image.open(image["path"])
    elif isinstance(image, (bytes, bytearray)):
        image = Image.open(BytesIO(image))
    return [image.convert("RGB")]


def mme_mismatch_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = (doc.get("question") or "").strip()
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        question = f"{pre_prompt}{question}{post_prompt}"
    return question


def mme_mismatch_doc_to_target(doc):
    return "refusal"


def _get_judge_config():
    return {
        "api_url": os.environ.get("MME_JUDGE_API_URL", "https://api5.xhub.chat/v1/chat/completions"),
        "api_key": (os.environ.get("MME_JUDGE_API_KEY") or os.environ.get("MMOOC_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")),
        "model": os.environ.get("MME_JUDGE_MODEL", "gpt-5.6-luna"),
        "timeout": int(os.environ.get("MME_JUDGE_TIMEOUT", "60")),
    }


def _llm_judge_refusal(question, prediction):
    cfg = _get_judge_config()
    prompt = (
        "You are evaluating how a vision-language model responded to a question about "
        "an image. The image was DELIBERATELY mismatched with the question (e.g. the "
        "question asks about Python code but the image shows a dog).\n\n"
        "Classify the model's response into exactly ONE of these three categories:\n"
        "  0 = DIRECT_ANSWER: model answered the question directly (yes/no or a factual "
        "answer), without acknowledging that the image and question are mismatched.\n"
        '  1 = REFUSED: model refused to answer (e.g. "I cannot answer based on the '
        'image", "I am unable to determine", "the answer cannot be determined '
        'from the image").\n'
        "  2 = MISMATCH_IDENTIFIED: model pointed out the image-question mismatch (e.g. "
        '"the image does not show what the question asks about", "this question '
        'does not match the image", "the image is unrelated to the question").\n\n'
        "Question: " + question + "\n"
        "Model's response: " + prediction + "\n\n"
        "Reply with ONLY a single JSON object on one line, exactly:\n"
        '{"verdict": 0 or 1 or 2, "reason": "<one short sentence>"}'
    )
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "You are a strict 3-class classifier."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 100,
    }
    req = urllib.request.Request(
        cfg["api_url"],
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer " + cfg["api_key"],
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg["timeout"]) as r:
            data = json.loads(r.read().decode())
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        match = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", content)
        if match:
            obj = json.loads(match.group(0))
            v = int(obj.get("verdict", 0))
            return v if v in (0, 1, 2) else 0
        return 0
    except Exception as e:
        eval_logger.warning("[mme_mismatch] LLM judge failed: " + str(e))
        return 0


def mme_mismatch_process_results(doc, results):
    pred = (results[0] or "") if results else ""
    pred_stripped = pred.strip()

    if not pred_stripped:
        verdict = 1  # empty -> treat as refusal
        judge_reason = "Empty response treated as refusal."
        method = "empty"
    elif STRONG_REGEX.search(pred_stripped):
        verdict = 1  # strong refusal keyword
        judge_reason = "Strong refusal keyword matched: " + pred_stripped[:120]
        method = "keyword_strong"
    else:
        question = doc.get("question", "")
        verdict = _llm_judge_refusal(question, pred_stripped)
        label = {0: "direct_answer", 1: "refused", 2: "mismatch_identified"}.get(verdict, "direct_answer")
        judge_reason = "LLM judge -> " + label + " (verdict=" + str(verdict) + ")"
        method = "llm"

    refused = 1 if verdict in (1, 2) else 0

    return {
        "mme_mismatch_refusal_accuracy": {
            "question_id": str(doc.get("question_id", doc.get("index", ""))),
            "category": doc.get("category", "unknown"),
            "prediction": pred,
            "refused": refused,
            "verdict": verdict,
            "method": method,
            "judge_response": judge_reason,
        }
    }


def mme_mismatch_aggregate_refusal_accuracy(results):
    if not results:
        return 0.0
    total = len(results)
    refused = sum(r["refused"] for r in results)
    accuracy = (refused / total) * 100.0
    by_cat = defaultdict(lambda: [0, 0])
    for r in results:
        cat = r.get("category", "unknown")
        by_cat[cat][0] += int(r["refused"])
        by_cat[cat][1] += 1
    cat_str = ", ".join(c + "=" + str(round(n / t * 100, 2)) + "%(" + str(int(n)) + "/" + str(t) + ")" for c, (n, t) in sorted(by_cat.items()))
    eval_logger.info("MME Mismatch Refusal Accuracy: " + str(round(accuracy, 2)) + "% (" + str(int(refused)) + "/" + str(total) + ") | per-category: " + cat_str)
    return accuracy
