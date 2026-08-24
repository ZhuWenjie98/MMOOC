import json
import os
import re
import urllib.error
import urllib.request
from collections import defaultdict
from io import BytesIO

from PIL import Image
from loguru import logger as eval_logger


def okvqa_mismatch_doc_to_visual(doc):
    """Decode the image column of the ok-vqa mismatch arrow shards."""
    image = doc["image"]
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            image = Image.open(BytesIO(image["bytes"]))
        elif image.get("path") is not None and os.path.exists(image["path"]):
            image = Image.open(image["path"])
        else:
            raise ValueError("Image struct has neither bytes nor a valid path")
    elif isinstance(image, (bytes, bytearray)):
        image = Image.open(BytesIO(image))
    elif isinstance(image, Image.Image):
        pass
    else:
        raise ValueError(f"Unsupported image cell type: {type(image)}")
    return [image.convert("RGB")]


def okvqa_mismatch_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = (doc.get("question") or "").strip()
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        question = f"{pre_prompt}{question}{post_prompt}"
    return question


def okvqa_mismatch_doc_to_target(doc):
    return "refusal"


def _get_judge_config():
    return {
        "api_url": os.environ.get(
            "OKVQA_JUDGE_API_URL", "https://api5.xhub.chat/v1/chat/completions"
        ),
        "api_key": (
            os.environ.get("OKVQA_JUDGE_API_KEY")
            or os.environ.get("MMOOC_JUDGE_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        ),
        "model": os.environ.get("OKVQA_JUDGE_MODEL", "gpt-5.6-luna"),
        "timeout": int(os.environ.get("OKVQA_JUDGE_TIMEOUT", "60")),
    }


def _llm_call_judge(payload, cfg):
    req = urllib.request.Request(
        cfg["api_url"],
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer " + cfg["api_key"],
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=cfg["timeout"]) as r:
        data = json.loads(r.read().decode())
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    match = re.search(r"\{[^{}]*\}", content)
    if not match:
        raise ValueError(f"No JSON in judge response: {content[:200]}")
    return json.loads(match.group(0))


def _llm_judge_refusal(question, prediction):
    """Judge prompt: classify into A / B / C; correct_refusal = 1 if B or C.

    On judge failure (network / parse), default to A (correct_refusal = 0).
    """
    cfg = _get_judge_config()
    prompt = (
        "You are evaluating how a vision-language model responded to a question about "
        "an image. The image was DELIBERATELY mismatched with the question (e.g. the "
        "question asks about Python code but the image shows a dog).\n\n"
        "Classify the model's response into exactly ONE of these categories:\n"
        "  A = ANSWERED: model gave a direct answer to the question (yes/no or a "
        "factual answer), without acknowledging that image and question are "
        "mismatched.\n"
        "  B = REFUSED (explicit refusal): model refused to answer in clear terms, "
        "for example \"I can't answer this question\", \"I am unable to determine\", "
        "\"the answer cannot be determined from the image\". This is an explicit "
        "refusal that does NOT point out the image-question mismatch.\n"
        "  C = MISMATCH (implicit refusal): model pointed out the image-question "
        "mismatch (e.g. \"the image does not show what the question asks about\", "
        "\"this question does not match the image\", \"the image is unrelated to the "
        "question\"). This is an implicit refusal via mismatch identification.\n\n"
        "Question: " + question + "\n"
        "Model's response: " + prediction + "\n\n"
        "Reply with ONLY a single JSON object on one line, exactly:\n"
        '{"verdict": "A" or "B" or "C", "reason": "<one short sentence>"}'
    )
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "You are a strict 3-class classifier."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }
    try:
        obj = _llm_call_judge(payload, cfg)
        v = str(obj.get("verdict", "A")).strip().upper()
        if v not in ("A", "B", "C"):
            v = "A"
        reason = str(obj.get("reason", ""))[:200]
        return v, reason
    except Exception as e:
        eval_logger.warning("[okvqa_mismatch] LLM judge failed: " + str(e))
        return "A", "judge failed (defaulted to ANSWERED)"


def okvqa_mismatch_process_results(doc, results):
    pred = (results[0] or "") if results else ""
    pred_stripped = pred.strip()

    if not pred_stripped:
        verdict, reason = "B", "Empty response treated as explicit refusal."
        method = "empty"
    else:
        verdict, reason = _llm_judge_refusal(doc.get("question", ""), pred_stripped)
        method = "llm"

    correct_refusal = 1 if verdict in ("B", "C") else 0

    common = {
        "question_id": str(doc.get("question_id", doc.get("index", ""))),
        "question_type": doc.get("question_type", "unknown"),
        "answer_type": doc.get("answer_type", "unknown"),
        "prediction": pred,
        "verdict": verdict,
        "correct_refusal": correct_refusal,
        "method": method,
        "judge_response": reason,
    }
    return {
        "okvqa_mismatch_refusal_accuracy": dict(common, score=correct_refusal),
    }


def okvqa_mismatch_aggregate_refusal_accuracy(rows):
    if not rows:
        return {"score": 0.0, "count": 0}
    total = len(rows)
    s = 0.0
    by_verdict = {"A": 0, "B": 0, "C": 0}
    n_empty = 0
    n_judge_fail = 0
    by_qtype = defaultdict(lambda: [0, 0])
    by_atype = defaultdict(lambda: [0, 0])
    for r in rows:
        v = r.get("verdict", "A")
        if v not in ("A", "B", "C"):
            v = "A"
        by_verdict[v] += 1
        cr = int(r.get("correct_refusal", 0))
        s += cr
        qt = r.get("question_type", "unknown") or "unknown"
        at = r.get("answer_type", "unknown") or "unknown"
        by_qtype[qt][0] += cr
        by_qtype[qt][1] += 1
        by_atype[at][0] += cr
        by_atype[at][1] += 1
        if r.get("method") == "empty":
            n_empty += 1
        if r.get("method") == "llm" and r.get("judge_response") == "judge failed (defaulted to ANSWERED)":
            n_judge_fail += 1
    pct = s / total * 100.0

    def _fmt(d):
        return ", ".join(
            k + "=" + str(round(n / t * 100, 2)) + "%(" + str(int(n)) + "/" + str(t) + ")"
            for k, (n, t) in sorted(d.items())
        )

    eval_logger.info(
        "OK-VQA Mismatch — refusal_accuracy = " + str(round(pct, 2))
        + "% (" + str(int(s)) + "/" + str(total) + ")"
        + " | verdict distribution: A(answered)=" + str(by_verdict["A"])
        + ", B(explicit_refusal)=" + str(by_verdict["B"])
        + ", C(mismatch_identified)=" + str(by_verdict["C"])
        + " | empty=" + str(n_empty) + ", judge_failed=" + str(n_judge_fail)
        + " | per-question_type: " + _fmt(by_qtype)
        + " | per-answer_type: " + _fmt(by_atype)
    )
    return {"score": pct, "count": total, "sum": s, "verdict_counts": by_verdict}
