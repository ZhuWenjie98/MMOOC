"""Shared dataset and image adapters for MMOOC tasks."""

from __future__ import annotations

import base64
import binascii
import io
import os
from pathlib import Path
from typing import Any, Iterable

import yaml
from loguru import logger as eval_logger
from PIL import Image

FILTER_KEYS = ("category", "question_type", "ooc_type")
QUESTION_ONLY_VALUES = {"1", "true", "yes", "question_only"}


class _TaskConfigLoader(yaml.SafeLoader):
    """YAML loader that treats lmms-eval's ``!function`` values as strings."""


_TaskConfigLoader.add_constructor(
    "!function",
    lambda loader, node: loader.construct_scalar(node),
)


def resolve_shuffle_seed(env_name: str) -> tuple[int, str]:
    """Return a user-provided seed or generate a per-run random seed."""
    raw_seed = os.getenv(env_name, "").strip()
    if raw_seed:
        try:
            return int(raw_seed), f"{env_name} (user)"
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer, got {raw_seed!r}") from exc
    return int.from_bytes(os.urandom(4), "big"), "auto-random"


def is_question_only_mode(task_name: str) -> bool:
    """Whether a task should receive a neutral image placeholder."""
    value = os.getenv(f"MMOOC_{task_name}_INPUT_MODE", "").strip().lower()
    return value in QUESTION_ONLY_VALUES


def blank_image() -> Image.Image:
    """Create the neutral image used by question-only evaluations."""
    return Image.new("RGB", (224, 224), color="white")


def load_rgb_image(image: Any) -> Image.Image:
    """Load a local, remote, base64, bytes, or PIL image as RGB."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if not isinstance(image, str):
        raise TypeError(f"Unsupported MMOOC image type: {type(image).__name__}")

    if image.startswith(("http://", "https://")):
        import requests

        response = requests.get(image, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    encoded = image.partition(",")[2] if image.startswith("data:image/") else image
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(image_bytes)) as decoded_image:
            return decoded_image.convert("RGB")
    except (binascii.Error, ValueError, OSError):
        with Image.open(image) as local_image:
            return local_image.convert("RGB")


def doc_to_visual(doc: dict[str, Any], task_name: str) -> list[Image.Image]:
    """Convert a dataset row into the visual list expected by lmms-eval."""
    if is_question_only_mode(task_name):
        return [blank_image()]
    if "image" not in doc:
        raise KeyError("MMOOC sample is missing the required 'image' field")
    return [load_rgb_image(doc["image"])]


def load_filter_config(yaml_name: str) -> dict[str, str]:
    """Read MMOOC filters from a task YAML's default task arguments."""
    yaml_path = Path(__file__).with_name(yaml_name)
    if not yaml_path.exists():
        raise FileNotFoundError(f"MMOOC task config not found: {yaml_path}")

    with yaml_path.open(encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=_TaskConfigLoader) or {}

    defaults = config.get("lmms_eval_specific_kwargs", {}).get("default", {})
    return {key: str(defaults.get(key) or "").strip() for key in FILTER_KEYS}


def _first_environment_value(names: Iterable[str], default: str) -> str:
    for name in names:
        if name in os.environ:
            return os.environ[name].strip()
    return default


def filter_and_shuffle_dataset(
    dataset: Any,
    *,
    task_name: str,
    yaml_name: str,
    legacy_ooc_type_env_names: tuple[str, ...] = (),
) -> Any:
    """Apply configured filters and shuffle an MMOOC dataset."""
    filters = load_filter_config(yaml_name)
    env_prefix = f"MMOOC_{task_name}"
    filters["category"] = os.getenv(f"{env_prefix}_CATEGORY", filters["category"]).strip()
    filters["question_type"] = os.getenv(
        f"{env_prefix}_QUESTION_TYPE",
        filters["question_type"],
    ).strip()
    filters["ooc_type"] = _first_environment_value(
        (f"{env_prefix}_OOC_TYPE", *legacy_ooc_type_env_names),
        filters["ooc_type"],
    )

    eval_logger.info(
        "MMOOC-{} filters: samples={}, category={!r}, question_type={!r}, ooc_type={!r}",
        task_name,
        len(dataset),
        filters["category"],
        filters["question_type"],
        filters["ooc_type"],
    )

    def matches_filters(doc: dict[str, Any]) -> bool:
        category_matches = not filters["category"] or doc.get("category", "") == filters["category"]
        question_type_matches = not filters["question_type"] or filters["question_type"].lower() in str(doc.get("question_type", "")).lower()
        ooc_type_matches = not filters["ooc_type"] or doc.get("ooc_type", "") == filters["ooc_type"]
        return category_matches and question_type_matches and ooc_type_matches

    filtered_dataset = dataset if not any(filters.values()) else dataset.filter(matches_filters)
    shuffle_seed, seed_source = resolve_shuffle_seed(f"{env_prefix}_SHUFFLE_SEED")
    eval_logger.info(
        "MMOOC-{} selected {} of {} samples; shuffle seed={} ({})",
        task_name,
        len(filtered_dataset),
        len(dataset),
        shuffle_seed,
        seed_source,
    )
    return filtered_dataset.shuffle(seed=shuffle_seed)
