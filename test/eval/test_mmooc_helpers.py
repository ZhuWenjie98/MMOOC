"""Regression tests for the modular MMOOC task helpers."""

import importlib.util
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

MMOOC_MODULE_DIR = Path(__file__).parents[2] / "lmms_eval" / "tasks" / "mmooc"


def _load_module(module_name):
    """Load a task helper directly to avoid task-registry side effects."""
    module_path = MMOOC_MODULE_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"mmooc_{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data_utils = _load_module("data_utils")
judge = _load_module("judge")
filter_and_shuffle_dataset = data_utils.filter_and_shuffle_dataset
load_filter_config = data_utils.load_filter_config
load_rgb_image = data_utils.load_rgb_image
resolve_shuffle_seed = data_utils.resolve_shuffle_seed
extract_response_text = judge._extract_response_text
parse_judge_json = judge.parse_judge_json


class FakeDataset(list):
    """Minimal Hugging Face Dataset stand-in for filter tests."""

    def __init__(self, rows, shuffle_seed=None):
        super().__init__(rows)
        self.shuffle_seed = shuffle_seed

    def filter(self, predicate):
        return FakeDataset([row for row in self if predicate(row)])

    def shuffle(self, seed):
        return FakeDataset(self, shuffle_seed=seed)


def test_load_filter_config_supports_function_tags():
    config = load_filter_config("mmooc_ooc.yaml")

    assert config == {
        "category": "",
        "question_type": "",
        "ooc_type": "",
    }


def test_filter_dataset_uses_normalized_ic_environment_name(monkeypatch):
    dataset = FakeDataset(
        [
            {"category": "math", "question_type": "mcq", "ooc_type": "missing"},
            {"category": "science", "question_type": "vqa", "ooc_type": "visual"},
        ]
    )
    monkeypatch.setenv("MMOOC_IC_CATEGORY", "math")
    monkeypatch.setenv("MMOOC_IC_OOC_TYPE", "missing")
    monkeypatch.setenv("MMOOC_IC_SHUFFLE_SEED", "42")

    result = filter_and_shuffle_dataset(
        dataset,
        task_name="IC",
        yaml_name="mmooc_ic.yaml",
        legacy_ooc_type_env_names=("MMOOC_IC_OOCTYPE",),
    )

    assert list(result) == [dataset[0]]
    assert result.shuffle_seed == 42


def test_legacy_ic_ooc_type_environment_name_remains_supported(monkeypatch):
    dataset = FakeDataset(
        [
            {"category": "math", "question_type": "mcq", "ooc_type": "missing"},
            {"category": "math", "question_type": "mcq", "ooc_type": "visual"},
        ]
    )
    monkeypatch.setenv("MMOOC_IC_OOCTYPE", "visual")
    monkeypatch.setenv("MMOOC_IC_SHUFFLE_SEED", "7")

    result = filter_and_shuffle_dataset(
        dataset,
        task_name="IC",
        yaml_name="mmooc_ic.yaml",
        legacy_ooc_type_env_names=("MMOOC_IC_OOCTYPE",),
    )

    assert list(result) == [dataset[1]]


def test_resolve_shuffle_seed_reports_invalid_value(monkeypatch):
    monkeypatch.setenv("MMOOC_TEST_SHUFFLE_SEED", "not-an-integer")

    with pytest.raises(ValueError, match="must be an integer"):
        resolve_shuffle_seed("MMOOC_TEST_SHUFFLE_SEED")


def test_load_rgb_image_accepts_bytes():
    image_buffer = BytesIO()
    Image.new("L", (2, 2), color=128).save(image_buffer, format="PNG")

    image = load_rgb_image(image_buffer.getvalue())

    assert image.mode == "RGB"
    assert image.size == (2, 2)


@pytest.mark.parametrize(
    ("raw_response", "expected"),
    [
        ('{"answer_score": 1, "reasoning_score": 0.75}', 0.75),
        (
            '```json\n{"answer_score": 1, "reasoning_score": 0.5}\n```',
            0.5,
        ),
        ("answer_score: 1, reasoning_score: 0.25", 0.25),
    ],
)
def test_parse_judge_json_handles_common_formats(raw_response, expected):
    parsed = parse_judge_json(raw_response)

    assert parsed is not None
    assert parsed["reasoning_score"] == expected


def test_extract_response_text_handles_content_blocks():
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second"},
                    ]
                }
            }
        ]
    }

    assert extract_response_text(response) == "first\nsecond"
