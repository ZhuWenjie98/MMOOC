# MMOOC evaluation tasks

This directory contains the in-context (`mmooc_ic`) and out-of-context
(`mmooc_ooc`) lmms-eval tasks.

## Module responsibilities

- `utils.py`: lmms-eval task callbacks and metric aggregation. Public callback
  names are kept stable because the YAML files reference them.
- `data_utils.py`: image decoding, question-only mode, dataset filtering, and
  shuffle configuration shared by IC and OOC.
- `judge.py`: OpenAI-compatible judge client, response parsing, and semantic
  scoring.

## Judge configuration

No credentials are stored in source code. Configure the judge with:

```bash
export MMOOC_JUDGE_API_KEY="..."
export MMOOC_JUDGE_API_URL="https://example.com/v1/chat/completions"
export MMOOC_JUDGE_MODEL="model-name"
```

`OPENAI_API_KEY`, `OPENAI_API_BASE`, and `OPENAI_MODEL` are supported as
fallbacks. Optional tuning variables are `MMOOC_JUDGE_TIMEOUT`,
`MMOOC_JUDGE_RETRIES`, and `MMOOC_JUDGE_RETRY_DELAY`.

## Dataset filtering

Filters can be set in each task YAML or overridden at runtime:

- `MMOOC_OOC_CATEGORY`, `MMOOC_OOC_QUESTION_TYPE`,
  `MMOOC_OOC_OOC_TYPE`
- `MMOOC_IC_CATEGORY`, `MMOOC_IC_QUESTION_TYPE`,
  `MMOOC_IC_OOC_TYPE`

The legacy `MMOOC_IC_OOCTYPE` spelling remains supported for existing scripts.
Set `MMOOC_<IC|OOC>_SHUFFLE_SEED` for reproducible ordering, or
`MMOOC_<IC|OOC>_INPUT_MODE=question_only` to replace images with a neutral
placeholder.
