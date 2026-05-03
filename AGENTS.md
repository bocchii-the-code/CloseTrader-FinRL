# AGENTS.md (CloseTrader-FinRL)

# Karpathy Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Fast orientation (what runs what)

- Library entrypoint: `uv run python -m finrl --mode=train|test|trade` (implemented in `finrl/main.py`).
- Code layout:
  - `finrl/meta/`: environments + preprocessors + data processors (data download/feature engineering lives here).
  - `finrl/agents/`: DRL library adapters (ElegantRL / RLlib / Stable-Baselines3).
  - `finrl/applications/` + `examples/`: runnable scripts/notebooks (most “end-to-end” demos are here).

## Documentation structure

- README.md — CloseTrader-FinRL project README (streamlined: quick start, tutorial, key features).
- README_FINRL.md — Original upstream FinRL README (full ecosystem, data sources, publications, news).

## Install (don’t guess which tool)

- Repo is now uv-friendly (PEP 621/735 in `pyproject.toml`), but `setup.py`/`requirements.txt` still exist for legacy installs.
  - Preferred: `uv sync` then `uv run <cmd>`.
  - Legacy: `pip install -e .`.
- Supported Python is `>=3.8,<3.13` (`pyproject.toml`).
- Repo pins Python via `.python-version` (use Python 3.11).
- `TA-lib` is a dependency and may require a non-pip install on some platforms (note in `requirements.txt`: `conda install -c conda-forge ta-lib`).

### Optional deps (extras)

- ElegantRL adapter is optional: `uv sync --extra elegantrl`
  - Note: `elegantrl` depends on `gym[box2d]` which can pull native-build dependencies (e.g., `pygame`, `box2d-py`) on some platforms.
- Box2D environments are optional: `uv sync --extra box2d`

## Run the canonical example pipeline

- `uv run python examples/FinRL_StockTrading_2026_1_data.py`
- `uv run python examples/FinRL_StockTrading_2026_2_train.py`
- `uv run python examples/FinRL_StockTrading_2026_3_Backtest.py`

## Tests (non-obvious gotchas)

- Tests live in `unit_tests/` (not `tests/`).
- Minimal local run: `uv run pytest unit_tests -v`.
- Some tests hit external services:
  - Yahoo downloader tests require internet access (and may fail in restricted regions).
  - `unit_tests/downloaders/test_alpaca_downloader.py` uses placeholder creds (`API_KEY = "???"`) and will fail unless you wire in real Alpaca credentials.
    - To run everything else: `uv run pytest unit_tests -k "not alpaca_downloader" -v`.

## CI parity (how CI actually runs tests)

- GitHub Actions builds a Docker image and runs: `python3 -m pytest . -v` inside it.
  - Local equivalent:
    - `bash docker/bin/build_container.sh` (builds image `finrl`)
    - `bash docker/bin/test.sh`
  - These scripts are bash (`#!/bin/bash`); on Windows use WSL/Git Bash.

## Secrets / local config

- `finrl/config_private.py` is the expected place for Alpaca keys for `--mode=trade` (see `finrl/main.py`). Keep real keys out of git.

## Repo .gitignore gotcha

- Common outputs are ignored repo-wide (`*.csv`, `*.zip`, `*.png`), so example data/models/plots won’t appear in `git status`.

## Formatting/lint (avoid pre-commit churn)

- Pre-commit is configured (`.pre-commit-config.yaml`) and will:
  - enforce `black`
  - run `reorder-python-imports` with `--add-import "from __future__ import annotations"`
  - run `pyupgrade --py37-plus`
  - run `flake8` (config in `setup.cfg`, max line length 127; ignores include `F401`)
- When adding new Python files, put `from __future__ import annotations` first to avoid auto-rewrites.
