# Repository Guidelines

## Project Structure & Module Organization
- `src/monoid/`: Python package entrypoint.
- `src/monoid/embed/`: MonoidEmbed model + config.
- `src/monoid/embed/monoid_cpu/`: CPU kernel + C++ extension sources.
- `scripts/`: runnable utilities for training, evals, and benchmarks.
- `docs/`: design and spec notes (source of truth is `docs/embed.md`).
- `tests/`: pytest suite (if present).

## Build, Test, and Development Commands
- Use `uv` for environment and dependencies.
  - `uv venv --system-site-packages .venv`
  - `uv pip install --python .venv/bin/python -r requirements.txt` (if a requirements file exists)
  - `uv add <package>` to update `pyproject.toml`.
- Always load credentials before runs: `source .env.sh`.
- Run scripts via `uv run` when possible (e.g., `source .env.sh; uv run --python .venv/bin/python -- scripts/run_mteb_eval.py --help`).
- Training entrypoint: `source .env.sh; uv run --python .venv/bin/python -- scripts/run_embed_training.py ...`.
- For training runs, include `--use_wandb` and a `--run_name`.
- Use `tmux` for long-running training/eval sessions.
- The CPU kernel extension is compiled on demand via `torch.utils.cpp_extension` in `src/monoid/embed/monoid_cpu/extension.py`.

## Coding Style & Naming Conventions
- Python with 4-space indentation.
- Naming: `snake_case` for files/functions, `CamelCase` for classes.
- Keep imports local and explicit; avoid side effects at import time in modules.

## Testing Guidelines
- Tests use pytest and live in `tests/`.
- Name new tests `test_<feature>.py` and functions `test_<behavior>`.

## Commit & Pull Request Guidelines
- Prefer conventional prefixes (`feat:`, `chore:`, `build:`) with short, imperative summaries.
- PRs should include a clear description, tests run, and any artifact impacts.

## Architecture Notes
- `docs/embed.md` is the canonical MonoidEmbed spec; align code changes to it.
- `MONOID_PRESET` environment variable can be used to select preset configs.

## Security & Configuration Tips
- Use environment variables for access tokens and API keys when needed (e.g., W&B).
