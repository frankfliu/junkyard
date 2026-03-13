# OpenAI Compatibility Test Suite (CTS)

This project is set up to test compatibility with OpenAI-compatible APIs.

## Setup

1. Install `uv` if you haven't already.
2. Install dependencies:
   ```bash
   uv sync
   ```

## Running Tests

Run the compatibility tests using `pytest`:
```bash
uv run pytest
```

## Development

Lint and format the code:
```bash
uv run ruff check . --fix
uv run ruff format .
uv run toml-sort pyproject.toml
```

Note: The current tests use a mock base URL (`http://localhost:8000/v1`). You should update the tests or set up a local server (like LiteLLM, vLLM, or LM Studio) to run real integration tests.
