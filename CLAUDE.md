# CLAUDE.md

orbit is an LLM agent that runs ML experiments on tabular CSVs: the LLM writes a training
script, orbit executes it in a Docker sandbox, scores predictions against hidden labels,
tracks everything in MLflow/W&B and writes a verified report. See `docs/DESIGN.md`.

## Commands

```bash
uv sync                      # install (add --extra wandb / --extra mcp as needed)
uv run pytest                # tests: no Docker or API key needed
uv run orbit doctor          # check LLM endpoint, Docker, sandbox image, tracker
uv run orbit run examples/data/breast_cancer.csv -t diagnosis --steps 10
docker compose --env-file .env -f docker/compose.yaml build sandbox
```

## Invariants: do not break these

- Generated code runs only through `DockerSandbox`. Never add a user-facing way to execute
  it on the host. `tests/conftest.py::LocalSandbox` exists for trusted fixtures only.
- `hidden/` is never mounted into the sandbox, and all scoring happens in `metrics.py` on the
  orchestrator side. Test labels are scored once, for the final best experiment (`Agent.finalize`).
- The sandbox gets no network and no credentials; tracking happens in the orchestrator.
- Tracker failures must not abort a search (see `tracking._safe`).
- The LLM protocol is plain text (`PLAN:` plus one fenced Python block). Don't require native
  tool calling.
- `docker/sandbox-requirements.txt` and `prompts.LIBRARIES` must list the same libraries.
- Anything copied into a sandbox input directory (extra tables, helper files, export inputs) must
  not contain labels of rows that are being validated or tested. The export input is built from
  the user's source file after final scoring, never from `hidden/`.
- Exported models are pickles written by generated code: they are scanned and verified in the
  sandbox, and only ever loaded in the `orbit-serve` container, never on the host. The pins in
  `registry.py` must match `docker/sandbox-requirements.txt`.

## Workflow

- Commit in small, focused pieces.
- A code knowledge graph is maintained with `code-review-graph`. Run
  `code-review-graph update` after every commit, and `code-review-graph impact` (or
  `detect-changes`) before and after editing shared modules (`task.py`, `metrics.py`,
  `agent.py`, `sandbox.py`) to see the blast radius.
- Code targets Python 3.11+ on the host; the sandbox image is Python 3.12 with pandas 2.x.
  Host code must work with both pandas 2 and 3 (e.g. use `pd.api.types` checks, not
  `dtype == object`).
- Line endings are LF (`.gitattributes`).
- Tests must not write into the repo; use `tmp_path` (MLflow writes artifacts to `./mlruns`,
  so its tests `chdir` into `tmp_path`).
- Run Compose from the repo root with `--env-file .env -f docker/compose.yaml`; otherwise it
  reads `docker/.env` and falls back to defaults. The vLLM image is pinned (`VLLM_IMAGE_TAG`),
  and MLflow/vLLM are published on 127.0.0.1 only (no auth).
- Never read, print or commit `.env`; it holds API keys. `.env.example` is the template.
- Never put invented results in docs; quote only numbers from real runs, with model and settings.
