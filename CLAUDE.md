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

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**This project has a knowledge graph. Start with the code-review-graph
MCP tools to narrow scope, then read the source.** The graph is cheaper than scanning files and
gives you structural context (callers, dependents, test coverage) that file search cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

### Verify in the source

- Narrow scope with the graph, then read the source. Do not change code from graph output alone.
- For any non-trivial change, read the implementation and the relevant tests before concluding.
- Verify the exact source when touching behavior, database logic, migrations, retries, fallbacks,
  recovery, or compatibility code.
- When the graph and the source disagree, the source wins. The graph may be stale or may not
  model that relationship.
- An empty graph result can mean "not indexed" or "not statically visible", not "does not exist".

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.
<!-- /code-review-graph MCP tools -->
