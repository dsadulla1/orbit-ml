# Querying orbit runs over MCP

orbit logs every experiment to MLflow, so an MCP-capable assistant can read those runs
through MLflow's official MCP server (MLflow 3.5.1 or newer) and answer questions such as
"which experiment won the last breast_cancer run, and what did it change from its parent?"

## Install

```bash
uv sync --extra mcp        # installs mlflow[mcp]
```

## Configure your client

The server talks to the tracking server named by `MLFLOW_TRACKING_URI` (for example the
compose service on `http://localhost:5000`). `MLFLOW_MCP_TOOLS=ml` limits the server to
experiment, run and model tools, which is what orbit runs need; the default also exposes
GenAI tracing tools.

**Claude Code**

```bash
claude mcp add mlflow --env MLFLOW_TRACKING_URI=http://localhost:5000 --env MLFLOW_MCP_TOOLS=ml -- uv run --directory /path/to/orbit mlflow mcp run
```

**Claude Desktop / Cursor** (`mcpServers` section of the client's JSON config):

```json
{
  "mcpServers": {
    "mlflow": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/orbit", "mlflow", "mcp", "run"],
      "env": { "MLFLOW_TRACKING_URI": "http://localhost:5000", "MLFLOW_MCP_TOOLS": "ml" }
    }
  }
}
```

## What orbit logs

| MLflow object | Contents |
|---|---|
| Experiment `orbit` | All orbit runs |
| Parent run (named after the run folder) | params: dataset, target, metric, model, budgets; metrics: `baseline`, per-step `experiment_val_score` and `best_val_score_so_far`, `best_val_score`, `best_test_score`; tags: `orbit.best_experiment`, `orbit.stop_reason`; artifacts: `data_profile.txt`, `journal.json`, `report.md` |
| Nested run `exp-NNN-<stage>` | params: `stage`, `parent`, `debug_depth`; metrics: `val_score`, `duration_s`, tokens; tags: `orbit.plan`, `orbit.status`; artifacts: `solution.py`, `prompt.md`, `response.md`, `stdout.txt`, `stderr.txt` |

Useful prompts once connected:

- "List the nested runs of the latest orbit run sorted by val_score and summarise what changed between the top three."
- "Show the stderr of the failed experiments in run X and group them by root cause."

orbit's own `orbit summarize` doesn't need MCP: it builds the report straight from
`journal.json`.

## Weights & Biases

If you track with W&B instead, W&B publishes its own MCP server; see the W&B documentation
for setup.
