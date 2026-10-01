# Docker setup

orbit needs Docker for the experiment sandbox. MLflow and a local vLLM server
are optional services defined in `compose.yaml`. Run every command below from
the repository root and pass `--env-file .env`, so Compose and orbit share one
configuration file (Compose would otherwise look for `docker/.env`).

```bash
cp .env.example .env        # then edit it
```

## Sandbox image

```bash
docker compose --env-file .env -f docker/compose.yaml build sandbox
```

This produces `orbit-sandbox:latest` (override with `ORBIT_SANDBOX_IMAGE`).
orbit starts one container per experiment with no network, a read-only root
filesystem, all capabilities dropped and CPU/memory/PID limits; see
`src/orbit/sandbox.py`. The libraries in `sandbox-requirements.txt` must match
`LIBRARIES` in `src/orbit/prompts.py`. Rebuild after changing either.

## MLflow

```bash
docker compose --env-file .env -f docker/compose.yaml up -d mlflow
```

UI and tracking API at http://localhost:5000 (`MLFLOW_PORT`, bound to
`MLFLOW_BIND`, default 127.0.0.1). Runs and
artifacts live in the `mlflow-data` volume. Set `MLFLOW_TRACKING_URI` in `.env`
so orbit logs there.

## Local LLM with vLLM

Prerequisites on the host:

1. An NVIDIA driver recent enough for the CUDA version of the vLLM image.
2. The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
   configured for Docker (`sudo nvidia-ctk runtime configure --runtime=docker`
   and restart Docker).
3. Sanity check: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
   should list your GPUs.

Start the server:

```bash
docker compose --env-file .env -f docker/compose.yaml --profile local-llm up -d vllm
docker compose --env-file .env -f docker/compose.yaml logs -f vllm   # first start downloads the weights
```

The default is `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` with tensor
parallelism 2 on GPUs 0 and 1, served as `orbit-coder`. Tensor parallelism must
divide the model's attention heads evenly, so a third GPU usually can't join;
it stays free for other work. To change GPUs, edit `device_ids` in
`compose.yaml` and keep `VLLM_TP` equal to its length. For a single 12 GB GPU,
use `VLLM_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`, `VLLM_TP=1` and one
device ID; that model is dense, so also drop `--enable-expert-parallel` from the
command.

Then point orbit at it in `.env`:

```
ORBIT_LLM_BASE_URL=http://localhost:8000/v1
ORBIT_LLM_API_KEY=EMPTY
ORBIT_LLM_MODEL=orbit-coder
ORBIT_LLM_TEMPERATURE=0.2
```

## Check everything

```bash
uv run orbit doctor
```

It pings the LLM, checks that Docker is reachable and the sandbox image exists,
and shows which tracker will be used.

## Hardening with gVisor (optional)

[gVisor](https://gvisor.dev/docs/user_guide/install/) runs each container
against a user-space kernel, which narrows the attack surface of generated code
further. After installing `runsc` and registering it with Docker, set
`ORBIT_SANDBOX_RUNTIME=runsc` in `.env`.

## Security notes

- Anyone who can run orbit can talk to the Docker daemon, which is
  root-equivalent on the host. Treat orbit users as host administrators.
- Generated code is untrusted. The sandbox flags limit what it can do, but
  container isolation is not a hard boundary; gVisor helps.
- The vLLM and MLflow endpoints have no authentication, so both are published
  on 127.0.0.1 only. To open the MLflow UI to other machines, set
  `MLFLOW_BIND=0.0.0.0`, and only on a trusted private network. An SSH tunnel
  (`ssh -L 5000:localhost:5000 <host>`) is the safer way to view it remotely.
