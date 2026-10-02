# Image that serves registered orbit models with `mlflow models serve`: the sandbox's
# pinned libraries (so a model unpickles exactly as it was fitted and verified) plus MLflow.
# It is only used for models that already passed export verification; generated experiment
# code still runs in orbit-sandbox, never here. Build it after the sandbox image:
#   docker compose -f docker/compose.yaml build serve
FROM orbit-sandbox:latest

# Keep in step with the MLflow server image in compose.yaml.
ARG MLFLOW_VERSION=3.16.1

LABEL org.opencontainers.image.title="orbit-serve" \
      org.opencontainers.image.description="MLflow scoring server for orbit's registered models" \
      org.opencontainers.image.source="https://github.com/dsadulla1/orbit-ml" \
      org.opencontainers.image.licenses="MIT"

USER root
# The sandbox's pins are frozen as constraints so installing MLflow cannot move them.
# `mlflow` (not mlflow-skinny) brings what the scoring server needs: uvicorn and fastapi
# (this MLflow version's default on Linux), plus flask and gunicorn for the legacy server.
RUN pip freeze > /tmp/constraints.txt \
 && pip install --no-cache-dir --only-binary=:all: --root-user-action=ignore \
        --disable-pip-version-check -c /tmp/constraints.txt "mlflow==${MLFLOW_VERSION}" \
 && rm /tmp/constraints.txt \
 && python -c "import fastapi, flask, gunicorn, mlflow, uvicorn; print(mlflow.__version__)"
USER orbit

EXPOSE 8080
# orbit passes the full `mlflow models serve ...` command.
CMD ["mlflow", "models", "serve", "--help"]
