# Image that runs LLM-generated experiment scripts. orbit starts it with
# --network none --read-only --cap-drop ALL and usually --user <host uid>, so
# everything must be baked in at build time and nothing may need write access
# outside /work and /tmp.
FROM python:3.12.14-slim

LABEL org.opencontainers.image.title="orbit-sandbox" \
      org.opencontainers.image.description="Isolated runtime for orbit's generated ML experiment scripts" \
      org.opencontainers.image.source="https://github.com/OWNER/orbit" \
      org.opencontainers.image.licenses="MIT"

# libgomp1: OpenMP runtime required by lightgbm and xgboost wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY sandbox-requirements.txt /tmp/requirements.txt
# Wheels only: fail rather than compile from source (no toolchain in the image).
# Bytecode is precompiled here because the root filesystem is read-only at runtime.
RUN pip install --no-cache-dir --only-binary=:all: --root-user-action=ignore \
        --disable-pip-version-check -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

RUN useradd --uid 1000 --user-group --no-create-home --home-dir /tmp \
        --shell /usr/sbin/nologin orbit \
 && mkdir -p /work && chown orbit:orbit /work

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

WORKDIR /work
USER orbit

CMD ["python", "solution.py"]
