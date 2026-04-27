# syntax=docker/dockerfile:1.6
#
# sae-muc on a shared GPU server. CUDA 12.2 matches the host driver (535.x);
# PyTorch wheels bundle their own cuDNN, so the -base image is enough.
# Build with scripts/docker/build.sh — it passes UID/GID build-args so the
# container's appuser owns files at the same UID as the host invoker.

FROM nvidia/cuda:12.2.2-base-ubuntu22.04

ARG USER_UID=1000
ARG USER_GID=1000
ARG USER_NAME=appuser

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# uv binary from the official distroless image (glibc-compatible static build).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN groupadd --gid ${USER_GID} ${USER_NAME} \
 && useradd  --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USER_NAME}

# Keep the venv outside /app so a runtime bind-mount of the repo onto /app
# does not shadow the installed dependencies.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1

RUN mkdir -p /opt/venv /opt/uv-python /app \
 && chown -R ${USER_UID}:${USER_GID} /opt/venv /opt/uv-python /app

USER ${USER_NAME}
WORKDIR /app

# Sync first with metadata + minimum source needed for the editable install
# of sae_muc. The uv cache mount keeps the torch wheel between rebuilds.
COPY --chown=${USER_UID}:${USER_GID} pyproject.toml uv.lock README.md ./
COPY --chown=${USER_UID}:${USER_GID} src ./src

RUN --mount=type=cache,target=/home/${USER_NAME}/.cache/uv,uid=${USER_UID},gid=${USER_GID} \
    uv sync --all-extras --frozen

# Bring in everything else (configs, scripts, tests). At runtime the repo is
# bind-mounted onto /app, overlaying these — but the editable .pth in
# /opt/venv keeps pointing at /app/src, so live edits work.
COPY --chown=${USER_UID}:${USER_GID} . .

# Wrapper that sets umask before exec, so files written to shared storage
# (`/mnt/ssd/sae-muc/`) inherit group-write (0664/0775) and other team
# members can dozapis'/resume runs. UMASK env var overrides the default.
USER root
RUN printf '#!/bin/bash\numask "${UMASK:-002}"\nexec "$@"\n' > /usr/local/bin/with-umask \
 && chmod +x /usr/local/bin/with-umask
USER ${USER_NAME}

ENTRYPOINT ["/usr/local/bin/with-umask", "sae-muc"]
CMD ["--help"]
