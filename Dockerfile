# python:3.12-slim pinned by its multi-arch index digest, so the same Dockerfile
# builds natively on arm64 and x86_64. Resolved 2026-10-14 with
# `docker buildx imagetools inspect python:3.12-slim`:
#   version: 3.12.15-slim-trixie, created 2026-10-06
#   linux/arm64/v8 manifest: sha256:16bf2b5c59a08523519d3c8589deca849285dee1d62a3653e124c3dba340f15e
#   linux/amd64 manifest:    sha256:2b4f19dae3a777dfc3b76730bda1e82e1f66ab2a2686fa93ca78edbfb4f04ffe
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

# Keep these in sync with requirements.txt.
ARG ORT_VERSION=1.31.0
ARG NUMPY_VERSION=2.4.6

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install "onnxruntime==${ORT_VERSION}" "numpy==${NUMPY_VERSION}"

WORKDIR /app
COPY bench/ bench/

# models/ and results/ are mounted at runtime (/app/models, /app/results), so the
# CLI's relative defaults resolve the same way as on the host.
# No CPU limits here: they are applied with `docker run --cpus`.
ENTRYPOINT ["python", "bench/run.py"]
