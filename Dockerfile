# ============================================================
# Adaptive Motion Planner — Docker image
# ============================================================
# Multi-stage build:
#   builder  — installs Python deps into a venv
#   runtime  — copies venv + source, runs as non-root user
#
# Quick start:
#   docker build -t amp .
#   mkdir -p results/plans results/trajectories results/cbf
#   docker run --rm -v $(pwd)/results:/app/results amp
# ============================================================

# ── Stage 1: builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV MPLBACKEND=Agg \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

RUN useradd -m -u 1000 planner
WORKDIR /app
COPY --chown=planner:planner . .

USER planner

# Default command — runs the full pipeline.
# Mount your host results/ dir before running:
#   mkdir -p results/plans results/trajectories results/cbf
#   docker run --rm -v $(pwd)/results:/app/results amp
CMD ["python", "pipeline.py"]
