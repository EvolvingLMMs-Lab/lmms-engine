FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace

# Copy source (filtered by .dockerignore)
COPY . .

# Install lmms-engine with all extras, flash-attn, and liger-kernel
RUN uv pip install --system -e ".[all]" && \
    uv pip install --system flash-attn --no-build-isolation && \
    uv pip install --system liger-kernel
