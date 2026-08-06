FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    VENV=/opt/lmms-engine-venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PATH=/opt/lmms-engine-venv/bin:/root/.local/bin:$PATH

# RDMA packages are required by NCCL's InfiniBand transport.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git ninja-build \
        ffmpeg libsndfile1 libgl1 libglib2.0-0 \
        ibverbs-providers ibverbs-utils libibverbs1 rdma-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN uv python install 3.13 \
    && uv venv "$VENV" --python 3.13 --seed \
    && uv pip install --python "$VENV/bin/python" \
        "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" \
        --index-url https://download.pytorch.org/whl/cu130

WORKDIR /workspace/lmms-engine
COPY . .

RUN uv pip install --python "$VENV/bin/python" -e ".[all]" \
    && uv pip install --python "$VENV/bin/python" \
        "transformers==5.7.0" flash-linear-attention liger-kernel packaging psutil \
    && uv pip install --python "$VENV/bin/python" torchcodec \
        --index-url https://download.pytorch.org/whl/cu130 \
    && uv pip install --python "$VENV/bin/python" --force-reinstall --no-deps \
        "opencv-python==4.12.0.88"

ARG FLASH_ATTN_CUDA_ARCHS="80;86;89;90;100"
ARG MAX_JOBS=20
RUN FLASH_ATTN_CUDA_ARCHS="$FLASH_ATTN_CUDA_ARCHS" MAX_JOBS="$MAX_JOBS" NVCC_THREADS=2 \
    uv pip install --python "$VENV/bin/python" flash-attn --no-build-isolation

CMD ["bash"]
