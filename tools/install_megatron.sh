#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
install_root="${MEGATRON_INSTALL_ROOT:-${repo_root}/output/megatron}"
venv_dir="${repo_root}/.venv"
runtime="${venv_dir}/bin/python"
python_bin="${PYTHON:-python3}"
uv_bin="${UV:-uv}"
runtime_mode="${MEGATRON_RUNTIME_MODE:-install}"
torch_version="${MEGATRON_TORCH_VERSION:-2.13.0}"
torchvision_version="${MEGATRON_TORCHVISION_VERSION:-0.28.0}"
torch_index="${MEGATRON_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
te_version="${MEGATRON_TE_VERSION:-2.18.0}"
apex_revision="${MEGATRON_APEX_REVISION:-575968bc1f9127ccd61003a681472a83af4ff1a1}"
bridge_revision=c0e164ed2aedac4ad1c877780e2564a19d5d54ec
core_revision=a0f793dfa4e776d99a8aa63bed74e0fafc91b0db

if ! command -v "${uv_bin}" >/dev/null 2>&1; then
    if [[ -x "${HOME}/.local/bin/uv" && -z "${UV:-}" ]]; then
        uv_bin="${HOME}/.local/bin/uv"
    else
        echo "Install uv or set UV to its executable path" >&2
        exit 1
    fi
fi
if [[ "${runtime_mode}" != install && "${runtime_mode}" != reuse ]]; then
    echo "MEGATRON_RUNTIME_MODE must be install or reuse" >&2
    exit 1
fi
if [[ ! -x "${runtime}" ]]; then
    if [[ -e "${venv_dir}" ]]; then
        echo "${venv_dir} exists but has no Python executable; refusing to replace it" >&2
        exit 1
    fi
    "${uv_bin}" venv --python "${python_bin}" "${venv_dir}"
fi
"${runtime}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit("Megatron Bridge v0.6.2 requires Python 3.12 in .venv")
if sys.prefix == sys.base_prefix:
    raise SystemExit("The target Python must belong to a virtual environment")
print(f'Installing into {sys.prefix}')
PY
if [[ "${runtime_mode}" == install ]]; then
    command -v c++ >/dev/null || { echo "A C++ compiler is required to build TE and Apex" >&2; exit 1; }
fi

mkdir -p "${install_root}"
checkout_source() {
    local source_url="$1" destination="$2" revision="$3"
    if [[ ! -e "${destination}" ]]; then
        git clone "${source_url}" "${destination}"
        git -C "${destination}" checkout --detach "${revision}"
    elif [[ "$(git -C "${destination}" rev-parse HEAD)" != "${revision}" ]] || \
        [[ -n "$(git -C "${destination}" status --porcelain --untracked-files=no)" ]]; then
        echo "${destination} must be clean at ${revision}; refusing to change it" >&2
        exit 1
    fi
}
checkout_source "${MEGATRON_BRIDGE_SOURCE:-https://github.com/NVIDIA/Megatron-Bridge.git}" \
    "${install_root}/Megatron-Bridge" "${bridge_revision}"
checkout_source "${MEGATRON_CORE_SOURCE:-https://github.com/NVIDIA/Megatron-LM.git}" \
    "${install_root}/Megatron-LM" "${core_revision}"

if [[ "${runtime_mode}" == install ]]; then
    "${uv_bin}" pip install --python "${runtime}" --index "${torch_index}" \
        --index-strategy unsafe-best-match "torch==${torch_version}" "torchvision==${torchvision_version}"
    "${uv_bin}" pip install --python "${runtime}" \
        'setuptools>=77,<82' wheel packaging ninja cmake pybind11 pip einops \
        onnx onnxscript pydantic nvdlfw-inspect 'nvidia-cudnn-frontend==1.26.0'
    cuda_version="$(${runtime} -c 'import torch; print(torch.version.cuda)')"
    if [[ "${cuda_version}" != 13.0 ]]; then
        echo "The install mode currently supports CUDA 13.0 PyTorch wheels; use MEGATRON_TORCH_INDEX=.../cu130" >&2
        exit 1
    fi
    "${uv_bin}" pip install --python "${runtime}" 'cuda-toolkit[nvcc,cccl,culibos,profiler]==13.0.3'
    export CUDA_HOME="$(${runtime} -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cu13")')"
    "${runtime}" - <<'PY'
import os
from pathlib import Path

libraries = Path(os.environ['CUDA_HOME']) / 'lib'
for name in ('cudart', 'cublas', 'cublasLt', 'curand', 'cusolver', 'cusparse'):
    link = libraries / f'lib{name}.so'
    versioned = libraries / f'lib{name}.so.13'
    if not link.exists() and not link.is_symlink() and versioned.exists():
        link.symlink_to(versioned.name)
PY
    export PATH="${CUDA_HOME}/bin:${venv_dir}/bin:${PATH}"
    export LD_LIBRARY_PATH="$(${runtime} -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent / "lib")')${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export CUDNN_PATH="$(${runtime} -c 'import importlib.metadata as m; print(m.distribution("nvidia-cudnn-cu13").locate_file("nvidia/cudnn"))')"
    export MAX_JOBS="${MAX_JOBS:-8}"
    export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(${runtime} -c 'import torch; print(";".join(sorted({"%d.%d" % torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})))')}"
    if [[ -z "${TORCH_CUDA_ARCH_LIST}" ]]; then
        echo "Expose a GPU or set TORCH_CUDA_ARCH_LIST for compilation" >&2
        exit 1
    fi
    "${runtime}" - <<'PY'
import os
import re
import subprocess
import torch

output = subprocess.check_output([os.environ['CUDA_HOME'] + '/bin/nvcc', '--version'], text=True)
version = re.search(r'release (\d+\.\d+)', output).group(1)
if version != torch.version.cuda:
    raise SystemExit(f'CUDA toolkit {version} differs from PyTorch CUDA {torch.version.cuda}')
if not torch.cuda.is_available():
    raise SystemExit('CUDA PyTorch cannot access the GPU; check the NVIDIA driver and container runtime')
print(f'Installing runtime: PyTorch {torch.__version__}, CUDA {version}')
PY
    "${uv_bin}" pip install --python "${runtime}" "transformer-engine[core_cu13]==${te_version}"
    NVTE_FRAMEWORK=pytorch "${uv_bin}" pip install --python "${runtime}" --no-build-isolation \
        "transformer-engine[pytorch,core_cu13]==${te_version}"
    checkout_source "${MEGATRON_APEX_SOURCE:-https://github.com/NVIDIA/apex.git}" \
        "${install_root}/apex" "${apex_revision}"
    APEX_CPP_EXT=1 APEX_CUDA_EXT=1 "${uv_bin}" pip install --python "${runtime}" \
        --no-build-isolation --no-deps "${install_root}/apex"
fi

"${runtime}" - <<'PY'
import torch
import transformer_engine.pytorch
import apex
import amp_C
import fused_weight_gradient_mlp_cuda
from apex.optimizers import FusedAdam

if not torch.cuda.is_available():
    raise SystemExit('CUDA PyTorch is required')
print(f'Runtime ready: PyTorch {torch.__version__}, CUDA {torch.version.cuda}, TE and compiled Apex')
PY

"${uv_bin}" pip install --python "${runtime}" packaging
"${runtime}" - "${repo_root}" "${install_root}" <<'PY'
import sys
import tomllib
from pathlib import Path
from packaging.requirements import Requirement

repo, root = map(Path, sys.argv[1:])
skip = {"torch", "torchvision", "triton", "megatron-core", "fast-hadamard-transform", "bitsandbytes"}
requirements = []
for project in (repo, root / "Megatron-Bridge"):
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    requirements.extend(dep for dep in metadata["project"]["dependencies"] if Requirement(dep).name not in skip)
requirements += [
    "transformers==5.8.1", "tokenizers==0.22.2", "flash-linear-attention==0.5.1",
    "fla-core==0.5.1", "tilelang==0.1.15", "megatron-energon==7.4.0", "omegaconf==2.3.0",
    "diffusers==0.41.0", "librosa==0.11.0", "deepspeed==0.18.8",
    "fsspec==2026.6.0", "s3fs==2026.6.0", "pycountry", "sentencepiece",
    "torchdiffeq", "decord", "tensorstore", "pybind11", "setuptools>=80,<82", "wheel",
    "nvidia-modelopt[torch]==0.44.0", "tiktoken", "flask-restful",
]
(root / "requirements.txt").write_text("\n".join(requirements) + "\n")
(root / "overrides.txt").write_text("\n".join(f"{name}; sys_platform == 'never'" for name in ("torch", "torchvision", "triton")) + "\n")
PY

"${uv_bin}" pip install --python "${runtime}" --no-build-isolation \
    --overrides "${install_root}/overrides.txt" -r "${install_root}/requirements.txt"
"${uv_bin}" pip install --python "${runtime}" --no-deps --no-build-isolation \
    -e "${install_root}/Megatron-LM" -e "${install_root}/Megatron-Bridge" -e "${repo_root}"
"${runtime}" - <<'PY'
from lmms_engine.megatron.patches import patch_bundled_flash_attention_cute
patch_bundled_flash_attention_cute()
from megatron.bridge import AutoBridge
from megatron.bridge.training.vlm_step import forward_step
from lmms_engine.datasets.processor.qwen3_vl_processor import Qwen3_VLDataProcessor
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.common.backends.tilelang import TileLangBackend
assert TileLangBackend.is_available(), "TileLang is required for Qwen3.5 on Hopper"
print("Megatron and LMMS processor imports passed")
PY
"${uv_bin}" pip freeze --python "${runtime}" > "${install_root}/installed.txt"
echo "Installed Megatron Bridge v0.6.2. Python: ${runtime}"
