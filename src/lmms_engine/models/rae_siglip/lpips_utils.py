"""Utility helpers for loading LPIPS weights (adapted from CompVis/taming-transformers)."""
import hashlib
import os
import pathlib
from typing import Optional


def _lazy_imports():  # pragma: no cover - runtime dependency check
    try:
        import requests
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Please install 'requests' and 'tqdm' to enable LPIPS support.") from exc
    return requests, tqdm


URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1",
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth",
}


def _download(url: str, local_path: str, chunk_size: int = 1024) -> None:
    requests, tqdm = _lazy_imports()
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as response:
        total_size = int(response.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as file_obj:
                for data in response.iter_content(chunk_size=chunk_size):
                    if data:
                        file_obj.write(data)
                        pbar.update(len(data))


def get_ckpt_path(name: str, root: Optional[str] = None, check: bool = False) -> str:
    assert name in URL_MAP, f"Unknown LPIPS checkpoint '{name}'"
    if root is None:
        torch_home = os.environ.get("TORCH_HOME", "~/.cache/torch")
        root = pathlib.Path(torch_home) / "checkpoints"
        root = os.path.abspath(root)
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path):
        print(f"Downloading {name} LPIPS weights to {path}")
        _download(URL_MAP[name], path)
    return path
