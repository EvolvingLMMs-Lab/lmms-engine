from __future__ import annotations

import fcntl
import hashlib
import errno
import json
import os
import shutil
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DELTA_FORMAT = "lmms-engine-disk-delta-v1"
PATCH_MAGIC = b"LMMSDELTA1\n"
STATE_DIR = ".delta_sync"
DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024
DEFAULT_LOCK_TIMEOUT_S = 600.0
LOCK_RETRY_INTERVAL_S = 0.05


def publish_delta_checkpoint(
    *,
    base_dir: str | Path,
    target_dir: str | Path,
    delta_dir: str | Path,
    base_version: int,
    target_version: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Path:
    """Publish a file-level byte delta from one HF checkpoint to another.

    The trainer still owns producing the target HF checkpoint. This function
    converts that full checkpoint into a deterministic on-disk delta directory
    that rollout actors can apply to a local materialized checkpoint.
    """

    base = Path(base_dir).expanduser().resolve()
    target = Path(target_dir).expanduser().resolve()
    final_delta = Path(delta_dir).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Delta base checkpoint does not exist: {base}")
    if not target.is_dir():
        raise FileNotFoundError(f"Delta target checkpoint does not exist: {target}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    tmp_delta = final_delta.with_name(f"{final_delta.name}.tmp-{os.getpid()}")
    if tmp_delta.exists():
        shutil.rmtree(tmp_delta)
    tmp_delta.mkdir(parents=True)
    (tmp_delta / "files").mkdir()
    (tmp_delta / "patches").mkdir()

    target_files = _checkpoint_files(target)
    base_files = _checkpoint_files(base)
    entries: list[dict[str, Any]] = []

    try:
        for relpath, target_path in sorted(target_files.items()):
            base_path = base_files.get(relpath)
            entry = _publish_file_delta(
                base_path=base_path,
                target_path=target_path,
                relpath=relpath,
                delta_dir=tmp_delta,
                block_size=block_size,
            )
            entries.append(entry)

        for relpath in sorted(set(base_files) - set(target_files)):
            entries.append({"path": relpath, "op": "delete"})

        manifest = {
            "format": DELTA_FORMAT,
            "base_version": int(base_version),
            "target_version": int(target_version),
            "block_size": int(block_size),
            "files": entries,
        }
        _atomic_write_json(tmp_delta / "delta_manifest.json", manifest)
        (tmp_delta / ".lmms_weight_sync_ready").write_text(
            f"version_id={int(target_version)}\n",
            encoding="utf-8",
        )

        if final_delta.exists():
            shutil.rmtree(final_delta)
        os.replace(tmp_delta, final_delta)
    except Exception:
        shutil.rmtree(tmp_delta, ignore_errors=True)
        raise

    return final_delta


def init_local_checkpoint(
    *,
    local_dir: str | Path,
    base_dir: str | Path,
    base_version: int,
    reset_if_newer: bool = False,
) -> Path:
    """Materialize a local checkpoint once, guarded by a per-directory lock."""

    local = Path(local_dir).expanduser().resolve()
    base = Path(base_dir).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Delta base checkpoint does not exist: {base}")

    local.mkdir(parents=True, exist_ok=True)
    with _apply_lock(local):
        base_identity = _checkpoint_identity(base)
        state = _read_state(local)
        if state is not None and _state_matches_base(
            state,
            base_dir=base,
            base_identity=base_identity,
            base_version=int(base_version),
            reset_if_newer=reset_if_newer,
        ):
            return local

        for child in local.iterdir():
            if child.name == STATE_DIR:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        _copy_checkpoint_files(base, local)
        _write_state(
            local,
            int(base_version),
            base_checkpoint_path=str(base),
            base_checkpoint_identity=base_identity,
        )
    return local


def apply_delta_checkpoints(
    *,
    local_dir: str | Path,
    delta_root: str | Path,
    target_version: int,
) -> Path:
    """Apply published deltas until local checkpoint reaches target_version."""

    local = Path(local_dir).expanduser().resolve()
    root = Path(delta_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Delta root does not exist: {root}")

    with _apply_lock(local):
        state = _read_state(local)
        if state is None:
            raise RuntimeError(f"Local delta checkpoint is not initialized: {local}")
        base_checkpoint_path = state.get("base_checkpoint_path")
        base_checkpoint_identity = state.get("base_checkpoint_identity")
        applied = int(state["version"])
        target = int(target_version)
        while applied != target:
            manifest_path = _find_next_delta_manifest(root, applied, target)
            manifest = _read_manifest(manifest_path)
            _apply_delta_dir(local, manifest_path.parent, manifest, expected_base_version=applied)
            applied = int(manifest["target_version"])
            _write_state(
                local,
                applied,
                base_checkpoint_path=base_checkpoint_path,
                base_checkpoint_identity=base_checkpoint_identity,
            )
    return local


def validate_delta_checkpoint(delta_dir: str | Path) -> dict[str, Any]:
    delta = Path(delta_dir).expanduser().resolve()
    manifest = _read_manifest(delta / "delta_manifest.json")
    return {
        "resolved_path": str(delta),
        "format": manifest["format"],
        "base_version": manifest["base_version"],
        "target_version": manifest["target_version"],
        "num_files": len(manifest.get("files", [])),
    }


def local_checkpoint_state(local_dir: str | Path) -> dict[str, Any] | None:
    local = Path(local_dir).expanduser().resolve()
    return _read_state(local)


def _publish_file_delta(
    *,
    base_path: Path | None,
    target_path: Path,
    relpath: str,
    delta_dir: Path,
    block_size: int,
) -> dict[str, Any]:
    target_sha = _sha256_file(target_path)
    target_size = target_path.stat().st_size
    if base_path is None or not base_path.exists():
        payload = _copy_payload_file(target_path, delta_dir, relpath)
        return {
            "path": relpath,
            "op": "copy",
            "payload": payload,
            "size": target_size,
            "sha256": target_sha,
        }

    if base_path.stat().st_size == target_size and _sha256_file(base_path) == target_sha:
        return {
            "path": relpath,
            "op": "same",
            "size": target_size,
            "sha256": target_sha,
        }

    if not _is_patchable_weight_file(target_path):
        payload = _copy_payload_file(target_path, delta_dir, relpath)
        return {
            "path": relpath,
            "op": "copy",
            "payload": payload,
            "size": target_size,
            "sha256": target_sha,
        }

    patch_relpath = f"patches/{_payload_name(relpath)}.patch"
    _write_patch_file(base_path, target_path, delta_dir / patch_relpath, block_size)
    return {
        "path": relpath,
        "op": "patch",
        "patch": patch_relpath,
        "size": target_size,
        "sha256": target_sha,
    }


def _apply_delta_dir(
    local_dir: Path,
    delta_dir: Path,
    manifest: dict[str, Any],
    *,
    expected_base_version: int,
) -> None:
    if int(manifest["base_version"]) != int(expected_base_version):
        raise RuntimeError(
            "Out-of-order delta apply: "
            f"local={expected_base_version}, manifest_base={manifest['base_version']}, delta={delta_dir}"
        )
    for entry in manifest.get("files", []):
        relpath = _safe_relpath(entry["path"])
        target_path = local_dir / relpath
        op = entry["op"]
        if op == "same":
            _verify_file(target_path, entry)
        elif op == "copy":
            payload_path = delta_dir / _safe_relpath(entry["payload"])
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(payload_path, target_path)
            _verify_file(target_path, entry)
        elif op == "patch":
            if not target_path.exists():
                raise FileNotFoundError(f"Cannot patch missing local checkpoint file: {target_path}")
            _apply_patch_file(target_path, delta_dir / _safe_relpath(entry["patch"]), int(entry["size"]))
            _verify_file(target_path, entry)
        elif op == "delete":
            target_path.unlink(missing_ok=True)
        else:
            raise ValueError(f"Unsupported delta file op {op!r} in {delta_dir}")


def _write_patch_file(base_path: Path, target_path: Path, patch_path: Path, block_size: int) -> None:
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with base_path.open("rb") as base_f, target_path.open("rb") as target_f, patch_path.open("wb") as patch_f:
        patch_f.write(PATCH_MAGIC)
        while True:
            target_chunk = target_f.read(block_size)
            if not target_chunk:
                break
            base_chunk = base_f.read(len(target_chunk))
            if base_chunk != target_chunk:
                patch_f.write(struct.pack("<QI", offset, len(target_chunk)))
                patch_f.write(target_chunk)
            offset += len(target_chunk)


def _apply_patch_file(target_path: Path, patch_path: Path, target_size: int) -> None:
    with target_path.open("r+b") as target_f, patch_path.open("rb") as patch_f:
        magic = patch_f.read(len(PATCH_MAGIC))
        if magic != PATCH_MAGIC:
            raise ValueError(f"Invalid delta patch magic: {patch_path}")
        while True:
            header = patch_f.read(struct.calcsize("<QI"))
            if not header:
                break
            if len(header) != struct.calcsize("<QI"):
                raise ValueError(f"Truncated delta patch header: {patch_path}")
            offset, length = struct.unpack("<QI", header)
            data = patch_f.read(length)
            if len(data) != length:
                raise ValueError(f"Truncated delta patch payload: {patch_path}")
            target_f.seek(offset)
            target_f.write(data)
        target_f.truncate(target_size)


def _find_next_delta_manifest(root: Path, applied_version: int, target_version: int) -> Path:
    candidates = []
    for manifest_path in root.glob("weight_v*/delta_manifest.json"):
        manifest = _read_manifest(manifest_path)
        base_version = int(manifest["base_version"])
        next_version = int(manifest["target_version"])
        if base_version == applied_version and next_version <= target_version:
            candidates.append((next_version, manifest_path))
    if not candidates:
        raise FileNotFoundError(
            "No delta checkpoint continues the local checkpoint chain: "
            f"root={root}, local_version={applied_version}, target_version={target_version}"
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != DELTA_FORMAT:
        raise ValueError(f"Unsupported delta manifest format in {path}: {manifest.get('format')!r}")
    return manifest


def _checkpoint_files(root: Path) -> dict[str, Path]:
    files = {}
    for path in root.rglob("*"):
        if not path.is_file() or STATE_DIR in path.parts:
            continue
        relpath = path.relative_to(root).as_posix()
        if relpath in {"delta_manifest.json", ".lmms_weight_sync_ready"}:
            continue
        files[relpath] = path
    return files


def _checkpoint_identity(root: Path) -> str:
    hasher = hashlib.sha256()
    for relpath, path in sorted(_checkpoint_files(root).items()):
        stat = path.stat()
        hasher.update(relpath.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(stat.st_size).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(str(stat.st_mtime_ns).encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _copy_checkpoint_files(src: Path, dst: Path) -> None:
    for relpath, src_path in _checkpoint_files(src).items():
        dst_path = dst / relpath
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def _copy_payload_file(path: Path, delta_dir: Path, relpath: str) -> str:
    payload_relpath = f"files/{_payload_name(relpath)}"
    payload_path = delta_dir / payload_relpath
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, payload_path)
    return payload_relpath


def _payload_name(relpath: str) -> str:
    digest = hashlib.sha256(relpath.encode("utf-8")).hexdigest()[:24]
    return f"{digest}-{Path(relpath).name}"


def _is_patchable_weight_file(path: Path) -> bool:
    return path.suffix in {".safetensors", ".bin"}


def _verify_file(path: Path, entry: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Delta apply expected file does not exist: {path}")
    if path.stat().st_size != int(entry["size"]):
        raise RuntimeError(f"Delta apply size mismatch for {path}: expected {entry['size']}, got {path.stat().st_size}")
    actual = _sha256_file(path)
    if actual != entry["sha256"]:
        raise RuntimeError(f"Delta apply sha256 mismatch for {path}: expected {entry['sha256']}, got {actual}")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_relpath(value: str) -> Path:
    relpath = Path(value)
    if relpath.is_absolute() or ".." in relpath.parts:
        raise ValueError(f"Unsafe relative path in delta manifest: {value!r}")
    return relpath


@contextmanager
def _apply_lock(local_dir: Path):
    sync_dir = local_dir / STATE_DIR
    sync_dir.mkdir(parents=True, exist_ok=True)
    lock_path = sync_dir / "lock"
    with lock_path.open("a+") as f:
        _acquire_file_lock(f, lock_path=lock_path, local_dir=local_dir)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _acquire_file_lock(handle: Any, *, lock_path: Path, local_dir: Path) -> None:
    deadline = time.monotonic() + DEFAULT_LOCK_TIMEOUT_S
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for local delta checkpoint lock: "
                    f"lock={lock_path}, local_dir={local_dir}, timeout_s={DEFAULT_LOCK_TIMEOUT_S}"
                ) from exc
            time.sleep(LOCK_RETRY_INTERVAL_S)


def _read_state(local_dir: Path) -> dict[str, Any] | None:
    state_path = local_dir / STATE_DIR / "state.json"
    try:
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _state_matches_base(
    state: dict[str, Any],
    *,
    base_dir: Path,
    base_identity: str,
    base_version: int,
    reset_if_newer: bool,
) -> bool:
    try:
        state_version = int(state["version"])
    except (KeyError, TypeError, ValueError):
        return False
    if state.get("base_checkpoint_path") != str(base_dir):
        return False
    if state.get("base_checkpoint_identity") != base_identity:
        return False
    if state_version < int(base_version):
        return False
    if reset_if_newer and state_version != int(base_version):
        return False
    return True


def _write_state(
    local_dir: Path,
    version: int,
    *,
    base_checkpoint_path: str | None = None,
    base_checkpoint_identity: str | None = None,
) -> None:
    state_path = local_dir / STATE_DIR / "state.json"
    payload: dict[str, Any] = {"version": int(version)}
    if base_checkpoint_path is not None:
        payload["base_checkpoint_path"] = base_checkpoint_path
    if base_checkpoint_identity is not None:
        payload["base_checkpoint_identity"] = base_checkpoint_identity
    _atomic_write_json(state_path, payload)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
