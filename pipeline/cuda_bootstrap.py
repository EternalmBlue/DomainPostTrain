from __future__ import annotations

import os
import site
import sys
from pathlib import Path


_REEXEC_FLAG = "DOMAINPOSTTRAIN_CUDA_BOOTSTRAP_REEXECED"
_DISABLE_FLAG = "DOMAINPOSTTRAIN_DISABLE_CUDA_BOOTSTRAP"


def _site_roots() -> list[Path]:
    roots: list[Path] = []
    for getter in (site.getusersitepackages, site.getsitepackages):
        try:
            value = getter()
        except Exception:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            path = Path(item).expanduser()
            if path not in roots:
                roots.append(path)
    for item in sys.path:
        if "site-packages" not in item:
            continue
        path = Path(item).expanduser()
        if path not in roots:
            roots.append(path)
    return roots


def _candidate_nccl_dirs() -> list[Path]:
    candidates: list[Path] = []
    for root in _site_roots():
        path = root / "nvidia" / "nccl" / "lib"
        if path.exists() and any(path.glob("libnccl.so*")) and path not in candidates:
            candidates.append(path)
    return candidates


def ensure_pip_cuda_libraries_preferred() -> None:
    """Prefer pip wheel CUDA libraries before importing torch.

    Some shared servers put an older system NCCL ahead of PyTorch's wheel
    dependencies. If torch imports against that library, importing torch can
    fail before application code runs. Re-exec is used because LD_LIBRARY_PATH
    is a dynamic-loader startup setting on Linux.
    """

    if os.name != "posix" or os.environ.get(_DISABLE_FLAG):
        return

    nccl_dirs = _candidate_nccl_dirs()
    if not nccl_dirs:
        return

    preferred = str(nccl_dirs[0].resolve())
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in current.split(":") if part]
    normalized_parts = [str(Path(part).expanduser().resolve()) for part in parts if Path(part).exists()]
    if normalized_parts[:1] == [preferred]:
        return

    reordered = [preferred] + [part for part in parts if str(Path(part).expanduser().resolve()) != preferred]
    os.environ["LD_LIBRARY_PATH"] = ":".join(reordered)
    if os.environ.get(_REEXEC_FLAG) == "1":
        return

    os.environ[_REEXEC_FLAG] = "1"
    print(f"Restarting with pip NCCL library first: {preferred}", file=sys.stderr)
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
