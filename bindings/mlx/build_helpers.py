"""Build-only helpers for this binding; no other framework is imported."""

import os
from pathlib import Path
import re
import shutil


def native_root(binding_dir):
    binding_dir = Path(binding_dir).resolve()
    snapshot = binding_dir / "_native"
    root = snapshot if snapshot.is_dir() else binding_dir.parents[1]
    for name in ("include", "src", "kernels", "dependencies/metalrt", "dependencies/fmt"):
        if not (root / name).is_dir():
            raise RuntimeError("Missing native sources; use the full checkout with initialized submodules or an sdist")
    return root


def project_version(root):
    text = (Path(root) / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"project\(\s*tiny_metal_nn\s+VERSION\s+([0-9.]+)", text, re.I)
    if match is None:
        raise RuntimeError("Could not read the version from CMakeLists.txt")
    return match.group(1)


def environment_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "on", "yes"):
        return True
    if value in ("0", "false", "off", "no"):
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, on/off, or yes/no")


def snapshot_files(root):
    """Copy canonical sources only into an sdist's private native snapshot."""
    root = Path(root)
    files = {root / "CMakeLists.txt", root / ".clang-format"}
    for directory in ("include", "src", "kernels", "dependencies", "bindings"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if any(part in (".git", "build", "dist", "__pycache__", ".pytest_cache", "_native")
                   or part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.is_file() and path.name != ".DS_Store" and path.suffix not in (".so", ".pyc"):
                files.add(path)
    return sorted(files)


def copy_runtime_assets(root, package_dir):
    destinations = []
    shader_dir = Path(root) / "kernels"
    for source in sorted(shader_dir.rglob("*.metal")):
        destination = Path(package_dir) / "rtc" / "include" / source.relative_to(shader_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        destinations.append(destination)
    return destinations
