from __future__ import annotations

import glob
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rust_crate_dir() -> Path:
    return _repo_root() / "src" / "resram_rust"


def _has_rustc() -> bool:
    return shutil.which("rustc") is not None


def _has_maturin() -> bool:
    return importlib.util.find_spec("maturin") is not None


def _check_rust_tooling() -> bool:
    if not _has_rustc():
        print("rustc not found. Skipping optional Rust build.")
        return False
    if not _has_maturin():
        print("maturin is not installed. Install with: uv pip install '.[rust]'")
        return False
    return True


def build_wheel() -> int:
    """Build the Rust extension wheel via maturin in the uv-managed environment."""
    if not _check_rust_tooling():
        return 0
    crate_dir = _rust_crate_dir()
    cmd = [sys.executable, "-m", "maturin", "build", "--release", "--manifest-path", str(crate_dir / "Cargo.toml")]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def install_latest_wheel() -> int:
    """Install the newest built wheel using uv pip against this interpreter."""
    wheel_dir = _rust_crate_dir() / "target" / "wheels"
    wheels = sorted(glob.glob(str(wheel_dir / "*.whl")))
    if not wheels:
        if _has_rustc():
            print(f"No wheels found in {wheel_dir}. Run resram-rust-build first.")
            return 1
        print("rustc not found and no prebuilt wheel available. Skipping optional Rust install.")
        return 0

    latest_wheel = wheels[-1]
    cmd = ["uv", "pip", "install", "--python", sys.executable, latest_wheel]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def develop() -> int:
    """Build and install the Rust extension in editable/develop mode via maturin."""
    if not _check_rust_tooling():
        return 0
    crate_dir = _rust_crate_dir()
    cmd = [sys.executable, "-m", "maturin", "develop", "--release", "--manifest-path", str(crate_dir / "Cargo.toml")]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)
