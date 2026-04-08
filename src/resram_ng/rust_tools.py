from __future__ import annotations

import glob
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rust_crate_dir() -> Path:
    return _repo_root() / "src" / "resram_rust"


def build_wheel() -> int:
    """Build the Rust extension wheel via maturin in the uv-managed environment."""
    crate_dir = _rust_crate_dir()
    cmd = [sys.executable, "-m", "maturin", "build", "--release", "--manifest-path", str(crate_dir / "Cargo.toml")]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def install_latest_wheel() -> int:
    """Install the newest built wheel using uv pip against this interpreter."""
    wheel_dir = _rust_crate_dir() / "target" / "wheels"
    wheels = sorted(glob.glob(str(wheel_dir / "*.whl")))
    if not wheels:
        print(f"No wheels found in {wheel_dir}. Run resram-rust-build first.")
        return 1

    latest_wheel = wheels[-1]
    cmd = ["uv", "pip", "install", "--python", sys.executable, latest_wheel]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def develop() -> int:
    """Build and install the Rust extension in editable/develop mode via maturin."""
    crate_dir = _rust_crate_dir()
    cmd = [sys.executable, "-m", "maturin", "develop", "--release", "--manifest-path", str(crate_dir / "Cargo.toml")]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)
