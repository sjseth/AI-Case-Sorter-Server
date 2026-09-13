"""Install and validate the AI Server's Python environment.

Ported from python_env/setup_torch.py in the parent CaseSorter project, with
the TensorFlow / ML.NET path removed (this server only serves PyTorch
ConvNeXt checkpoints) and the server-side HTTP deps (FastAPI, uvicorn,
Pillow) added. The auto-detect logic prefers a CUDA wheel on Ampere+ GPUs
with sufficient VRAM and otherwise falls back to the CPU wheel.

Usage:
    python setup.py            # auto-detect (default)
    python setup.py --mode cpu # force CPU
    python setup.py --mode gpu # force GPU (errors if unsuitable)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Modern PyTorch build (CUDA 12.8 wheels target Ampere+ / CC 8.0+).
TORCH_VERSION = "2.9.1"
TORCHVISION_VERSION = "0.24.1"
CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"

# PyTorch >= 2.2 supports NumPy 2.x. We pin to the modern major.
NUMPY_SPEC = "numpy>=2.0"

# Server-side HTTP stack, plus what the registry / remote-client API and the
# community integration need. Bump SERVER_DEPS_VERSION whenever this list
# changes so existing installs pick up the new packages on the next start.
SERVER_DEPS = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "Pillow>=10.0",
    "python-multipart>=0.0.9",   # multipart uploads (training images, ZIP import)
    "requests>=2.31",            # community backend client
    "msal>=1.28",                # community sign-in (Azure AD B2C)
]
SERVER_DEPS_VERSION = "2"

# Skip the install path entirely when this marker matches the current mode.
TORCH_MARKER_FILE = ".torch_setup_complete"
SERVER_DEPS_MARKER_FILE = ".server_deps_complete"

# Minimum requirements for the GPU path. Anything below is routed to CPU.
MIN_CUDA_COMPUTE = 8.0   # Ampere (RTX 30xx) and newer
MIN_VRAM_GB = 4.0


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"[SETUP] Running: {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed (rc={result.returncode}): {cmd}")
    return result


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def has_nvidia_gpu() -> bool:
    try:
        result = run("nvidia-smi --query-gpu=name --format=csv,noheader", check=False)
        return result.returncode == 0 and result.stdout.strip() != ""
    except Exception:
        return False


def get_gpu_compute_capability() -> float | None:
    try:
        result = run("nvidia-smi --query-gpu=compute_cap --format=csv,noheader", check=False)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip().splitlines()[0].strip())
    except Exception as exc:
        print(f"[SETUP] Could not detect GPU compute capability: {exc}", flush=True)
    return None


def get_gpu_vram_gb() -> float | None:
    try:
        result = run(
            "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            mib = float(result.stdout.strip().splitlines()[0].strip())
            return mib / 1024.0
    except Exception as exc:
        print(f"[SETUP] Could not detect GPU VRAM: {exc}", flush=True)
    return None


# ---------------------------------------------------------------------------
# Pip helpers
# ---------------------------------------------------------------------------

def pip_install(args: str) -> bool:
    cmd = f'"{sys.executable}" -m pip install {args}'
    result = run(cmd, check=False)
    return result.returncode == 0


def uninstall_torch() -> None:
    print("[SETUP] Uninstalling torch + torchvision...", flush=True)
    run(f'"{sys.executable}" -m pip uninstall -y torch torchvision', check=False)


def install_torch(index_url: str) -> bool:
    print(
        f"[SETUP] Installing PyTorch {TORCH_VERSION} / torchvision {TORCHVISION_VERSION} "
        f"from {index_url}",
        flush=True,
    )
    print(
        "[SETUP] This may take several minutes depending on your network speed.",
        flush=True,
    )
    return pip_install(
        f"--force-reinstall --no-cache-dir "
        f"torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} "
        f"--index-url {index_url}"
    )


def install_numpy() -> bool:
    print(f"[SETUP] Installing numpy ({NUMPY_SPEC})...", flush=True)
    return pip_install(f'--force-reinstall --no-cache-dir "{NUMPY_SPEC}"')


def install_server_deps(base_dir: Path) -> bool:
    marker = base_dir / SERVER_DEPS_MARKER_FILE
    if marker.exists() and marker.read_text().strip() == SERVER_DEPS_VERSION:
        print("[SETUP] Server deps already installed. Skipping.", flush=True)
        return True

    print("[SETUP] Installing server dependencies (FastAPI, uvicorn, Pillow, msal, ...)...", flush=True)
    if not pip_install(" ".join(f'"{spec}"' for spec in SERVER_DEPS)):
        print(
            "[SETUP] ERROR: Server dep install failed. The server will not start.",
            flush=True,
        )
        return False

    marker.write_text(SERVER_DEPS_VERSION)
    print("[SETUP] Server deps installed successfully.", flush=True)
    return True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def torch_self_test() -> tuple[bool, bool, bool]:
    """Returns (import_ok, cuda_available, numpy_ok).

    Runs in a fresh interpreter so it isn't fooled by import caching.
    """
    print("[SETUP] Self-testing torch + numpy...", flush=True)
    code = (
        "import torch\n"
        "import numpy\n"
        "print('IMPORT_OK')\n"
        "print('NUMPY:', numpy.__version__)\n"
        "print('CUDA:', torch.cuda.is_available())\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    if result.returncode != 0:
        return False, False, False
    return (
        "IMPORT_OK" in result.stdout,
        "CUDA: True" in result.stdout,
        "NUMPY:" in result.stdout,
    )


def verify_existing_install(expected_mode: str) -> bool:
    print("[SETUP] Verifying existing PyTorch installation...", flush=True)
    import_ok, cuda_ok, numpy_ok = torch_self_test()
    if not import_ok or not numpy_ok:
        print("[SETUP] Existing torch/numpy failed self-test. Will reinstall.", flush=True)
        return False
    if expected_mode == "gpu" and not cuda_ok:
        print("[SETUP] GPU mode expected but CUDA not available. Will reinstall.", flush=True)
        return False
    print("[SETUP] Existing installation verified.", flush=True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Install and validate AI Server PyTorch env")
    parser.add_argument(
        "--mode",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Install mode preference",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    torch_marker = base_dir / TORCH_MARKER_FILE

    if torch_marker.exists():
        expected_mode = torch_marker.read_text().strip() or "cpu"
        if verify_existing_install(expected_mode):
            print("[SETUP] Torch setup verified. Skipping torch install.", flush=True)
            return 0 if install_server_deps(base_dir) else 1
        print("[SETUP] Existing torch install needs repair. Proceeding.", flush=True)
        torch_marker.unlink()

    gpu_present = has_nvidia_gpu()
    force_cpu = args.mode == "cpu"
    force_gpu = args.mode == "gpu"
    print(f"[SETUP] Requested mode: {args.mode}", flush=True)
    print(f"[SETUP] NVIDIA GPU detected: {gpu_present}", flush=True)

    if force_gpu and not gpu_present:
        print("[SETUP] ERROR: GPU mode requested but no NVIDIA GPU detected.", flush=True)
        return 1

    use_gpu = gpu_present and not force_cpu
    if use_gpu:
        compute_cap = get_gpu_compute_capability()
        vram_gb = get_gpu_vram_gb()
        print(f"[SETUP] GPU compute capability: {compute_cap}", flush=True)
        print(
            "[SETUP] GPU VRAM: "
            + (f"{vram_gb:.1f} GB" if vram_gb is not None else "unknown"),
            flush=True,
        )

        if compute_cap is None:
            print("[SETUP] Could not determine compute capability.", flush=True)
            if force_gpu:
                return 1
            use_gpu = False
        elif compute_cap < MIN_CUDA_COMPUTE:
            print(
                f"[SETUP] GPU compute capability {compute_cap:.1f} is below the "
                f"minimum {MIN_CUDA_COMPUTE}. Falling back to CPU.",
                flush=True,
            )
            if force_gpu:
                return 1
            use_gpu = False
        elif vram_gb is not None and vram_gb < MIN_VRAM_GB:
            print(
                f"[SETUP] GPU has {vram_gb:.1f} GB VRAM, below {MIN_VRAM_GB:.0f} GB "
                f"minimum. Falling back to CPU.",
                flush=True,
            )
            if force_gpu:
                return 1
            use_gpu = False

    if use_gpu:
        index_url = CUDA_INDEX_URL
        marker_text = "gpu"
        expected_cuda = True
    else:
        index_url = CPU_INDEX_URL
        marker_text = "cpu"
        expected_cuda = False

    uninstall_torch()
    if not install_torch(index_url):
        print("[SETUP] ERROR: torch install failed.", flush=True)
        return 1

    if not install_numpy():
        print("[SETUP] WARNING: numpy install failed; inference may be unstable.", flush=True)

    import_ok, cuda_ok, numpy_ok = torch_self_test()
    if not import_ok or not numpy_ok:
        print("[SETUP] ERROR: self-test failed after install.", flush=True)
        return 1
    if expected_cuda and not cuda_ok:
        print(
            "[SETUP] ERROR: GPU torch installed but CUDA is not available. "
            "Check NVIDIA driver / CUDA compatibility.",
            flush=True,
        )
        return 1

    torch_marker.write_text(marker_text)
    print(f"[SETUP] PyTorch ({marker_text}) install OK.", flush=True)

    if not install_server_deps(base_dir):
        return 1

    print("[SETUP] Done. Start the server with: python server.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
