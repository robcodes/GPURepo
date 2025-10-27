"""Helper snippet for running sprite-matting-runner inside Kaggle notebooks.

Paste the contents of this file into a single Python cell.  Update the three
paths (``IMG``, ``MSK``, and ``SRC_WEIGHTS``) to match your dataset locations
before executing the cell.  The script keeps both ``GPURepo`` and
``FBA_Matting`` in sync with their upstream ``main``/``master`` branches,
performs an editable install of the runner, and finally executes the CLI with
the correct environment variables so the FBA model can be imported.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], **kwargs) -> None:
    """Print and execute a subprocess command."""

    print("$", " ".join(cmd))
    subprocess.check_call(cmd, **kwargs)


def ensure_repo(path: Path, url: str) -> None:
    """Clone the repository if it does not exist, otherwise hard-reset it."""

    if not path.exists():
        run(["git", "clone", "-q", url, str(path)])
        return

    run(["git", "-C", str(path), "fetch", "origin", "--prune"])
    remote_head = (
        subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "origin/HEAD"]
        )
        .decode()
        .strip()
    )
    branch = remote_head.split("/", 1)[1] if "/" in remote_head else remote_head
    run(["git", "-C", str(path), "checkout", branch])
    run(["git", "-C", str(path), "reset", "--hard", f"origin/{branch}"])


# --- customise these paths for your notebook ---
IMG = "/kaggle/input/your-dataset/spritesheet.png"
MSK = "/kaggle/input/your-dataset/spritesheet_mask.png"
SRC_WEIGHTS = "/kaggle/working/checkpoints/fba_matting.pth"


def _require_path(path_str: str, label: str) -> Path:
    """Validate that the user customised ``path_str`` and that it exists."""

    if not path_str or "your-dataset" in path_str:
        raise ValueError(
            f"Update the {label} path before running this cell (current value: {path_str!r})."
        )

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at {path}. Double-check the dataset/weights location."
        )
    return path


# --- fixed locations inside the Kaggle runtime ---
REPO_ROOT = Path("/kaggle/working/GPURepo")
RUNNER_DIR = REPO_ROOT / "sprite-matting-runner"
FBA_DIR = Path("/kaggle/working/FBA_Matting")
OUT_DIR = Path("/kaggle/working/alpha_out")

# 1) update code repositories
ensure_repo(REPO_ROOT, "https://github.com/robcodes/GPURepo.git")
ensure_repo(FBA_DIR, "https://github.com/MarcoForte/FBA_Matting.git")

# 2) install the runner (editable) so the console script is available
run([sys.executable, "-m", "pip", "install", "-q", "-e", str(RUNNER_DIR)])

# 3) resolve user-specified assets (will raise early if placeholders remain)
img_path = _require_path(IMG, "IMG")
mask_path = _require_path(MSK, "MSK")
weights_src = _require_path(SRC_WEIGHTS, "SRC_WEIGHTS")

# copy weights into the repo-local checkpoints directory
dst_ckpt = RUNNER_DIR / "checkpoints" / "fba_matting.pth"
dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
if weights_src.resolve() != dst_ckpt.resolve():
    print(f"Copying weights to {dst_ckpt} ...")
    shutil.copy2(weights_src, dst_ckpt)
else:
    print("Weights already in place.")

# 4) ensure the output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 5) run the CLI with the FBA repo on the module path via SMR_FBA_REPO
cmd = [
    "sprite-matting",
    "run",
    "--image",
    str(img_path),
    "--mask",
    str(mask_path),
    "--model",
    "fba",
    "--weights",
    str(RUNNER_DIR / "checkpoints" / "fba_matting.pth"),
    "--out-dir",
    str(OUT_DIR),
    "--tri-fg",
    "8",
    "--tri-bg",
    "8",
    "--min-area",
    "100",
    "--pad",
    "20",
    "--max-crop-pixels",
    "1500000",
    "--pixel-budget",
    "2000000",
    "--amp",
    "--device",
    "cuda:0",
]

env = os.environ.copy()
env["SMR_FBA_REPO"] = str(FBA_DIR)

print("\nRunning sprite-matting ...")
run(cmd, env=env)

print("\nAlphas written to:", OUT_DIR)
