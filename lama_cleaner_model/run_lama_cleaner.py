import os
import subprocess
import shutil


def run_lama_cleaner(image_path: str, mask_path: str, output_path: str) -> None:
    """Run LaMa inpainting using the ``lama-cleaner`` CLI.

    This function requires the ``lama-cleaner`` package to be installed.
    It will call the command line interface to perform inpainting. The
    device (CPU/GPU) is automatically determined via PyTorch if available.
    """
    try:
        import torch  # noqa: F401
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    lama_bin = shutil.which("lama-cleaner")
    if lama_bin is None:
        raise RuntimeError(
            "lama-cleaner executable not found. Please install the lama-cleaner package."
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        lama_bin,
        "--model",
        "lama",
        "--device",
        device,
        "--input",
        image_path,
        "--mask",
        mask_path,
        "--output",
        output_path,
        "--interactive",
        "0",
    ]

    subprocess.run(cmd, check=True)
