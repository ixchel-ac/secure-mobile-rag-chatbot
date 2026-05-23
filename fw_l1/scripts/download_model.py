"""Download the latest FW-L1 ONNX model from W&B.

Downloads the fw-l1-model:latest artifact (ONNX + tokenizer) to fw_l1/models/.
Also copies the fused model to the Android app assets if the path exists.

Usage:
    cd fw_l1 && uv run l1-download                     # download latest
    cd fw_l1 && uv run l1-download --version v2         # download specific version
    cd fw_l1 && uv run l1-download --android-assets ~/Documents/local-dev/BaselineChatbot/app/src/main/assets
"""

import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"

# Load .env from project root (picks up WANDB_API_KEY)
load_dotenv(PROJECT_ROOT / ".env")

WANDB_PROJECT = "mobile-rag-firewall"
WANDB_ARTIFACT = "fw-l1-model"

# Default Android assets path (relative to project root's sibling)
DEFAULT_ANDROID_ASSETS = PROJECT_ROOT.parent / "BaselineChatbot" / "app" / "src" / "main" / "assets"


def main():
    parser = argparse.ArgumentParser(description="Download FW-L1 ONNX model from W&B")
    parser.add_argument("--version", default="latest",
                        help="Artifact version (default: latest). Examples: latest, v0, v1, v2")
    parser.add_argument("--output", default=None,
                        help=f"Output directory (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--android-assets", default=None,
                        help="Path to Android app assets/ directory. If provided (or auto-detected), "
                             "copies the fused model there for on-device deployment.")
    parser.add_argument("--no-android", action="store_true",
                        help="Skip Android asset copy even if the path exists")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else DEFAULT_MODEL_DIR
    artifact_name = f"{WANDB_ARTIFACT}:{args.version}"

    # Step 1: Download from W&B
    print(f"[download] Artifact: {artifact_name}")
    print(f"[download] Output:   {output_dir}")

    import wandb

    run = wandb.init(project=WANDB_PROJECT, job_type="download-model")
    artifact = run.use_artifact(artifact_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact.download(root=str(output_dir))
    run.finish()

    # Verify files
    onnx_path = output_dir / "fw_l1.onnx"
    tokenizer_path = output_dir / "tokenizer"

    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / 1024 / 1024
        print(f"[download] fw_l1.onnx: {size_mb:.1f} MB")
    else:
        print(f"[download] WARNING: fw_l1.onnx not found in artifact")

    if tokenizer_path.exists():
        print(f"[download] tokenizer/: {sum(1 for _ in tokenizer_path.iterdir())} files")
    else:
        print(f"[download] WARNING: tokenizer/ not found in artifact")

    # Step 2: Copy fused model to Android assets (if applicable)
    fused_path = output_dir / "fw_l1_fused.onnx"
    if args.no_android:
        print("[download] Skipping Android asset copy (--no-android)")
    else:
        android_assets = Path(args.android_assets) if args.android_assets else DEFAULT_ANDROID_ASSETS
        if android_assets.exists():
            if fused_path.exists():
                dest = android_assets / "fw_l1_fused.onnx"
                shutil.copy2(fused_path, dest)
                print(f"[download] Copied fused model to {dest}")
            else:
                # No fused model — user needs to run fuse_tokenizer.py first
                print(f"[download] No fused model found at {fused_path}")
                print(f"[download] Run: uv run --python 3.13 --with 'onnxruntime-extensions>=0.13' "
                      f"python scripts/fuse_tokenizer.py")
                print(f"[download] Then re-run this script to copy to Android assets")
        elif args.android_assets:
            print(f"[download] WARNING: Android assets path not found: {android_assets}")
        else:
            print(f"[download] Android assets not found at default path — skipping")

    print(f"\n[download] Done. Model ready at {output_dir}")
    print(f"[download] Artifact version: {artifact.version}")


if __name__ == "__main__":
    main()
