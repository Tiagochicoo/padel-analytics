"""
download_data.py
================
Download training datasets from Roboflow (players + court keypoints).

Roboflow datasets are identified by  <workspace>/<project>/<version> and an
export format. We use the Ultralytics "yolov8" format, which Ultralytics reads
directly (works for both detection and pose/keypoint projects).

Usage:
    python src/download_data.py --task detection
    python src/download_data.py --task pose

Credentials + dataset IDs come from the .env file (see .env.example).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make project root importable so `python src/...` resolves utils.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Map each task to its env-var namespace and local destination.
TASK_MAP = {
    "detection": {
        "workspace": "ROBOFLOW_PLAYER_WORKSPACE",
        "project": "ROBOFLOW_PLAYER_PROJECT",
        "version": "ROBOFLOW_PLAYER_VERSION",
        "dest": PROJECT_ROOT / "data" / "datasets" / "players",
    },
    "pose": {
        "workspace": "ROBOFLOW_COURT_WORKSPACE",
        "project": "ROBOFLOW_COURT_PROJECT",
        "version": "ROBOFLOW_COURT_VERSION",
        "dest": PROJECT_ROOT / "data" / "datasets" / "court_keypoints",
    },
}

# Roboflow export format -> Ultralytics. "yolov8" covers detection + keypoints.
EXPORT_FORMAT = os.getenv("ROBOFLOW_EXPORT_FORMAT", "yolov8")


def download(task: str) -> Path:
    """Download and extract a Roboflow dataset for the given task."""
    try:
        from roboflow import Roboflow
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "roboflow not installed. Run: uv pip install -r requirements.txt"
        ) from exc

    cfg = TASK_MAP[task]

    workspace = os.getenv(cfg["workspace"])
    project = os.getenv(cfg["project"])
    version = os.getenv(cfg["version"])
    api_key = os.getenv("ROBOFLOW_API_KEY")

    missing = [
        name
        for name, val in [
            ("ROBOFLOW_API_KEY", api_key),
            (cfg["workspace"], workspace),
            (cfg["project"], project),
            (cfg["version"], version),
        ]
        if not val
    ]
    if missing:
        raise SystemExit(
            f"Missing env vars: {missing}. Copy .env.example -> .env and fill them in."
        )

    dest = cfg["dest"]
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[{task}] Downloading {workspace}/{project} v{version} -> {dest}")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(int(version)).download(EXPORT_FORMAT, location=str(dest))

    print(f"[{task}] Done. Dataset at: {dataset.location}")
    print("      Verify the data.yaml `names`/`nc`/`kpt_shape` match configs/*.yaml")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Roboflow datasets.")
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASK_MAP.keys()),
        help="Which dataset to download.",
    )
    args = parser.parse_args()

    # Load .env from the project root.
    load_dotenv(PROJECT_ROOT / ".env")
    download(args.task)


if __name__ == "__main__":
    main()
