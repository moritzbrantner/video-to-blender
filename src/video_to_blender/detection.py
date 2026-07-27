from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from .io import read_json, run_command, write_json


def detect_mock(shot: dict) -> list[dict]:
    frames = sorted((Path(shot["clip"]).parent / "frames").glob("*"))
    labels = ("chair", "table", "plant")
    detections: list[dict] = []

    for frame_index, frame in enumerate(frames):
        label = labels[(frame_index + int(shot["id"].split("_")[1])) % len(labels)]
        digest = hashlib.sha1(f"{shot['id']}:{frame.name}".encode()).digest()
        x = 0.18 + digest[0] / 255 * 0.35
        y = 0.22 + digest[1] / 255 * 0.25
        detections.append(
            {
                "frame": frame.name,
                "frame_index": frame_index,
                "label": label,
                "confidence": round(0.68 + digest[2] / 255 * 0.25, 3),
                "bbox_norm": [
                    round(x, 4),
                    round(y, 4),
                    round(min(x + 0.2, 0.94), 4),
                    round(min(y + 0.34, 0.94), 4),
                ],
            }
        )

    path = Path(shot["clip"]).parent / "detections.json"
    write_json(path, {"backend": "mock", "detections": detections})
    return detections


def detect_yolo(shot: dict, model_name: str = "yolo11n.pt") -> list[dict]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed in the selected Python environment"
        ) from exc

    frames = sorted((Path(shot["clip"]).parent / "frames").glob("*"))
    if not frames:
        raise RuntimeError(f"no reconstructed frames found for {shot['id']}")

    model = YOLO(model_name)
    detections: list[dict] = []
    results = model.predict(
        source=[str(frame) for frame in frames],
        stream=True,
        verbose=False,
        conf=0.3,
    )
    for frame_index, result in enumerate(results):
        height, width = result.orig_shape
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            detections.append(
                {
                    "frame": Path(result.path).name,
                    "frame_index": frame_index,
                    "label": result.names[class_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "frame_size": [int(width), int(height)],
                    "bbox_norm": [
                        round(x1 / width, 6),
                        round(y1 / height, 6),
                        round(x2 / width, 6),
                        round(y2 / height, 6),
                    ],
                }
            )

    path = Path(shot["clip"]).parent / "detections.json"
    write_json(path, {"backend": "yolo", "model": model_name, "detections": detections})
    return detections


def detect_yolo_external(shot: dict, python: str, model_name: str) -> list[dict]:
    shot_dir = Path(shot["clip"]).parent
    source_root = str(Path(__file__).resolve().parents[1])
    inherited_path = os.environ.get("PYTHONPATH")
    python_path = source_root + (
        f"{os.pathsep}{inherited_path}" if inherited_path else ""
    )
    run_command(
        [
            python,
            "-m",
            "video_to_blender.detection",
            "--shot-id",
            shot["id"],
            "--clip",
            shot["clip"],
            "--model",
            model_name,
        ],
        env={**os.environ, "PYTHONPATH": python_path},
    )
    return read_json(shot_dir / "detections.json")["detections"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO for one reconstructed shot")
    parser.add_argument("--shot-id", required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    args = parser.parse_args()
    detect_yolo(
        {"id": args.shot_id, "clip": str(args.clip.resolve())},
        args.model,
    )


if __name__ == "__main__":
    main()
