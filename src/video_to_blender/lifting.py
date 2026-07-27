from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .io import read_json, write_json


def _merge_observation(objects: list[dict], observation: dict) -> None:
    candidates = [item for item in objects if item["label"] == observation["label"]]
    nearest = None
    nearest_distance = float("inf")
    for candidate in candidates:
        distance = math.dist(candidate["center"], observation["center"])
        threshold = max(0.75, max(candidate["dimensions"]) * 0.8)
        if distance < threshold and distance < nearest_distance:
            nearest = candidate
            nearest_distance = distance

    if nearest is None:
        objects.append(
            {
                "id": f"{observation['label']}_{len(candidates) + 1:03d}",
                "label": observation["label"],
                "center": observation["center"],
                "dimensions": observation["dimensions"],
                "confidence": observation["confidence"],
                "observations": 1,
            }
        )
        return

    count = nearest["observations"]
    weight = 1 / (count + 1)
    nearest["center"] = [
        round(old * (1 - weight) + new * weight, 5)
        for old, new in zip(nearest["center"], observation["center"])
    ]
    nearest["dimensions"] = [
        round(max(old, new), 5)
        for old, new in zip(nearest["dimensions"], observation["dimensions"])
    ]
    nearest["confidence"] = round(
        max(nearest["confidence"], observation["confidence"]), 4
    )
    nearest["observations"] += 1


def lift_mock(shot: dict) -> list[dict]:
    detections = read_json(Path(shot["clip"]).parent / "detections.json")["detections"]
    shot_number = int(shot["id"].split("_")[1])
    objects: list[dict] = []
    anchor_by_label = {
        "chair": (0.8, 0.9, 0.55, 0.7, 0.7, 1.1),
        "table": (-1.0, 1.4, 0.45, 1.5, 0.9, 0.9),
        "plant": (1.8, 2.2, 0.65, 0.7, 0.7, 1.3),
    }
    for detection in detections:
        x, y, z, width, depth, height = anchor_by_label[detection["label"]]
        wobble = (detection["frame_index"] % 3 - 1) * 0.05
        _merge_observation(
            objects,
            {
                "label": detection["label"],
                "center": [x + wobble, y, z],
                "dimensions": [width, depth, height],
                "confidence": detection["confidence"],
            },
        )

    for item in objects:
        item["source_shot"] = shot["id"]
        item["mock"] = True
    write_json(
        Path(shot["clip"]).parent / "objects.json",
        {"backend": "mock", "objects": objects},
    )
    return objects


def _load_prediction_frame(predictions_dir: Path, index: int) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to lift LingBot predictions") from exc

    path = predictions_dir / f"frame_{index:06d}.npz"
    if not path.is_file():
        raise RuntimeError(f"prediction frame not found: {path}")
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def _first(data: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
    raise RuntimeError(f"none of the required prediction keys exist: {', '.join(names)}")


def _bbox_in_prediction(detection: dict, width: int, height: int) -> tuple[int, int, int, int]:
    x1n, y1n, x2n, y2n = detection["bbox_norm"]
    frame_size = detection.get("frame_size")
    if frame_size:
        source_width, source_height = frame_size
        scale = width / source_width
        resized_height = round(source_height * scale / 14) * 14
        crop_top = max(0.0, (resized_height - height) / 2)
        y1_value = y1n * source_height * scale - crop_top
        y2_value = y2n * source_height * scale - crop_top
    else:
        y1_value = y1n * height
        y2_value = y2n * height

    x1 = max(0, min(width - 1, int(x1n * width)))
    x2 = max(x1 + 1, min(width, int(x2n * width)))
    y1 = max(0, min(height - 1, int(y1_value)))
    y2 = max(y1 + 1, min(height, int(y2_value)))
    return x1, y1, x2, y2


def lift_lingbot(shot: dict) -> list[dict]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to lift LingBot predictions") from exc

    shot_dir = Path(shot["clip"]).parent
    reconstruction = read_json(shot_dir / "reconstruction" / "result.json")
    predictions_dir = Path(reconstruction["predictions"])
    detections = read_json(shot_dir / "detections.json")["detections"]
    objects: list[dict] = []

    for detection in detections:
        data = _load_prediction_frame(predictions_dir, detection["frame_index"])
        depth = np.squeeze(_first(data, "depth", "depths")).astype(float)
        intrinsic = np.squeeze(_first(data, "K", "intrinsics")).astype(float)
        c2w = np.squeeze(_first(data, "c2w", "camera_to_world")).astype(float)
        confidence = np.squeeze(data["confidence"]).astype(float) if "confidence" in data else None
        height, width = depth.shape[-2:]
        x1, y1, x2, y2 = _bbox_in_prediction(detection, width, height)

        region = depth[y1:y2, x1:x2]
        valid = np.isfinite(region) & (region > 0)
        if confidence is not None and confidence.shape == depth.shape:
            confidence_region = confidence[y1:y2, x1:x2]
            valid &= confidence_region >= np.percentile(confidence_region, 35)
        values = region[valid]
        if values.size < 10:
            continue

        z = float(np.median(values))
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        camera_point = np.array(
            [(u - cx) * z / fx, (v - cy) * z / fy, z, 1.0],
            dtype=float,
        )
        world = c2w @ camera_point
        box_width = max(0.1, (x2 - x1) * z / fx)
        box_height = max(0.1, (y2 - y1) * z / fy)
        box_depth = max(0.1, min(box_width, box_height) * 0.55)
        _merge_observation(
            objects,
            {
                "label": detection["label"],
                "center": [round(float(value), 5) for value in world[:3]],
                "dimensions": [
                    round(box_width, 5),
                    round(box_depth, 5),
                    round(box_height, 5),
                ],
                "confidence": detection["confidence"],
            },
        )

    for item in objects:
        item["source_shot"] = shot["id"]
        item["mock"] = False
    write_json(shot_dir / "objects.json", {"backend": "lingbot", "objects": objects})
    return objects
