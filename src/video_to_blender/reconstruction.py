from __future__ import annotations

import math
import random
from pathlib import Path

from .io import require_command, run_command, write_json


def _extract_frames(clip: Path, frames_dir: Path, max_frames: int) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_command("ffmpeg")
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(clip),
            "-vf",
            f"fps=2,scale=640:-2",
            "-frames:v",
            str(max_frames),
            "-q:v",
            "2",
            "-y",
            str(frames_dir / "%06d.jpg"),
        ]
    )
    return sorted(frames_dir.glob("*.jpg"))


def _write_mock_ply(path: Path, seed: int) -> None:
    random.seed(seed)
    vertices: list[tuple[float, float, float, int, int, int]] = []

    for x_index in range(-45, 46):
        for y_index in range(-35, 36):
            x = x_index / 9
            y = y_index / 9
            z = random.uniform(-0.015, 0.015)
            vertices.append((x, y, z, 95, 104, 118))

    for x_index in range(-45, 46):
        for z_index in range(0, 31):
            x = x_index / 9
            y = 3.9 + random.uniform(-0.015, 0.015)
            z = z_index / 10
            vertices.append((x, y, z, 166, 177, 193))

    for angle_index in range(720):
        angle = angle_index / 720 * math.tau
        radius = 0.9 + random.uniform(-0.04, 0.04)
        x = 0.8 + math.cos(angle) * radius
        y = 0.9 + math.sin(angle) * radius * 0.55
        z = 0.6 + random.uniform(-0.3, 0.3)
        vertices.append((x, y, z, 163, 109, 77))

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    body = [
        f"{x:.5f} {y:.5f} {z:.5f} {red} {green} {blue}"
        for x, y, z, red, green, blue in vertices
    ]
    path.write_text("\n".join([*header, *body]) + "\n")


def reconstruct_mock(shot: dict, max_frames: int) -> dict:
    shot_dir = Path(shot["clip"]).parent
    frames = _extract_frames(Path(shot["clip"]), shot_dir / "frames", max_frames)
    reconstruction_dir = shot_dir / "reconstruction"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    point_cloud = reconstruction_dir / f"{shot['id']}.ply"
    _write_mock_ply(point_cloud, seed=int(shot["id"].split("_")[1]))
    result = {
        "backend": "mock",
        "frames": len(frames),
        "point_cloud": str(point_cloud.resolve()),
        "predictions": None,
    }
    write_json(reconstruction_dir / "result.json", result)
    return result


def reconstruct_lingbot(
    shot: dict,
    lingbot_dir: Path,
    model_path: Path,
    python: str,
    max_frames: int,
) -> dict:
    batch_demo = lingbot_dir / "demo_render" / "batch_demo.py"
    if not batch_demo.is_file():
        raise RuntimeError(f"LingBot batch runner not found: {batch_demo}")
    if not model_path.is_file():
        raise RuntimeError(f"LingBot checkpoint not found: {model_path}")

    shot_dir = Path(shot["clip"]).parent
    output_dir = shot_dir / "reconstruction"
    frames_dir = shot_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python,
        str(batch_demo),
        "--video_path",
        shot["clip"],
        "--output_folder",
        str(output_dir),
        "--model_path",
        str(model_path),
        "--target_frames",
        str(max_frames),
        "--save_frames_dir",
        str(frames_dir),
        "--mode",
        "windowed",
        "--window_size",
        "64",
        "--keyframe_interval",
        "2",
        "--overlap_keyframes",
        "8",
        "--save_predictions",
        "--save_glb",
        "--no_render",
        "--use_sdpa",
    ]
    run_command(command, cwd=lingbot_dir)

    output_stem = Path(shot["clip"]).stem
    prediction_dir = output_dir / output_stem
    glb = output_dir / f"{output_stem}.glb"
    if not prediction_dir.is_dir():
        raise RuntimeError(f"LingBot did not create predictions: {prediction_dir}")
    if not glb.is_file():
        raise RuntimeError(f"LingBot did not create a GLB: {glb}")
    result = {
        "backend": "lingbot",
        "frames": len(list(frames_dir.glob("*"))),
        "point_cloud": str(glb.resolve()),
        "predictions": str(prediction_dir.resolve()),
        "command": command,
    }
    write_json(output_dir / "result.json", result)
    return result
