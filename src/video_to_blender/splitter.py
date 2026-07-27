from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .io import require_command, run_command, write_json


PTS_TIME = re.compile(r"pts_time:([0-9.]+)")


def probe_video(video: Path) -> dict:
    ffprobe = require_command("ffprobe")
    result = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(video),
        ]
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return {
        "path": str(video.resolve()),
        "duration": float(data["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream["avg_frame_rate"],
    }


def detect_cuts(video: Path, threshold: float, duration: float) -> list[float]:
    ffmpeg = require_command("ffmpeg")
    command = [
        ffmpeg,
        "-hide_banner",
        "-i",
        str(video),
        "-filter:v",
        f"select='gt(scene,{threshold})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    cuts = [float(match.group(1)) for match in PTS_TIME.finditer(result.stderr)]
    return sorted({cut for cut in cuts if 0.15 < cut < duration - 0.15})


def split_video(video: Path, run_dir: Path, threshold: float) -> list[dict]:
    metadata = probe_video(video)
    duration = metadata["duration"]
    cuts = detect_cuts(video, threshold, duration)
    boundaries = [0.0, *cuts, duration]
    ffmpeg = require_command("ffmpeg")
    shots: list[dict] = []

    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        shot_id = f"shot_{index:03d}"
        shot_dir = run_dir / "shots" / shot_id
        shot_dir.mkdir(parents=True, exist_ok=True)
        clip = shot_dir / "clip.mp4"
        run_command(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.6f}",
                "-i",
                str(video),
                "-t",
                f"{end - start:.6f}",
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(clip),
            ]
        )
        shots.append(
            {
                "id": shot_id,
                "start": round(start, 6),
                "end": round(end, 6),
                "duration": round(end - start, 6),
                "clip": str(clip.resolve()),
            }
        )

    write_json(run_dir / "shots.json", {"threshold": threshold, "shots": shots})
    return shots

