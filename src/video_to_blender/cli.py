from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .domain import PipelineState, STAGES
from .io import make_run_dir, require_command
from .pipeline import Pipeline, PipelineConfig
from .tui import run_tui


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video", type=Path)
    parser.add_argument("--backend", choices=("mock", "lingbot"), default="mock")
    parser.add_argument("--scene-threshold", type=float, default=0.32)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--lingbot-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--yolo-model", default="yolo11n.pt")


def make_pipeline(args: argparse.Namespace) -> Pipeline:
    video = args.video.expanduser().resolve()
    run_dir = make_run_dir(PROJECT_ROOT, video)
    state = PipelineState(
        video=video,
        run_dir=run_dir,
        backend=args.backend,
    )
    config = PipelineConfig(
        backend=args.backend,
        scene_threshold=args.scene_threshold,
        max_frames=args.max_frames,
        lingbot_dir=args.lingbot_dir.expanduser().resolve() if args.lingbot_dir else None,
        model_path=args.model.expanduser().resolve() if args.model else None,
        python=args.python,
        yolo_model=args.yolo_model,
    )
    return Pipeline(state, config)


def generate_demo_video(path: Path) -> None:
    ffmpeg = require_command("ffmpeg")
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=0x1e3a5f:s=640x360:d=2:r=24",
        "-f",
        "lavfi",
        "-i",
        "color=c=0xb45309:s=640x360:d=2:r=24",
        "-f",
        "lavfi",
        "-i",
        "color=c=0x166534:s=640x360:d=2:r=24",
        "-filter_complex",
        (
            "[0:v]drawbox=x=120:y=100:w=180:h=160:color=0x93c5fd:t=fill,"
            "drawtext=text='SHOT 1':x=20:y=20:fontsize=32:fontcolor=white[v0];"
            "[1:v]drawbox=x=280:y=90:w=220:h=190:color=0xfed7aa:t=fill,"
            "drawtext=text='SHOT 2':x=20:y=20:fontsize=32:fontcolor=white[v1];"
            "[2:v]drawbox=x=180:y=70:w=160:h=220:color=0x86efac:t=fill,"
            "drawtext=text='SHOT 3':x=20:y=20:fontsize=32:fontcolor=white[v2];"
            "[v0][v1][v2]concat=n=3:v=1:a=0[out]"
        ),
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(path),
    ]
    subprocess.run(command, check=True)


def print_summary(pipeline: Pipeline) -> None:
    state = pipeline.state
    print("\nPipeline complete")
    print(f"  shots:      {state.shots}")
    print(f"  detections: {state.detections}")
    print(f"  objects:    {state.objects}")
    print(f"  blend:      {state.blend_file}")
    print(f"  run:        {state.run_dir}")


def run_all(pipeline: Pipeline) -> None:
    for stage in STAGES:
        print(f"[{stage.value}]")
        state = pipeline.execute(stage)
        print(f"  {state.stages[stage].summary}")
    print_summary(pipeline)


def doctor(args: argparse.Namespace) -> int:
    checks = {
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "blender": shutil.which("blender"),
        "python": shutil.which(args.python) if "/" not in args.python else args.python,
    }
    if args.lingbot_dir:
        checks["lingbot batch runner"] = str(
            args.lingbot_dir / "demo_render" / "batch_demo.py"
        )
    if args.model:
        checks["lingbot checkpoint"] = str(args.model)
    failed = False
    for label, value in checks.items():
        exists = bool(value and Path(value).exists())
        failed |= not exists
        print(f"{'OK' if exists else 'MISSING':<8} {label:<24} {value or '—'}")
    if args.python:
        probe = subprocess.run(
            [args.python, "-c", "import importlib.util as i; print(bool(i.find_spec('ultralytics')))"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        ultralytics_ok = probe.returncode == 0 and probe.stdout.strip() == "True"
        if args.lingbot_dir:
            failed |= not ultralytics_ok
        print(
            f"{'OK' if ultralytics_ok else 'MISSING':<8} "
            f"{'ultralytics':<24} selected Python"
        )
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v2b",
        description="PROTOTYPE: turn video shots into LingBot references and Blender proxies",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run an end-to-end simulated demo")
    demo.add_argument("--keep-video", action="store_true")

    run = subparsers.add_parser("run", help="run all pipeline stages")
    add_pipeline_arguments(run)

    prototype = subparsers.add_parser("prototype", help="drive the pipeline state by hand")
    add_pipeline_arguments(prototype)

    check = subparsers.add_parser("doctor", help="check the runtime")
    check.add_argument("--lingbot-dir", type=Path)
    check.add_argument("--model", type=Path)
    check.add_argument("--python", default=sys.executable)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "doctor":
        raise SystemExit(doctor(args))

    if args.command == "demo":
        demo_video = PROJECT_ROOT / ".runtime" / "demo.mp4"
        generate_demo_video(demo_video)
        args.video = demo_video
        args.backend = "mock"
        args.scene_threshold = 0.20
        args.max_frames = 12
        args.lingbot_dir = None
        args.model = None
        args.python = sys.executable
        args.yolo_model = "yolo11n.pt"

    pipeline = make_pipeline(args)
    if args.command == "prototype":
        run_tui(pipeline)
    else:
        run_all(pipeline)


if __name__ == "__main__":
    main()
