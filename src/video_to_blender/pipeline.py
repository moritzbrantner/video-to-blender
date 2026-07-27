from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .blender import assemble_blender
from .detection import detect_mock, detect_yolo_external
from .domain import Action, PipelineState, Stage, reduce
from .io import read_json, write_json
from .lifting import lift_lingbot, lift_mock
from .reconstruction import reconstruct_lingbot, reconstruct_mock
from .splitter import probe_video, split_video


@dataclass(frozen=True)
class PipelineConfig:
    backend: str = "mock"
    scene_threshold: float = 0.32
    max_frames: int = 120
    lingbot_dir: Path | None = None
    model_path: Path | None = None
    python: str = sys.executable
    yolo_model: str = "yolo11n.pt"


class Pipeline:
    def __init__(self, state: PipelineState, config: PipelineConfig):
        self.state = state
        self.config = config

    def _shots(self) -> list[dict]:
        return read_json(self.state.run_dir / "shots.json")["shots"]

    def execute(self, stage: Stage) -> PipelineState:
        try:
            self.state = reduce(
                self.state,
                Action("start", stage, f"running {stage.value}"),
            )
            summary, metrics = getattr(self, f"_{stage.value}")()
        except Exception as exc:
            self.state = reduce(self.state, Action("fail", stage, str(exc)))
            raise
        self.state = reduce(
            self.state,
            Action("complete", stage, summary, metrics),
        )
        return self.state

    def _ingest(self) -> tuple[str, dict]:
        if not self.state.video.is_file():
            raise RuntimeError(f"video does not exist: {self.state.video}")
        metadata = probe_video(self.state.video)
        write_json(self.state.run_dir / "source.json", metadata)
        return (
            f"{metadata['duration']:.1f}s, {metadata['width']}×{metadata['height']}",
            {},
        )

    def _split(self) -> tuple[str, dict]:
        shots = split_video(
            self.state.video,
            self.state.run_dir,
            self.config.scene_threshold,
        )
        return f"{len(shots)} shots", {"shots": len(shots)}

    def _reconstruct(self) -> tuple[str, dict]:
        shots = self._shots()
        frame_count = 0
        for shot in shots:
            if self.config.backend == "mock":
                result = reconstruct_mock(shot, self.config.max_frames)
            else:
                if self.config.lingbot_dir is None or self.config.model_path is None:
                    raise RuntimeError(
                        "--lingbot-dir and --model are required for the lingbot backend"
                    )
                result = reconstruct_lingbot(
                    shot,
                    self.config.lingbot_dir,
                    self.config.model_path,
                    self.config.python,
                    self.config.max_frames,
                )
            frame_count += result["frames"]
        return f"{len(shots)} point clouds, {frame_count} frames", {}

    def _detect(self) -> tuple[str, dict]:
        detections = 0
        for shot in self._shots():
            found = (
                detect_mock(shot)
                if self.config.backend == "mock"
                else detect_yolo_external(
                    shot,
                    self.config.python,
                    self.config.yolo_model,
                )
            )
            detections += len(found)
        return f"{detections} 2D detections", {"detections": detections}

    def _lift(self) -> tuple[str, dict]:
        object_count = 0
        for shot in self._shots():
            objects = (
                lift_mock(shot)
                if self.config.backend == "mock"
                else lift_lingbot(shot)
            )
            object_count += len(objects)
        return f"{object_count} merged 3D proxies", {"objects": object_count}

    def _assemble(self) -> tuple[str, dict]:
        output = assemble_blender(self.state.run_dir, self._shots())
        return output.name, {"blend_file": output}
