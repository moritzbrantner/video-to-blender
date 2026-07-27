from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any


class Stage(str, Enum):
    INGEST = "ingest"
    SPLIT = "split"
    RECONSTRUCT = "reconstruct"
    DETECT = "detect"
    LIFT = "lift"
    ASSEMBLE = "assemble"


STAGES = tuple(Stage)


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class StageState:
    status: Status = Status.PENDING
    summary: str = ""


@dataclass(frozen=True)
class PipelineState:
    video: Path
    run_dir: Path
    backend: str
    stages: dict[Stage, StageState] = field(
        default_factory=lambda: {stage: StageState() for stage in STAGES}
    )
    shots: int = 0
    detections: int = 0
    objects: int = 0
    blend_file: Path | None = None
    last_error: str = ""


@dataclass(frozen=True)
class Action:
    kind: str
    stage: Stage
    summary: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


def can_start(state: PipelineState, stage: Stage) -> tuple[bool, str]:
    index = STAGES.index(stage)
    if index == 0:
        return True, ""
    previous = STAGES[index - 1]
    if state.stages[previous].status != Status.COMPLETE:
        return False, f"{previous.value} must complete before {stage.value}"
    return True, ""


def next_stage(state: PipelineState) -> Stage | None:
    for stage in STAGES:
        if state.stages[stage].status != Status.COMPLETE:
            return stage
    return None


def reduce(state: PipelineState, action: Action) -> PipelineState:
    stages = dict(state.stages)

    if action.kind == "start":
        allowed, reason = can_start(state, action.stage)
        if not allowed:
            raise ValueError(reason)
        index = STAGES.index(action.stage)
        for downstream in STAGES[index:]:
            stages[downstream] = StageState()
        stages[action.stage] = StageState(Status.RUNNING, action.summary)
        return replace(
            state,
            stages=stages,
            shots=state.shots if index > STAGES.index(Stage.SPLIT) else 0,
            detections=state.detections if index > STAGES.index(Stage.DETECT) else 0,
            objects=state.objects if index > STAGES.index(Stage.LIFT) else 0,
            blend_file=None,
            last_error="",
        )

    if action.kind == "complete":
        stages[action.stage] = StageState(Status.COMPLETE, action.summary)
        return replace(
            state,
            stages=stages,
            shots=int(action.metrics.get("shots", state.shots)),
            detections=int(action.metrics.get("detections", state.detections)),
            objects=int(action.metrics.get("objects", state.objects)),
            blend_file=action.metrics.get("blend_file", state.blend_file),
            last_error="",
        )

    if action.kind == "fail":
        stages[action.stage] = StageState(Status.FAILED, action.summary)
        return replace(state, stages=stages, last_error=action.summary)

    raise ValueError(f"unknown action: {action.kind}")
