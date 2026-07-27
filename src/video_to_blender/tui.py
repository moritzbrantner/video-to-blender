from __future__ import annotations

import json
from pathlib import Path

from .domain import PipelineState, STAGES, Stage, Status, next_stage
from .pipeline import Pipeline


BOLD = "\x1b[1m"
DIM = "\x1b[2m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
RESET = "\x1b[0m"

KEYS = {
    "i": Stage.INGEST,
    "s": Stage.SPLIT,
    "r": Stage.RECONSTRUCT,
    "d": Stage.DETECT,
    "l": Stage.LIFT,
    "b": Stage.ASSEMBLE,
}


def _status(value: Status) -> str:
    colors = {
        Status.COMPLETE: GREEN,
        Status.RUNNING: YELLOW,
        Status.FAILED: RED,
        Status.PENDING: DIM,
    }
    return f"{colors[value]}{value.value:<8}{RESET}"


def render(state: PipelineState) -> None:
    print("\033[2J\033[H", end="")
    print(f"{BOLD}VIDEO → BLENDER / PIPELINE PROTOTYPE{RESET}")
    print(f"{DIM}State is in memory; artifacts are written under the run directory.{RESET}\n")
    print(f"{BOLD}video{RESET}       {state.video}")
    print(f"{BOLD}run_dir{RESET}     {state.run_dir}")
    print(f"{BOLD}backend{RESET}     {state.backend}")
    print(f"{BOLD}shots{RESET}       {state.shots}")
    print(f"{BOLD}detections{RESET}  {state.detections}")
    print(f"{BOLD}objects{RESET}     {state.objects}")
    print(f"{BOLD}blend_file{RESET}  {state.blend_file or '—'}")
    if state.last_error:
        print(f"{BOLD}{RED}last_error{RESET}  {state.last_error}")
    print(f"\n{BOLD}Stages{RESET}")
    for stage in STAGES:
        item = state.stages[stage]
        print(f"  {stage.value:<12} {_status(item.status)} {DIM}{item.summary}{RESET}")
    print(
        f"\n{BOLD}[a]{RESET} advance  "
        f"{BOLD}[i]{RESET} ingest  {BOLD}[s]{RESET} split  "
        f"{BOLD}[r]{RESET} reconstruct  {BOLD}[d]{RESET} detect  "
        f"{BOLD}[l]{RESET} lift  {BOLD}[b]{RESET} blender  "
        f"{BOLD}[q]{RESET} quit"
    )


def run_tui(pipeline: Pipeline) -> None:
    while True:
        render(pipeline.state)
        key = input(f"\n{BOLD}>{RESET} ").strip().lower()[:1]
        if key == "q":
            return
        stage = next_stage(pipeline.state) if key == "a" else KEYS.get(key)
        if stage is None:
            continue
        try:
            pipeline.execute(stage)
        except Exception:
            pass

