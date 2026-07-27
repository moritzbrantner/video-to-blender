from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def run_command(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        if len(details) > 4000:
            details = details[-4000:]
        rendered = " ".join(command)
        raise RuntimeError(
            f"command failed with exit code {exc.returncode}: {rendered}"
            + (f"\n{details}" if details else "")
        ) from exc


def require_command(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"required command is not installed: {name}")
    return executable


def make_run_dir(project_root: Path, video: Path) -> Path:
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "-" for c in video.stem)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = project_root / "runs" / f"{safe_stem}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
