# Video to Blender — pipeline prototype

> **Prototype status:** throwaway shell around a portable pipeline state model.

## The question this MVP answers

Can a video-to-Blender workflow expose useful, reviewable intermediate artifacts
instead of pretending that reconstruction produces a finished scene in one step?

This prototype makes that question tangible:

```text
video
  → detected shots + clips
  → LingBot-Map point cloud, camera poses, and depth
  → YOLO detections on the reconstructed frames
  → depth-lifted, merged 3D object proxies
  → Blender collections with point clouds and named proxy meshes
```

The point cloud is a geometric reference. The proxy objects are deliberately
simple cubes that a Blender artist can rename, resize, replace, or model over.

## Try it now

The demo path has no Python package dependencies. It generates a tiny test
video, exercises every stage with a simulated reconstruction/detector, and
asks the installed Blender to save a real `.blend`:

```bash
./v2b demo
```

Drive the same state machine by hand:

```bash
./v2b prototype /path/to/video.mp4 --backend mock
```

Use `a` to run the next stage, or the individual stage keys shown in the TUI.
The full in-memory pipeline state is redrawn after every action.

Inspect the host before a real run:

```bash
./v2b doctor --lingbot-dir /path/to/lingbot-map --model /path/to/lingbot-map.pt
```

## Run with LingBot-Map and YOLO

LingBot-Map recommends Python 3.10, PyTorch 2.8.0, CUDA 12.8, and its own
installation instructions. Its balanced checkpoint is about 4.6 GB. Set it up
outside this repository so that this wrapper stays small:

```bash
git clone https://github.com/Robbyant/lingbot-map.git /path/to/lingbot-map
cd /path/to/lingbot-map
conda create -n lingbot-map python=3.10 -y
conda activate lingbot-map
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[vis]"
pip install ultralytics
hf download robbyant/lingbot-map lingbot-map.pt \
  --local-dir /path/to/checkpoints
```

Then run:

```bash
./v2b run /path/to/video.mp4 \
  --backend lingbot \
  --lingbot-dir /path/to/lingbot-map \
  --model /path/to/checkpoints/lingbot-map.pt \
  --python /path/to/conda/envs/lingbot-map/bin/python
```

The defaults are a conservative starting point for the local 8 GB GPU: at most
120 sampled frames per shot, windowed inference, a 64-slot window, every second
frame as a keyframe, SDPA, no LingBot render video, and GLB + NPZ prediction
output. The real model has not been run on this machine yet, so begin with a
short clip and treat out-of-memory behavior as an open validation item.

## Output

Every run is placed in `runs/<video>-<timestamp>/`:

```text
source.json
shots.json
shots/
  shot_001/
    clip.mp4
    frames/
    reconstruction/
      clip/                     # LingBot per-frame NPZ predictions
      clip.glb                  # reconstructed point cloud
    detections.json             # YOLO boxes, normalized to each frame
    objects.json                # merged 3D object proxies
blender/
  build_scene.py
  scene.blend
```

Each Blender proxy has `detected_label`, `confidence`, `observations`, and
`source_shot` custom properties. Each shot is a separate collection and is
offset along X because independent reconstructions do not share a coordinate
system yet.

## What is real, and what is not

- Real: FFmpeg shot detection and clip creation.
- Real: the adapter command for LingBot-Map's `demo_render/batch_demo.py`.
- Real: YOLO inference when the selected Python environment has `ultralytics`.
- Real: lifting boxes with LingBot depth, intrinsics, and camera-to-world poses.
- Real: headless Blender assembly into a `.blend`.
- Simulated only under `--backend mock`: geometry and object observations, so
  the complete workflow can be evaluated without a multi-gigabyte checkpoint.

This is not object-quality scene reconstruction. A 2D box often contains
background depth, dynamic objects violate static-scene assumptions, and shot
reconstructions are not aligned. Those are the next research slices, not
details hidden by the MVP.

## Prototype verdict

Record what the workflow teaches us in
[`prototype/NOTES.md`](prototype/NOTES.md), then either delete the TUI shell or
absorb the validated state model into a production job runner.
