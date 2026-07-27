from __future__ import annotations

import json
from pathlib import Path

from .io import require_command, run_command, write_json


BLENDER_SCRIPT = r'''
import argparse
import json
import sys
from pathlib import Path

import bpy


def args_after_double_dash():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(values)


def material_for(label):
    palette = {
        "chair": (0.19, 0.55, 0.91, 0.42),
        "table": (0.95, 0.48, 0.18, 0.42),
        "plant": (0.20, 0.76, 0.42, 0.42),
    }
    color = palette.get(label, (0.86, 0.31, 0.52, 0.42))
    name = f"Detected::{label}"
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Alpha"].default_value = color[3]
        bsdf.inputs["Roughness"].default_value = 0.55
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    elif hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
    return material


def move_selected_to(collection):
    for obj in list(bpy.context.selected_objects):
        for existing in list(obj.users_collection):
            existing.objects.unlink(obj)
        collection.objects.link(obj)


def import_reference(path, collection):
    suffix = path.suffix.lower()
    before = set(bpy.data.objects)
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(path))
    else:
        raise RuntimeError(f"unsupported reconstruction format: {suffix}")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    for obj in imported:
        for existing in list(obj.users_collection):
            existing.objects.unlink(obj)
        collection.objects.link(obj)
        obj.name = f"Reference::{obj.name}"


def proxy_transform(center, dimensions, reference_path):
    if reference_path.suffix.lower() in {".glb", ".gltf"}:
        # glTF is Y-up; Blender's importer maps it to Z-up.
        return (
            (center[0], -center[2], center[1]),
            (dimensions[0], dimensions[2], dimensions[1]),
        )
    return (tuple(center), tuple(dimensions))


def main():
    args = args_after_double_dash()
    manifest = json.loads(Path(args.manifest).read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.name != "Collection":
            bpy.data.collections.remove(collection)

    root = bpy.data.collections.new("Video Reconstruction")
    bpy.context.scene.collection.children.link(root)

    for shot_index, shot in enumerate(manifest["shots"]):
        collection = bpy.data.collections.new(shot["id"])
        root.children.link(collection)
        offset = (shot_index * manifest["shot_spacing"], 0.0, 0.0)
        reference_path = Path(shot["point_cloud"])
        import_reference(reference_path, collection)
        for obj in collection.objects:
            if obj.parent is None:
                obj.location.x += offset[0]

        for proxy in shot["objects"]:
            center, dimensions = proxy_transform(
                proxy["center"],
                proxy["dimensions"],
                reference_path,
            )
            bpy.ops.mesh.primitive_cube_add(
                location=(
                    center[0] + offset[0],
                    center[1] + offset[1],
                    center[2] + offset[2],
                )
            )
            obj = bpy.context.object
            obj.name = f"Detected::{proxy['id']}"
            obj.dimensions = dimensions
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            obj.data.materials.append(material_for(proxy["label"]))
            obj.display_type = "WIRE"
            obj.show_name = True
            obj["detected_label"] = proxy["label"]
            obj["confidence"] = proxy["confidence"]
            obj["observations"] = proxy["observations"]
            obj["source_shot"] = shot["id"]
            for existing in list(obj.users_collection):
                existing.objects.unlink(obj)
            collection.objects.link(obj)

    bpy.context.scene["pipeline"] = "video-to-blender prototype"
    bpy.context.scene["source_video"] = manifest["source_video"]
    bpy.context.scene["coordinate_note"] = (
        "Each shot has an independent LingBot-Map coordinate system and is offset on X."
    )
    try:
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print(f"Saved Blender scene to {args.output}")


main()
'''


def assemble_blender(run_dir: Path, shots: list[dict], spacing: float = 12.0) -> Path:
    blender = require_command("blender")
    output_dir = run_dir / "blender"
    output_dir.mkdir(parents=True, exist_ok=True)
    script = output_dir / "build_scene.py"
    script.write_text(BLENDER_SCRIPT)

    manifest_shots = []
    for shot in shots:
        shot_dir = Path(shot["clip"]).parent
        reconstruction = json.loads(
            (shot_dir / "reconstruction" / "result.json").read_text()
        )
        objects = json.loads((shot_dir / "objects.json").read_text())["objects"]
        manifest_shots.append(
            {
                "id": shot["id"],
                "point_cloud": reconstruction["point_cloud"],
                "objects": objects,
            }
        )

    source = json.loads((run_dir / "source.json").read_text())
    manifest = {
        "source_video": source["path"],
        "shot_spacing": spacing,
        "shots": manifest_shots,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    output = output_dir / "scene.blend"
    run_command(
        [
            blender,
            "--background",
            "--factory-startup",
            "--python",
            str(script),
            "--",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
        ]
    )
    if not output.is_file():
        raise RuntimeError("Blender exited without creating the .blend file")
    return output
