#!/usr/bin/env python3
"""Stage the full Intel Sponza for File > Samples."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from PIL import Image, __version__ as pillow_version

ASSETS = Path(__file__).resolve().parents[1] / "out/benchmark-assets/intel-sponza"
TEXTURE_LIMIT = 1024
CREDITS = "credits_license.txt"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_verified(root, filename, record, stage):
    source = root / filename
    if source.stat().st_size != record["bytes"] or digest(source) != record["sha256"]:
        raise ValueError(f"Source asset disagrees with its manifest: {source}")
    shutil.copyfile(source, stage / filename)
    return {"file": filename, **record}


def stage_samples(assets, output):
    source_manifest = json.loads((assets / "manifest.json").read_bytes())
    scene = Path(source_manifest["scene"])
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".sponza-samples-", dir=output.parent) as temp:
        stage = Path(temp) / "samples"
        stage.mkdir()
        root = assets / scene.parent
        records = source_manifest["files"]

        def record(filename):
            return records[(scene.parent / filename).as_posix()]

        credits = copy_verified(root, CREDITS, record(CREDITS), stage)
        gltf_record = copy_verified(root, scene.name, record(scene.name), stage)
        gltf = json.loads((stage / scene.name).read_bytes())
        buffers = [
            copy_verified(root, item["uri"], record(item["uri"]), stage)
            for item in gltf["buffers"]
        ]
        images = []
        for entry in gltf["images"]:
            filename = entry["uri"]
            original = root / filename
            source_record = record(filename)
            if original.stat().st_size != source_record["bytes"] or digest(original) != source_record["sha256"]:
                raise ValueError(f"Source texture disagrees with its manifest: {original}")
            dest = stage / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(original) as image:
                size = image.size
                image.thumbnail((TEXTURE_LIMIT, TEXTURE_LIMIT), Image.Resampling.LANCZOS)
                image.save(dest, format="PNG")
                images.append({"file": filename, "originalSize": size, "size": image.size,
                               "sourceSha256": source_record["sha256"], "sha256": digest(dest)})

        data = {
            "edition": "Intel Sponza Base Scene, unchanged geometry",
            "triangles": source_manifest["sceneInventory"]["instancedTriangles"],
            "gltfSha256": gltf_record["sha256"],
            "buffers": buffers,
            "credits": credits,
            "pillow": pillow_version,
            "images": images,
        }
        (stage / "sample-edition.json").write_bytes((json.dumps(data, indent=2) + "\n").encode())
        stage.rename(output)
        print(f"Staged full Intel Sponza with {len(images)} textures at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stage_samples(args.assets.resolve(), (args.output or args.assets / "web-sample-visual").resolve())
