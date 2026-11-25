#!/usr/bin/env python3
"""Stage and verify Intel's original Sponza Base Scene glTF package."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile


SOURCE = "https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html"
DOWNLOAD = "https://cdrdv2.intel.com/v1/dl/getContent/830833"
ARCHIVE = "main_sponza.zip"
ARCHIVE_BYTES = 3_987_608_266
ARCHIVE_SHA256 = "b8bb853884ab1566b3beb35666bd09882a4e0dc16661e4684e103792cf0229b9"
SCENE = PurePosixPath("main_sponza/NewSponza_Main_glTF_003.gltf")
CREDITS = SCENE.with_name("credits_license.txt")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "out/benchmark-assets/intel-sponza"
BLOCK_BYTES = 1024 * 1024


def checked_archive(root, offline):
    final = root / ARCHIVE
    partial = final.with_suffix(".zip.partial")
    if not final.exists():
        if offline:
            raise ValueError(f"Missing verified download at {final}")
        if not partial.exists() or partial.stat().st_size != ARCHIVE_BYTES:
            subprocess.run(
                [
                    "curl", "--proto", "=https", "--proto-redir", "=https",
                    "--fail", "--location", "--show-error", "--silent", "--retry", "3",
                    "--continue-at", "-", "--max-filesize", str(ARCHIVE_BYTES),
                    "--output", str(partial), DOWNLOAD,
                ],
                check=True,
            )
    source = final if final.exists() else partial
    with source.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if source.stat().st_size != ARCHIVE_BYTES or checksum != ARCHIVE_SHA256:
        raise ValueError(f"Intel archive did not match the pinned download: {source}, SHA-256 {checksum}")
    if source == partial:
        partial.rename(final)
    return final


def external_member(uri):
    parsed = urlsplit(uri)
    decoded = unquote(parsed.path, errors="strict")
    path = PurePosixPath(decoded)
    if (parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
            or path.is_absolute() or not path.parts or ".." in path.parts
            or any(ord(char) < 32 or char == "\\" for char in decoded)):
        raise ValueError(f"Invalid local glTF resource URI: {uri}")
    return SCENE.parent / path


def inventory(collection, scene):
    materials = scene["materials"]
    meshes = scene["meshes"]
    nodes = scene["nodes"]
    accessors = scene["accessors"]
    primitives = [primitive for mesh in meshes for primitive in mesh["primitives"]]
    if any(primitive.get("mode", 4) != 4 for primitive in primitives):
        raise ValueError("Expected triangle-list primitives in the pinned Base Scene")
    mesh_triangles = [
        sum(accessors[p["indices"]]["count"] // 3 for p in mesh["primitives"])
        for mesh in meshes
    ]
    references = {SCENE, CREDITS}
    for buffer in scene["buffers"]:
        member = external_member(buffer["uri"])
        if collection.getinfo(str(member)).file_size != buffer["byteLength"]:
            raise ValueError(f"Buffer size disagrees with glTF: {member}")
        references.add(member)
    references.update(external_member(image["uri"]) for image in scene["images"])
    return sorted(references), {
        **{key: len(scene[key]) for key in (
            "scenes", "nodes", "meshes", "materials", "buffers", "images", "textures", "cameras"
        )},
        "primitives": len(primitives),
        "meshNodes": sum("mesh" in node for node in nodes),
        "meshTriangles": sum(mesh_triangles),
        "instancedTriangles": sum(mesh_triangles[node["mesh"]] for node in nodes if "mesh" in node),
        "pbrTextureMaterials": {
            **{key: sum(key in mat["pbrMetallicRoughness"] for mat in materials) for key in (
                "baseColorTexture", "metallicRoughnessTexture"
            )},
            **{key: sum(key in mat for mat in materials) for key in (
                "normalTexture", "occlusionTexture", "emissiveTexture"
            )},
        },
        "alphaModes": dict(Counter(mat.get("alphaMode", "OPAQUE") for mat in materials)),
        "extensionsUsed": scene.get("extensionsUsed", []),
        "punctualLights": len(scene.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])),
    }


def file_records(collection, paths, root, stage):
    published = (root / SCENE.parent).exists()
    records = {}
    for member in paths:
        path = str(member)
        digest = hashlib.sha256()
        with collection.open(path) as source:
            if published:
                expected = hashlib.file_digest(source, "sha256").hexdigest()
                with (root / path).open("rb") as existing:
                    if hashlib.file_digest(existing, "sha256").hexdigest() != expected:
                        raise ValueError(f"Existing asset disagrees with the Intel archive: {root / path}")
                checksum = expected
            else:
                target = stage / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    while block := source.read(BLOCK_BYTES):
                        digest.update(block)
                        output.write(block)
                checksum = digest.hexdigest()
        records[path] = {"sha256": checksum, "bytes": collection.getinfo(path).file_size}
    if not published:
        (stage / SCENE.parent).rename(root / SCENE.parent)
    return records


def stage_package(root, offline):
    root.mkdir(parents=True, exist_ok=True)
    archive = checked_archive(root, offline)
    with ZipFile(archive) as collection, TemporaryDirectory(prefix=".ingest-", dir=root) as temp:
        stage = Path(temp)
        scene = json.loads(collection.read(str(SCENE)))
        notice = collection.read(str(CREDITS)).decode("utf-8-sig")
        if scene["asset"]["version"] != "2.0" or "Creative Commons Attribution 4.0" not in notice:
            raise ValueError("Expected the Base Scene glTF 2.0 and Intel's bundled credits and license")
        paths, summary = inventory(collection, scene)
        records = file_records(collection, paths, root, stage)
        manifest = {
            "asset": "Intel Sponza Base Scene (Sponza 2022)",
            "source": SOURCE,
            "download": DOWNLOAD,
            "archive": {"path": ARCHIVE, "sha256": ARCHIVE_SHA256, "bytes": ARCHIVE_BYTES},
            "scene": str(SCENE),
            "creditsAndLicense": str(CREDITS),
            "sceneInventory": summary,
            "files": records,
        }
        (stage / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
        hashes = {ARCHIVE: ARCHIVE_SHA256, **{path: record["sha256"] for path, record in records.items()}}
        (stage / "SHA256SUMS").write_bytes(
            "".join(f"{checksum}  {path}\n" for path, checksum in sorted(hashes.items())).encode()
        )
        for receipt in ("manifest.json", "SHA256SUMS"):
            (stage / receipt).replace(root / receipt)
    print(f"Verified {len(records)} files and the original archive in {root}")
    print(root / SCENE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Asset directory on disk")
    parser.add_argument("--offline", action="store_true", help="Require the ZIP to be present locally")
    args = parser.parse_args()
    stage_package(args.output.resolve(), args.offline)
