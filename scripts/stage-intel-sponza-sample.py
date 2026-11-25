"""Prepare the File > Samples edition from the verified Intel Sponza RT proxy."""
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, __version__ as pillow_version

assets = Path(__file__).resolve().parents[1] / "out/benchmark-assets/intel-sponza"
source = assets / "rt-proxy-245k"
target = assets / "web-sample"
gltf_name = "Intel_Sponza_RT_Proxy_245k.gltf"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    gltf = json.loads((source / gltf_name).read_bytes())
    target.mkdir(parents=True, exist_ok=True)
    for name in [gltf_name, "credits_license.txt", *(buf["uri"] for buf in gltf["buffers"])]:
        shutil.copyfile(source / name, target / name)
    images = []
    for entry in gltf["images"]:
        name = entry["uri"]
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source / name) as image:
            original = image.size
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            image.save(dest, format="PNG")
            images.append({"file": name, "originalSize": original, "size": image.size, "sha256": digest(dest)})
    receipt = {"edition": "Intel Sponza RT proxy, sample textures up to 1024 px",
               "gltfSha256": digest(target / gltf_name), "pillow": pillow_version, "images": images}
    (target / "sample-edition.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
