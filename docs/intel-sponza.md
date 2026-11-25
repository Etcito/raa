# Intel Sponza benchmark assets

The benchmark source is **Intel Sponza Base Scene**, download 830833 in the
[Intel GPU Research Samples Library](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html).
The archive contains Intel's glTF 2.0 export, other source formats and reference images.
Its `credits_license.txt` contains the attribution, Intel's requested publication citation,
its personal and educational use and limited commercial use notice, and the CC BY 4.0 legal text.
For publication, use the citation supplied there for Frank Meinl and Anton Kaplanyan, *Intel
Sample Library*, 2022, and identify any changes made to the scene.

## Stage the source

Run from the RAA renderer worktree with Python 3.11 or later and `curl`. The default output
directory is `out/benchmark-assets/intel-sponza`, on the worktree filesystem.

```sh
eval "$(kong env)"
python3 -B scripts/stage-intel-sponza.py
(cd out/benchmark-assets/intel-sponza && sha256sum -c SHA256SUMS)
```

The script verifies the pinned original ZIP and stages the unmodified
`main_sponza/NewSponza_Main_glTF_003.gltf`, its binary buffer, all 72 referenced textures,
and the bundled credits. The full ZIP remains alongside these files. The manifest records
the exact source hashes and scene inventory. Run the same stage command with `--offline`
to verify an existing ZIP and extraction byte for byte without connecting to Intel.

## Generate the RT proxy

The full scene has 3,747,018 triangles. The RAA mesh trace geometry limit is 250,000 triangles
per view. The separate **Intel Sponza Base — RT proxy (245k target)** benchmark uses a
245,000-triangle whole-scene budget, leaving room for additional traced geometry.

```sh
node scripts/build-intel-sponza-rt-proxy.mjs
(cd out/benchmark-assets/intel-sponza/rt-proxy-245k && sha256sum -c SHA256SUMS)
```

The script resolves the existing MeshoptSimplifier 1.3.0 from the sibling `MAMA` worktree.
It budgets triangles per primitive, simplifies geometry using normals and primary UVs, and
compacts the surviving vertices with their original attributes. It preserves the scene hierarchy,
node transforms, material assignments, images, cameras and lights. The output manifest records the
actual triangle counts for every primitive and material, any position-only simplifier fallback,
the simplifier implementation hash and the original and generated asset checksums. Use
`rt-proxy-245k/Intel_Sponza_RT_Proxy_245k.gltf` for the proxy row in a benchmark and
`main_sponza/NewSponza_Main_glTF_003.gltf` for the complete source row.

The pinned build retains 244,388 triangles, 6.5222% of the full geometry, and leaves
5,612 triangles beneath the mesh RT cap. It retains all 405 primitives, 28 materials,
72 images, 155 nodes, six cameras and 24 punctual lights. It simplifies 289 primitives
and leaves the triangle counts of 116 unchanged. The position-only fallback was unused.
