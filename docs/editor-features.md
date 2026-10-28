# Editor rendering features

The public engine now supports polygon edges, instance selection and orthographic
cameras. Mesh replacement already existed as `updateMesh`; it is reused without
introducing another mesh identity or a remove/add interval.

## JavaScript API

The methods are native wasm-bindgen `Engine` methods returned by
`loadRaa().createEngine(canvas, options)`. The additions are declared in
`packages/raa/index.d.ts`; the existing mesh API is in
`packages/raa/types/m07-meshes-hdr.d.ts`. Released in 0.4.0.

### Replace geometry

```ts
updateMesh(mesh: number, desc: MeshDesc): void
// MeshDesc: positions, normals, indices, optional uvs, tangents, colors,
// subMeshes: { first, count, material? }[], and now edges?: Uint32Array
```

```js
engine.updateMesh(meshId, {
  positions: nextPositions, normals: nextNormals, indices: nextIndices,
  subMeshes: [{ first: 0, count: nextIndices.length, material: materialId }],
  edges: polygonEdges,
});
```

All instances using the handle follow the replacement. CPU validation, including
the optional edge list and material handles, precedes upload. The buffers are
replaced under the existing internal mesh identity; affected instance groups are
rebuilt before the next frame is prepared. Nodes, visibility, instance handles,
selection colours and wireframe overrides are preserved. A rejected descriptor
keeps the previous mesh. Changed index/submesh counts are supported.

Cost: new geometry buffers at edit time, CPU edge extraction and one group rebuild
per affected instance at the next frame. There is no remove/add frame and no
additional rendering pass. Superseded buffers follow WebGPU resource lifetime
rules. Replacement also invalidates the cached edge buffer.

### Wireframe

```ts
type WireframeMode = 'off' | 'overlay' | 'only';
setWireframe(mode: WireframeMode | 'inherit', instance?: number): void
// MeshDesc.edges?: Uint32Array — pairs of vertex indices
```

```js
const mesh = engine.addMesh({
  positions, normals, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  edges: new Uint32Array([0, 1, 1, 2, 2, 3, 3, 0]), // quad, no diagonal
});
engine.setWireframe('overlay');                 // global default
engine.setWireframe('only', selectedInstance); // per-instance override
engine.setWireframe('off', shadedInstance);    // override the global default
engine.setWireframe('inherit', selectedInstance); // restore global default
```

An omitted/null edge list uses unique undirected triangle-index edges. An empty
typed array deliberately draws no edges. Odd lengths, non-Uint32Array values and
out-of-range indices are refused before replacing geometry. The list is replaced,
not merged, by every `updateMesh`.

Each edge is expanded in the vertex shader into a six-vertex screen-space quad.
Its nominal stroke is about 1.5 backing-store pixels, with a smooth coverage ramp
out to 2.5 pixels total. Near-plane crossings are clipped before dividing by w.
Fragment depth is compared with the existing opaque surface prepass, with a
small capped slope-scaled tolerance. Hidden edges are discarded; this is not x-ray
wireframe. The width does not depend on object scale, distance or projection.

Edges draw once per mesh instance, including meshes with several submeshes.
With GPU skinning enabled the wireframe reads each instance's posed edge segments,
so it follows the skinned pose; disabling skinning returns it to the rest edges in
the same frame. `only` suppresses shaded surface draws but retains the surface depth prepass,
so back edges remain hidden. A background-colour fill in the display overlay
removes the surface tone-mapping footprint; the line colour switches to a
contrasting light colour on dark backgrounds. Built-in opaque, transparency and fur surface
submission respect this mode. Rust contributors drawing additional mesh surfaces
should call `LitHandle::draw_surface_instances` to respect instance overrides.

Cost when active: one display overlay render pass, six vertex invocations per
edge per enabled instance and covered edge fragments. Edge vertex buffers are
created lazily on first wireframe use; CPU edge segments are retained with meshes.
An active editor uploads 32 bytes of settings per prepared submesh-instance.
Edges-only may split instanced surface batches to retain instances with overrides,
and adds a flat background draw for each affected visible submesh. It uses the
configured flat background rather than reconstructing a procedural sky behind objects.

### Selection outline

```ts
setSelectionOutline(instance: number, desc: { color?: Vec3 } | null): void
```

```js
engine.setSelectionOutline(instanceA, { color: [1, 0.7, 0.1] });
engine.setSelectionOutline(instanceB, { color: [0.15, 0.85, 1] });
engine.setSelectionOutline(instanceA, null); // independent of instanceB
```

Colours are display sRGB in [0, 1]; the default is [1, 0.6, 0.05].
Instance handles are validated before mutation. Removing an instance or its node
clears its outline and wireframe override.

Selected triangles write a full-resolution RGBA8 colour mask after a strict
depth comparison with the surface prepass, respecting material face culling. The
mask draws the instance's posed mesh view while skinning is live, so the selection
outline follows the skinned pose; disabling skinning returns it to the rest mesh
in the same frame. A fullscreen shader dilates the visible mask
with a circular kernel, omits its interior and blends an antialiased outline
about 2–3 backing-store pixels wide into the final canvas. Multiple selected
instances share one mask; where their visible masks meet they form a union.

**Hidden portions are omitted.** This keeps occluders readable and avoids implying
that a hidden surface is visible. Partly occluded selections still receive an
outline around the visible mask. Dilation can extend a few pixels across an
occlusion boundary; it never draws the hidden object's whole contour.

Cost when selected: one RGBA8 texture (4 × canvas width × height bytes), one
mask pass over selected submeshes, and one fullscreen dilation/blend in the same
display overlay pass used by wireframe (49 mask taps per background fragment).
Pipelines, bind groups and the settings buffer are cached and rebuilt on resize
or capacity growth. The current resource bundle also allocates the mask when
wireframe alone is first enabled.

### Camera

```ts
setCamera({
  eye?: Vec3, position?: Vec3, target: Vec3, up?: Vec3,
  projection?: 'perspective' | 'orthographic',
  fovYDeg?: number, orthoHeight?: number, near?: number, far?: number,
}): void
resize(width: number, height: number): void
```

```js
engine.setCamera({
  position: [4, 3, 6], target: [0, 0, 0], up: [0, 1, 0],
  projection: 'orthographic', orthoHeight: 8, near: 0.05, far: 200,
});
engine.resize(canvas.clientWidth, canvas.clientHeight);
engine.setCamera({
  eye: [4, 3, 6], target: [0, 0, 0], projection: 'perspective',
  fovYDeg: 50, near: 0.05, far: 200,
});
```

Previously accepted: eye/target/up, vertical FOV, near/far, automatic aspect.
Added: orthographic projection, vertical orthographic span, `position` alias
(takes precedence over `eye`), rejection of a zero/parallel up vector, and an
unchanged-size resize fast path. At least one of eye/position is required.
Projection defaults to perspective; orthoHeight defaults to 10 metres.
The camera descriptor is complete, not a patch: omitted fields use defaults.
The aspect always follows the backing-store width/height, including after
projection switches. Both projections use right-handed view space and [0, 1]
clip depth. Orthographic near may be zero; perspective near must be positive.
Zero resize dimensions clamp to one.

Cost: projection matrix math only; no added GPU pass. Pixel widths refer to the
backing store: size it to CSS dimensions for CSS-pixel widths, or account for DPR.

## Disabled cost and pass ordering

With global wireframe off, no enabled instance override and no selected instance,
the editor records **zero GPU passes, draws, uploads or texture/buffer allocations**.
After use, its cached allocations remain until resize/reuse or engine disposal.
A small CPU state check remains. The existing instance stream uses four formerly
reserved padding bytes for the instance tag; its size and upload count do not grow.

The pipeline has an `editor` entry at order 210. It returns before recording when
inactive. `FeaturePass::output_overlay()` defaults to false; a true value marks a
canvas overlay that must follow every HDR pass and tonemap. Normal HDR passes
still cannot be reordered after tonemap. This prevents tone mapping, exposure,
DOF or background substitution from changing selection colours or silhouettes.
The disabled path uses the original surface draw batching. Existing draw-call
statistics count prepared surface batches, not the additional editor draws.

## Demo and captures

The demo is `examples/editor/` (JavaScript only). It loads an external source asset through the actual
`@etcito/raa-sdfm` loader, including its materials. No asset, mesh binary or
dependency bundle is committed. The tested source fixture is `candy.spinner-bar`.
Two instances share its mesh; the replacement timer updates that mesh's positions
and normals under the same id. A generated unselected quad supplies an explicit
four-edge list and occludes part of the selected object. Its triangle diagonal
must never appear.

Use a MAMA checkout with its existing browser build and installed dependencies.
Commands below build only the package required by this page and stage local
inputs into ignored output. Each build/capture subprocess has a short timeout;
on a cold cache rerun a timed-out build to reuse completed dependencies.

```sh
timeout 55 cargo build -p raa-js --target wasm32-unknown-unknown
timeout 55 wasm-bindgen --target web --out-dir packages/raa/pkg target/wasm32-unknown-unknown/debug/raa_js.wasm
node scripts/build-editor-demo.mjs "$MAMA_CHECKOUT" "$SOURCE_ASSET_JSON"
node web/serve.mjs dist/editor --port 8080
# Open http://127.0.0.1:8080/ ; stop the server by its PID afterwards.
```

```sh
timeout 58 node scripts/capture-editor.mjs
```

The capture command starts its own server and headless Chrome via `nmc-chrome`,
then stops both by their recorded child PIDs in `finally`. It defaults to
SwiftShader through Vulkan; override the JSON array `EDITOR_CHROME_FLAGS` to
test another backend. This environment's default ANGLE/SwiftShader route also
loses its device in a standalone JavaScript WebGPU clear; the Vulkan route is
used for the proof. `CHROME_LAUNCHER` can select another compatible launcher.

The JSON record is [proofs/results/editor.json](../proofs/results/editor.json).
It records adapter/hardware status, nonempty probes, GPU pass labels, source
mesh identity, replacement phases and asynchronous validation/device errors.
Captures are ignored PNG review artifacts in `proofs/captures/`:

- [editor-overlay-perspective.png](../proofs/captures/editor-overlay-perspective.png)
- [editor-only-perspective.png](../proofs/captures/editor-only-perspective.png)
- [editor-off-perspective.png](../proofs/captures/editor-off-perspective.png) (outlines still enabled)
- [editor-overlay-orthographic.png](../proofs/captures/editor-overlay-orthographic.png)
- [editor-disabled-perspective.png](../proofs/captures/editor-disabled-perspective.png) (wireframe and selection both off)
- [editor-overlay-perspective-replaced.png](../proofs/captures/editor-overlay-perspective-replaced.png)

All six captures were opened and inspected. Edges follow the shaded faces and
stop behind the grey occluder. The quad shows four boundary edges without its
triangle diagonal. Edges-only has no shaded or grey-filled surface. Yellow and
cyan outlines follow the visible contours in both projections, with no contour
showing through the occluder. The replacement pair shows both shared instances
expand while keeping mesh id 1; the final record reports eight replacements,
three public instances and no GPU errors. The fully disabled variant records
only prepass, forward and tonemap (zero editor passes). Both projections also
complete 1280×600 → 960×540 → 1280×600 resize cycles, including repeated identical
resize calls.

The capture script refuses an empty frame or GPU error and asserts that the
fully disabled variant records no editor render pass. The staging script copies the source asset unchanged. The adapter checks its hash
with MAMA's `serializeDocument(document)` UTF-8 bytes.

## Validation and remaining work

Validation completed: 210 targeted native tests across the touched crates and
demo crates; native Clippy with warnings denied; WASM Clippy with warnings denied;
the raa-js WASM build and wasm-bindgen package generation; and a TypeScript usage
check against the staged, merged declarations. No workspace check suite ran.

Targeted tests cover orthographic span/clip depth/resize, explicit quad edges
without a diagonal, wireframe override precedence, multiple outline colours,
shader validation, actual native GPU overlay draws, edges-only suppression and
pixel-identical disabled rendering. Existing mesh replacement/group tests cover
submesh count changes and preserving nodes, visibility and tags. Pipeline tests
cover the new display-overlay ordering rule.

The proof uses **software WebGPU (SwiftShader)**. It establishes visual behaviour,
not hardware performance, startup budgets or release WASM size. The demo package
is a development build; no archive, version bump, commit, push or release is made.

Open limits:
- Overlay geometry follows rigid node transforms and GPU-skinned poses. Material-defined
  vertex deformation and squash/jiggle are not replayed by this independent
  overlay shader; those silhouettes still need shared vertex composition.
  Fur outlines follow the polygon base rather than the outer shell envelope.
- Transparent objects do not occlude overlays as solid objects do; the depth
  prepass intentionally represents opaque surfaces.
- The outline mask is single-sampled, with antialiasing on the dilation boundary;
  hardware/MSAA edge cases and extreme camera near-plane cases need further review.
- Camera-dependent optional post effects were disabled for the polygon proof.
  Their orthographic assumptions have not been audited.
- The source adapter preserves optional MAMA polygon edge lists, including material-local
  vertex offsets; older meshes without them use triangle edges.
