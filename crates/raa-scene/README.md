# raa-scene

Transforms, instances, camera and lights.

This crate owns the scene side of the engine: a hierarchy of nodes with local and
world transforms, instances (a mesh drawn with a material at a node), the camera
with an orbit controller, and the two lights the forward renderer shades with. It
produces a `RenderList` that the renderer turns into draw calls. It depends on
`raa-math` (math types) and `raa-core` (the engine-wide `MeshId` and `MaterialId`).
It compiles for the host and for `wasm32-unknown-unknown`, and needs no GPU.

## Public items

- `Transform { translation, rotation, scale }`: `IDENTITY`, `from_trs`, `to_mat4()`
  (`T * R * S`).
- `NodeId`, `InstanceId`: generational handles (Copy, Eq, Hash, Debug) with
  `to_bits() -> u64` (generation in the high half, slot index in the low half). A
  removed slot is reused with a new generation, so old handles are refused as stale.
- `Camera { eye, target, up, fov_y_rad, near, far, aspect }`: default eye (0, 2, 6)
  looking at the origin, Y up, 60 degrees, near 0.1, far 200, aspect 16/9. Methods
  `view()`, `projection()` (right-handed, depth 0..1), `view_proj()`,
  `set_viewport(width, height)`.
- `OrbitController { target, distance, yaw_rad, pitch_rad, min_distance, max_distance }`:
  `new(target, distance)`, `rotate` (pitch clamped to +-89 degrees), `zoom` (distance
  clamped), `pan` (in the camera plane, scaled by distance), `apply(&mut Camera)`.
- `DirectionalLight { direction, color, intensity }` (default normalize(-0.4, -1, -0.3),
  white, 3.0) and `Ambient { color, intensity }` (default (0.6, 0.65, 0.75), 0.3).
  Colours are linear RGB.
- `RenderItem { mesh, material, world }`, `Batch { mesh, first, count }` and
  `RenderList { items, batches }` with `with_capacity(n)` and `clear()` (keeps the
  capacity).
- `Scene { camera, light, ambient, .. }`: `new()`, `with_capacity(nodes, instances)`,
  `add_node(parent, local)`, `remove_node(id)` (removes the subtree and its
  instances), `set_local(id, t)` (marks the subtree dirty), `local(id)`, `world(id)`
  (as of the last `update_world`), `add_instance(node, mesh, material)`,
  `remove_instance(id)`, `set_instance_visible(id, visible)`,
  `set_instance_material(id, material)`, `set_instance_tag(id, tag)`,
  `instance_tag(id)`, `node_count()`, `instance_count()`, `update_world()`,
  `build_render_list(&mut RenderList)`,
  `build_render_list_tagged(&mut RenderList, &mut Vec<u32>)`.
- Instance tags: every instance carries a caller-defined `tag: u32`, 0 when it is
  added (a reused slot starts at 0 again). `set_instance_tag` and `instance_tag`
  refuse a stale id with `scene.stale_instance`. `build_render_list_tagged` gives the
  same items, batches and order as `build_render_list` (both share one private body)
  and fills `tags` so that `tags[i]` is the tag of `out.items[i]`.
- `SceneError { StaleNode, StaleInstance, ParentCycle }` with `code()`
  (`scene.stale_node`, `scene.stale_instance`, `scene.parent_cycle`), `Display` and
  `Error`. `ParentCycle` is reserved: the API has no reparenting yet.

## Frame path: no allocation

`Scene::update_world`, `Scene::build_render_list` and
`Scene::build_render_list_tagged` are part of the engine's frame
path and perform no heap allocation after warm-up:

- `update_world` walks a node order list (parents before children) that is kept on
  insert and compacted on remove, and recomputes `world = parent.world * local` only
  for dirty nodes and their subtrees. Untouched subtrees keep bit-identical worlds.
- `build_render_list` clears the output list (keeping its capacity) and fills it with
  the visible instances in (mesh, material, instance slot) order, plus one batch per
  contiguous run of the same mesh. The order is maintained when instances are added,
  removed or change material, so nothing is sorted per frame. The result is
  deterministic.
- Reuse one `RenderList` across frames and give it room for every instance, for
  example `RenderList::with_capacity(scene.instance_count())`. A list that is too
  small grows once and then stays allocation-free.
- `build_render_list_tagged` clears `tags` (keeping its capacity) and pushes one tag
  per item; reuse one `Vec<u32>` with the same capacity across frames.

Mutating calls (`add_node`, `add_instance`, `remove_*`, `set_instance_material`) may
allocate; `Scene::with_capacity` reserves room up front.

The zero-allocation claim is measured outside this crate by `raa-bench frame-allocs`
(a counting global allocator lives only in test and bench binaries).

## How to test

```sh
cargo test -p raa-scene
cargo clippy -p raa-scene --all-targets -- -D warnings
cargo fmt -p raa-scene -- --check
```

Unit tests live next to the code; integration tests are in `tests/`:
`hierarchy.rs` (world composition, dirty subtrees, stale handles, slot reuse),
`camera.rs` (camera and orbit controller), `render_list.rs` (1000 instances,
batches, ordering, capacity reuse, hidden instances) and `tags.rs` (tags follow the
sorted item order, set/get round-trip, stale refusal, removal, capacity reuse). All tests are CPU only.
