/*
MIT License

Copyright (c) 2026 SamG-Coder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Adapted from solstice@7e465b599c13da165948ebaa44df824c36c3ae07
// crates/solstice-render/shaders/vg/pass_view.wgsl:5-83,
// crates/solstice-render/shaders/common/scene.wgsl:1-160 and the `pcg` hash of
// crates/solstice-render/shaders/common/math.wgsl:30-34.
// Rewritten for the revised S32 instance record (`ranges` and `ids`) and the S29 packed
// records: no paging bits and no material struct. Declares no binding and no entry point.

// One view through the culling and rasterization passes: the main camera in either
// occlusion phase, a shadow cascade, or a voxelization volume. Twin of `PassView` in
// vg/culling.rs: 432 bytes, the shared state.
struct PassView {
    raster_view_proj: mat4x4<f32>,
    occlusion_view: mat4x4<f32>,
    planes: array<vec4<f32>, 6>,
    camera_position: vec3<f32>,
    // Largest allowed projected error, in pixels.
    lod_threshold: f32,
    viewport: vec2<f32>,
    // Projection scale of the view used for the occlusion test.
    proj_scale: vec2<f32>,
    hzb_size: vec2<f32>,
    near: f32,
    // Pixels per world unit: at unit distance if perspective, everywhere if not.
    lod_scale: f32,
    is_ortho: u32,
    // 0 tests the previous frame's depth; 1 tests this frame's.
    phase: u32,
    occlusion_enabled: u32,
    // 0: hardware only. 1: small clusters go to software. 2: every cluster does.
    allow_software: u32,
    // For an orthographic view drawn for a camera: that camera, its pixels per world unit at
    // unit distance (0 for none), and the nearest it sees anything drawn here.
    eye_position: vec3<f32>,
    eye_lod_scale: f32,
    eye_near: f32,
    // Further planes a cluster must lie inside, and how many of them are in use: for a shadow
    // view, the bounds of what it shadows, open towards the light.
    receiver_plane_count: u32,
    // Which instances the view draws: one of the INSTANCES_ constants.
    instance_filter: u32,
    // For a depth map drawn by the software rasterizer: which drawing this is, 1 to 255.
    stamp: u32,
    receiver_planes: array<vec4<f32>, 6>,
    // Towards the light, for a shadow view.
    eye_light: vec3<f32>,
}

// Twins of `InstanceFilter` in vg/culling.rs.
const INSTANCES_ALL: u32 = 0u;
// Views kept from frame to frame leave out what may move, or they would keep it where it was.
const INSTANCES_STATIC: u32 = 1u;
const INSTANCES_MOVABLE: u32 = 2u;
// The GI voxels: static instances, and movable ones that stay in them.
const INSTANCES_GI: u32 = 3u;

// Twins of the INSTANCE_ flags in vg/scene.rs, held in `VgInstance.ids.z`.
const PREVIOUS_VALID: u32 = 1u;
const DOUBLE_SIDED: u32 = 2u;
const CASTS_SHADOW: u32 = 4u;
const MOVABLE: u32 = 8u;

// Written by the cull passes. Atomic view of `CullStateRead`.
struct CullState {
    read_begin: u32,
    read_end: u32,
    write_head: atomic<u32>,
    retry_count: atomic<u32>,
    sw_count: atomic<u32>,
    hw_count: atomic<u32>,
    sw_base: u32,
    hw_base: u32,
    triangle_count: atomic<u32>,
    // Queue entries and clusters that found no room in their lists, and so were lost. Nonzero
    // means the lists are too small for this view, and the image or shadow has holes.
    dropped: atomic<u32>,
    // Entries of the visible list claimed so far, from both ends together.
    reserved: atomic<u32>,
    _pad2: u32,
}

// Read by the raster passes, after culling has finished.
struct CullStateRead {
    read_begin: u32,
    read_end: u32,
    write_head: u32,
    retry_count: u32,
    sw_count: u32,
    hw_count: u32,
    sw_base: u32,
    hw_base: u32,
    triangle_count: u32,
    dropped: u32,
    reserved: u32,
    _pad2: u32,
}

// One virtualized scene instance, 192 bytes (S32). Twin of `VgInstanceRecord` in vg/scene.rs:
// `model` @0, previous model rows @64/80/96, `quant` @112 (origin xyz, step), `uv_quant` @128
// (origin xy, step xy), `ranges` @144 (node, cluster, vertex and triangle base), `ids` @160
// (scene slot, material record, flags and hierarchy word) and `bounds` @176 (centre, radius).
struct VgInstance {
    model: mat4x4<f32>,
    prev_row0: vec4<f32>,
    prev_row1: vec4<f32>,
    prev_row2: vec4<f32>,
    quant: vec4<f32>,
    uv_quant: vec4<f32>,
    ranges: vec4<u32>,
    ids: vec4<u32>,
    bounds: vec4<f32>,
}

// One S29 packed hierarchy node, 48 bytes. `first` and `count` are relative to the instance's
// section; `is_leaf` is 0 for an interior node.
struct VgNode {
    center: vec3<f32>,
    radius: f32,
    lod_center: vec3<f32>,
    lod_radius: f32,
    max_parent_error: f32,
    first: u32,
    count: u32,
    is_leaf: u32,
}

// One S29 packed cluster, 48 bytes: `counts` is vertex_count | triangle_count << 8 | level << 16.
struct VgCluster {
    center: vec3<f32>,
    radius: f32,
    lod_center: vec3<f32>,
    lod_radius: f32,
    lod_error: f32,
    vertex_offset: u32,
    triangle_offset: u32,
    counts: u32,
}

// One S29 packed vertex, 16 bytes: a = grid x | y << 16, b = grid z | oct nx << 16,
// c = oct ny | u << 16, d = v.
struct PackedVertex {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

// One selected cluster. The index of this entry is what the visibility buffer stores.
struct VisibleCluster {
    instance: u32,
    cluster: u32,
}

// The cluster word of a visible entry: absolute cluster index in bits 0-23, then the two flags.
const VISIBLE_SOFTWARE: u32 = 0x40000000u;
const VISIBLE_SECOND: u32 = 0x80000000u;

fn visible_cluster_index(e: VisibleCluster) -> u32 {
    return e.cluster & 0xffffffu;
}

// One queued hierarchy item: `item` is a node unless the CLUSTER_BIT marks a cluster record.
struct QueueEntry {
    instance: u32,
    item: u32,
}

const CLUSTER_BIT: u32 = 0x80000000u;

// The absolute root node of the instance's section.
fn instance_root(i: VgInstance) -> u32 {
    return i.ranges.x + (i.ids.w >> 8u);
}

// The instance's hierarchy depth, in bits 0-7 of the hierarchy word.
fn instance_depth(i: VgInstance) -> u32 {
    return i.ids.w & 0xffu;
}

fn cluster_triangle_count(c: VgCluster) -> u32 {
    return (c.counts >> 8u) & 0xffu;
}

fn cluster_vertex_count(c: VgCluster) -> u32 {
    return c.counts & 0xffu;
}

fn cluster_level(c: VgCluster) -> u32 {
    return (c.counts >> 16u) & 0xffu;
}

fn transform_point(i: VgInstance, p: vec3<f32>) -> vec3<f32> {
    return (i.model * vec4<f32>(p, 1.0)).xyz;
}

fn transform_point_prev(i: VgInstance, p: vec3<f32>) -> vec3<f32> {
    let h = vec4<f32>(p, 1.0);
    return vec3<f32>(dot(i.prev_row0, h), dot(i.prev_row1, h), dot(i.prev_row2, h));
}

// The inverse-transpose of the model's upper 3x3, normalised: the cofactor matrix over the
// determinant. The rows of the inverse are the cross products of the model columns.
fn transform_normal(i: VgInstance, n: vec3<f32>) -> vec3<f32> {
    let c0 = i.model[0].xyz;
    let c1 = i.model[1].xyz;
    let c2 = i.model[2].xyz;
    let r0 = cross(c1, c2);
    let r1 = cross(c2, c0);
    let r2 = cross(c0, c1);
    let det = dot(c0, r0);
    if abs(det) < 1.0e-20 {
        return normalize(n);
    }
    let cofactor = vec3<f32>(
        dot(vec3<f32>(r0.x, r1.x, r2.x), n),
        dot(vec3<f32>(r0.y, r1.y, r2.y), n),
        dot(vec3<f32>(r0.z, r1.z, r2.z), n),
    );
    return normalize(cofactor * (1.0 / det));
}

fn decode_position(i: VgInstance, v: PackedVertex) -> vec3<f32> {
    let grid = vec3<f32>(f32(v.a & 0xffffu), f32(v.a >> 16u), f32(v.b & 0xffffu));
    return i.quant.xyz + grid * i.quant.w;
}

// Signed octahedral coordinates packed as two i16 words scaled by 32767 (S29 dag.rs:518-533).
// `pack_vertex` puts normal x in the high half of `b` and normal y in the low half of `c`; both
// are sign-extended. Reading x from the low half of `b` would decode the grid z instead.
fn decode_normal(v: PackedVertex) -> vec3<f32> {
    let e = vec2<f32>(
        f32(i32(v.b) >> 16u),
        f32(i32(v.c << 16u) >> 16u),
    ) / 32767.0;
    var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if n.z < 0.0 {
        let sx = select(-1.0, 1.0, n.x >= 0.0);
        let sy = select(-1.0, 1.0, n.y >= 0.0);
        let folded = (vec2<f32>(1.0) - abs(n.yx)) * vec2<f32>(sx, sy);
        n = vec3<f32>(folded.x, folded.y, n.z);
    }
    return normalize(n);
}

fn decode_uv(i: VgInstance, v: PackedVertex) -> vec2<f32> {
    return i.uv_quant.xy + vec2<f32>(f32(v.c >> 16u), f32(v.d & 0xffffu)) * i.uv_quant.zw;
}

// The largest axis scale of the model, which bounds a local radius in world space.
fn instance_max_scale(i: VgInstance) -> f32 {
    return max(length(i.model[0].xyz), max(length(i.model[1].xyz), length(i.model[2].xyz)));
}

fn instance_two_sided(i: VgInstance) -> bool {
    return (i.ids.z & DOUBLE_SIDED) != 0u;
}

// PCG hash: a well-distributed 32-bit integer hash (solstice common/math.wgsl).
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
