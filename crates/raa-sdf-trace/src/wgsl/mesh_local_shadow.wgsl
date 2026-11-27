// Cached cube shadow maps of local lights (`shadow.local`, mesh/local_shadow.rs): one invocation per
// texel of one cube face traces the shared world-space mesh BVH (the DDGI tracer's) from the light
// and stores the distance from the light to the first occluder. Rays start `near` metres from the
// light (its emitter radius: the light's own housing, a lantern's bulb and glass, casts nothing)
// and stop at its range; a miss stores 1e30. The forward reads the faces with textureLoad
// (raa-shader local_lighting.wgsl `raa_local_shadow`), whose face mapping this file inverts.
//
// Group 0: 0 params, 1 options (options[0].x = BVH triangle count, as the mesh trace), 2 BVH nodes,
// 3 indexed geometry, 5 the R32Float face array (layer 6 x slot + face).

struct TraceNode { lo: vec3<f32>, left: u32, hi: vec3<f32>, count: u32 }

// One face to bake: origin.xyz = light position, origin.w = near; lanes.x = layer, lanes.y = face,
// lanes.z = bitcast range (metres).
struct ShadowJob { origin: vec4<f32>, lanes: vec4<u32> }

struct ShadowParams {
    resolution: u32,
    jobs: u32,
    pad0: u32,
    pad1: u32,
    items: array<ShadowJob, LOCAL_SHADOW_MAX_JOBS>,
}

@group(0) @binding(0) var<uniform> params: ShadowParams;
@group(0) @binding(1) var<uniform> options: array<vec4<u32>, 4>;
@group(0) @binding(2) var<storage, read> nodes: array<TraceNode>;
@group(0) @binding(3) var<storage, read> mesh_geometry: array<u32>;
@group(0) @binding(5) var out_faces: texture_storage_2d_array<r32float, write>;

const LOCAL_SHADOW_MISS: f32 = 1e30;
const LOCAL_SHADOW_EPS: f32 = 0.001;

fn triangle_refs(index: u32) -> vec4<u32> {
    let at = 1u + index * 4u;
    return vec4<u32>(mesh_geometry[at], mesh_geometry[at + 1u], mesh_geometry[at + 2u], mesh_geometry[at + 3u]);
}

fn vertex_at(index: u32) -> u32 {
    return 1u + options[0].x * 4u + index * mesh_geometry[0];
}

fn geometry_vec3(at: u32) -> vec3<f32> {
    return bitcast<vec3<f32>>(vec3<u32>(mesh_geometry[at], mesh_geometry[at + 1u], mesh_geometry[at + 2u]));
}

// The mesh trace's front-to-back closest hit (mesh_trace.wgsl `closest`), distance only.
fn nearest_hit(origin: vec3<f32>, direction: vec3<f32>, limit: f32) -> f32 {
    var best = limit;
    if options[0].x == 0u { return best; }
    var stack: array<u32, 32>;
    stack[0] = 0u;
    var count = 1u;
    loop {
        if count == 0u { break; }
        count--;
        let entry = stack[count];
        let axis = entry >> 30u;
        let node = nodes[entry & 0x3fffffffu];
        let interval = trace_interval(node.lo, node.hi, origin, direction, best);
        if interval.x > interval.y { continue; }
        if node.count == 0u {
            let next_axis = select(axis + 1u, 0u, axis == 2u) << 30u;
            let near_left = direction[axis] > 0.0;
            stack[count] = next_axis | (node.left + u32(near_left));
            stack[count + 1u] = next_axis | (node.left + u32(!near_left));
            count += 2u;
            continue;
        }
        for (var i = node.left; i < node.left + node.count; i++) {
            let refs = triangle_refs(i);
            let a = geometry_vec3(vertex_at(refs.x));
            let edge1 = geometry_vec3(vertex_at(refs.y)) - a;
            let edge2 = geometry_vec3(vertex_at(refs.z)) - a;
            let p = cross(direction, edge2);
            let det = dot(edge1, p);
            if abs(det) < 1e-8 { continue; }
            let offset = origin - a;
            let u = dot(offset, p) / det;
            if u < 0.0 || u > 1.0 { continue; }
            let q = cross(offset, edge1);
            let v = dot(direction, q) / det;
            if v < 0.0 || u + v > 1.0 { continue; }
            let t = dot(edge2, q) / det;
            if t >= LOCAL_SHADOW_EPS && t < best { best = t; }
        }
    }
    return best;
}

// Inverse of raa_local_shadow_face: the direction through (s, t) in [-1, 1]^2 of `face`.
fn face_direction(face: u32, s: f32, t: f32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>(1.0, -t, -s); }
        case 1u: { return vec3<f32>(-1.0, -t, s); }
        case 2u: { return vec3<f32>(s, 1.0, t); }
        case 3u: { return vec3<f32>(s, -1.0, -t); }
        case 4u: { return vec3<f32>(s, -t, 1.0); }
        default: { return vec3<f32>(-s, -t, -1.0); }
    }
}

@compute @workgroup_size(8, 8, 1)
fn bake_faces(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = params.resolution;
    if id.x >= res || id.y >= res || id.z >= params.jobs { return; }
    let job = params.items[id.z];
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(res) * 2.0 - 1.0;
    let direction = normalize(face_direction(job.lanes.y, uv.x, uv.y));
    let near = max(job.origin.w, 0.0);
    let range = bitcast<f32>(job.lanes.z);
    var stored = LOCAL_SHADOW_MISS;
    if range > near {
        let t = nearest_hit(job.origin.xyz + direction * near, direction, range - near);
        if t < range - near { stored = near + t; }
    }
    textureStore(out_faces, vec2<i32>(id.xy), i32(job.lanes.x), vec4<f32>(stored, 0.0, 0.0, 0.0));
}
