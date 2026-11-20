struct Params {
    count: u32, views: u32, resolution: u32, iteration: u32,
    silhouette: f32, normal: f32, edge: f32, temperature: f32,
    learning_rate: f32, opacity_target: f32, capacity: u32, generation: u32,
    reserved: vec4<u32>,
}
struct Triangle { p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, normal: vec4<f32>, reserved: vec4<f32> }
struct View { matrix: mat4x4<f32>, near_far: vec4<f32> }
struct ViewTable { views: array<View, 128> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<storage, read_write> bins: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> table: ViewTable;

@compute @workgroup_size(64)
fn reset_bins(@builtin(global_invocation_id) id: vec3<u32>) {
    let side = (params.resolution + 15u) / 16u;
    if (id.x >= side * side * params.views) { return; }
    atomicStore(&bins[id.x * (params.capacity + 1u)], 0u);
}

@compute @workgroup_size(64)
fn fill_bins(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.count * params.views) { return; }
    let view = id.x / params.count;
    let index = id.x % params.count;
    let triangle = triangles[index];
    let matrix = table.views[view].matrix;
    let a = matrix * vec4<f32>(triangle.p0.xyz, 1.0);
    let b = matrix * vec4<f32>(triangle.p1.xyz, 1.0);
    let c = matrix * vec4<f32>(triangle.p2.xyz, 1.0);
    // Homogeneous clip planes reject only triangles wholly outside one plane.
    if ((a.x < -a.w && b.x < -b.w && c.x < -c.w) ||
        (a.x > a.w && b.x > b.w && c.x > c.w) ||
        (a.y < -a.w && b.y < -b.w && c.y < -c.w) ||
        (a.y > a.w && b.y > b.w && c.y > c.w) ||
        (a.z < 0.0 && b.z < 0.0 && c.z < 0.0) ||
        (a.z > a.w && b.z > b.w && c.z > c.w)) { return; }
    let side = (params.resolution + 15u) / 16u;
    var lo = vec2<u32>(0u);
    var hi = vec2<u32>(side - 1u);
    // Crossing the eye or near plane conservatively covers the entire viewport.
    if (min(min(a.w, b.w), c.w) > 0.0 && min(min(a.z, b.z), c.z) >= 0.0) {
        let pa = vec2<f32>(a.x / a.w, -a.y / a.w);
        let pb = vec2<f32>(b.x / b.w, -b.y / b.w);
        let pc = vec2<f32>(c.x / c.w, -c.y / c.w);
        let min_px = clamp((min(min(pa, pb), pc) * 0.5 + 0.5) * f32(params.resolution), vec2<f32>(0.0), vec2<f32>(f32(params.resolution)));
        let max_px = clamp((max(max(pa, pb), pc) * 0.5 + 0.5) * f32(params.resolution), vec2<f32>(0.0), vec2<f32>(f32(params.resolution)));
        lo = min(vec2<u32>(floor(min_px / 16.0)), vec2<u32>(side - 1u));
        hi = min(vec2<u32>(floor(max_px / 16.0)), vec2<u32>(side - 1u));
    }
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            let base = (view * side * side + y * side + x) * (params.capacity + 1u);
            let slot = atomicAdd(&bins[base], 1u);
            if (slot < params.capacity) { atomicStore(&bins[base + 1u + slot], index); }
        }
    }
}
