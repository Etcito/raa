// Convex-hull bake layer conversion (M7). Reads the world normal/coverage and the
// standard-Z depth reference arrays and writes signed source-root normals with the
// normalized source-root view depth in W. The world-to-root matrix is the inverse
// normal matrix of the source-root object frame, supplied in the uniform. A texel without
// a reference fragment (coverage W of the normal/coverage reference 0) packs a zero normal
// and the clear depth 1, so a linear sample next to the silhouette blends towards the far
// plane and the parallax march misses there instead of hitting at the hull entry.
struct PackParams {
    world_to_root: mat3x3<f32>,
};

@group(0) @binding(0) var<uniform> params: PackParams;
@group(0) @binding(1) var normal_coverage: texture_2d_array<f32>;
@group(0) @binding(2) var depth_in: texture_2d_array<f32>;
@group(0) @binding(3) var normal_depth: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(normal_coverage);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let layer = i32(id.z);
    let nc = textureLoad(normal_coverage, coord, layer, 0);
    let d = textureLoad(depth_in, coord, layer, 0);
    let n = params.world_to_root * nc.xyz;
    let len = length(n);
    var out_n = vec3<f32>(0.0);
    if (len > 1e-6) {
        out_n = n / len;
    }
    if (!(nc.w > 0.0)) {
        textureStore(normal_depth, coord, layer, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let depth = clamp(d.x, 0.0, 1.0);
    textureStore(normal_depth, coord, layer, vec4<f32>(out_n, depth));
}
