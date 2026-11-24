// Pure optical reprojection of a retained real frame. CPU admission proves the
// entire target frustum is covered and the camera eye and scene are unchanged.
struct OpticalWarp {
    target_to_source: mat4x4<f32>,
    source_to_target: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> optical: OpticalWarp;
@group(0) @binding(1) var source_hdr: texture_2d<f32>;
@group(0) @binding(2) var source_depth: texture_depth_2d;
@group(0) @binding(3) var output_hdr: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var output_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var<storage, read_write> incomplete: atomic<u32>;
struct OpticalSample {
    hdr: vec4<f32>,
    depth: f32,
    motion: vec2<f32>,
}

fn optical_sample(xy: vec2<u32>, extent: vec2<u32>) -> OpticalSample {
    let dims = vec2<f32>(extent);
    let target_uv = (vec2<f32>(xy) + 0.5) / dims;
    let ndc = target_uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    let mapped = optical.target_to_source * vec4<f32>(ndc, 0.5, 1.0);
    let old_ndc = mapped.xy / mapped.w;
    let uv = old_ndc * vec2<f32>(0.5, -0.5) + 0.5;
    if mapped.w <= 0.0 || !all(uv >= vec2<f32>(0.0)) || !all(uv <= vec2<f32>(1.0)) {
        atomicStore(&incomplete, 1u);
        return OpticalSample();
    }
    let upper = vec2<i32>(extent) - 1;
    let nearest = clamp(vec2<i32>(uv * dims), vec2<i32>(0), upper);
    let z = textureLoad(source_depth, nearest, 0);
    let p = uv * dims - 0.5;
    let base = vec2<i32>(floor(p));
    let f = fract(p);
    var rgb = vec3<f32>(0.0);
    var total = 0.0;
    for (var tap = 0u; tap < 4u; tap++) {
        let corner = vec2<u32>(tap & 1u, tap >> 1u);
        let q = clamp(base + vec2<i32>(corner), vec2<i32>(0), upper);
        if (textureLoad(source_depth, q, 0) >= 1.0) != (z >= 1.0) { continue; }
        let w = select(vec2<f32>(1.0) - f, f, corner != vec2<u32>(0u));
        let weight = w.x * w.y;
        rgb += weight * textureLoad(source_hdr, q, 0).rgb;
        total += weight;
    }
    if total <= 0.0 || !all(abs(rgb) <= vec3<f32>(65504.0)) {
        atomicStore(&incomplete, 1u);
        return OpticalSample();
    }
    let now = optical.source_to_target * vec4<f32>(old_ndc, z, 1.0);
    let target_depth = select(clamp(now.z / now.w, 0.0, 1.0), 1.0, z >= 1.0);
    return OpticalSample(vec4<f32>(rgb / total, 1.0), target_depth, target_uv - uv);
}

@compute @workgroup_size(8, 8)
fn reproject_optics(@builtin(global_invocation_id) id: vec3<u32>) {
    let extent = textureDimensions(output_hdr);
    if any(id.xy >= extent) { return; }
    let sample = optical_sample(id.xy, extent);
    textureStore(output_hdr, vec2<i32>(id.xy), sample.hdr);
    textureStore(output_depth, vec2<i32>(id.xy), vec4<f32>(sample.depth, 0.0, 0.0, 0.0));
}
