// Target motion.xy = target UV minus real-source UV, z = reject, w = expected
// source standard-Z. Lighting influence, including shadows and reflections,
// is marked separately by the target guide producer.
struct Parameters {
    depth_tolerance: f32,
    refresh_limit: u32,
}

struct RefreshWork {
    groups_x: atomic<u32>,
    groups_y: u32,
    groups_z: u32,
    count: atomic<u32>,
    incomplete: atomic<u32>,
    pixels: array<u32>,
}

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var source_hdr: texture_2d<f32>;
@group(0) @binding(2) var source_depth: texture_depth_2d;
@group(0) @binding(3) var target_motion: texture_2d<f32>;
@group(0) @binding(4) var target_depth: texture_2d<f32>;
@group(0) @binding(5) var lighting_changed: texture_2d<u32>;
@group(0) @binding(6) var warped: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var<storage, read_write> work: RefreshWork;

fn eligible_color(value: vec3<f32>) -> bool {
    return all(abs(value) <= vec3<f32>(65504.0));
}

@compute @workgroup_size(8, 8)
fn warp(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(target_motion);
    if any(id.xy >= size) {
        return;
    }
    let p = vec2<i32>(id.xy);
    let motion = textureLoad(target_motion, p, 0);
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let previous = raa_temporal_surface_uv(uv, motion);
    let depth = textureLoad(target_depth, p, 0).x;
    if previous.z == 1.0 && depth >= 0.0 && depth <= 1.0 && motion.w >= 0.0 && motion.w <= 1.0
        && textureLoad(lighting_changed, p, 0).x == 0u {
        let source = vec2<i32>(previous.xy * vec2<f32>(size));
        let actual = textureLoad(source_depth, source, 0);
        let color = textureLoad(source_hdr, source, 0).rgb;
        if (depth == 1.0) == (actual == 1.0) && (actual == 1.0) == (motion.w == 1.0)
            && abs(actual - motion.w) <= params.depth_tolerance
            && eligible_color(color) {
            textureStore(warped, p, vec4<f32>(color, 1.0));
            return;
        }
    }
    textureStore(warped, p, vec4<f32>(0.0));
    let index = atomicAdd(&work.count, 1u);
    if index < params.refresh_limit {
        work.pixels[index] = id.y * size.x + id.x;
        atomicMax(&work.groups_x, index / 64u + 1u);
    } else {
        atomicStore(&work.incomplete, 1u);
    }
}

@group(0) @binding(8) var warp_result: texture_2d<f32>;
@group(0) @binding(9) var refreshed: texture_2d<f32>;
@group(0) @binding(10) var resolved: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(warp_result);
    if any(id.xy >= size) {
        return;
    }
    let p = vec2<i32>(id.xy);
    var color = textureLoad(warp_result, p, 0);
    if color.a == 0.0 {
        color = textureLoad(refreshed, p, 0);
        if !(color.a == 1.0 && eligible_color(color.rgb)) {
            atomicStore(&work.incomplete, 1u);
            color = vec4<f32>(0.0);
        }
    }
    textureStore(resolved, p, vec4<f32>(color.rgb, 1.0));
}
