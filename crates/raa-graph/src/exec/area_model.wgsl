// Bridge scene-grid area trace planes to the FP32 ticket 09 NHWC ABI.
struct SlotParams {
    slot: u32,
    history_valid: u32,
    light_change: u32,
    active_slots: u32,
    history_offset: vec2<f32>,
}

@group(0) @binding(0) var raw: texture_2d_array<f32>;
@group(0) @binding(1) var depth: texture_2d<f32>;
@group(0) @binding(2) var normal: texture_2d<f32>;
@group(0) @binding(3) var motion: texture_2d<f32>;
@group(0) @binding(4) var previous_raw: texture_2d_array<f32>;
@group(0) @binding(5) var previous_depth: texture_2d<f32>;
@group(0) @binding(6) var<uniform> params: SlotParams;
@group(0) @binding(7) var<storage, read_write> input: array<f32>;
@group(0) @binding(8) var<storage, read> output: array<f32>;
@group(0) @binding(9) var<storage, read_write> visibility: array<f32>;
@group(0) @binding(10) var pairs: texture_storage_2d_array<rgba16float, write>;

fn pair_for_slot(texel: vec4<f32>, slot: u32) -> vec2<f32> {
    return select(texel.xy, texel.zw, slot % 2u == 1u);
}

@compute @workgroup_size(8, 8)
fn pack(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(depth);
    if any(id.xy >= size) { return; }
    let p = vec2<i32>(id.xy);
    let base = (id.y * size.x + id.x) * 12u;
    let z = textureLoad(depth, p, 0).x;
    var lanes: array<f32, 12>;
    if z > 0.0 {
        let slot = params.slot;
        let layer = i32(slot / 2u);
        let current = pair_for_slot(textureLoad(raw, p, layer, 0), slot);
        let n = textureLoad(normal, p, 0).xyz;
        let m = textureLoad(motion, p, 0).xyz;
        lanes[0] = current.x;
        lanes[1] = current.y;
        lanes[4] = n.x;
        lanes[5] = n.y;
        lanes[6] = n.z;
        lanes[7] = z;
        lanes[8] = m.x;
        lanes[9] = m.y;
        lanes[11] = f32(params.light_change);
        let q = floor(vec2<f32>(p) + 0.5 + (params.history_offset - m.xy) * vec2<f32>(size));
        if params.history_valid != 0u && m.z < 0.5 &&
            all(q >= vec2<f32>(0.0)) && all(q < vec2<f32>(size)) {
            let previous = vec2<i32>(q);
            let old_z = textureLoad(previous_depth, previous, 0).x;
            if old_z > 0.0 && abs(old_z - z) <= 0.05 * z {
                let old = pair_for_slot(textureLoad(previous_raw, previous, layer, 0), slot);
                lanes[2] = old.x;
                lanes[3] = old.y;
                lanes[10] = 1.0;
            }
        }
    }
    for (var lane = 0u; lane < 12u; lane++) {
        input[base + lane] = lanes[lane];
    }
}

@compute @workgroup_size(8, 8)
fn store_visibility(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(depth);
    if any(id.xy >= size) { return; }
    let pixel = id.y * size.x + id.x;
    visibility[params.slot * (size.x * size.y) + pixel] = output[pixel * 12u + 11u];
}

fn resolved_pair(slot: u32, pixel: u32, pixels: u32, fallback: vec2<f32>) -> vec2<f32> {
    if (params.active_slots & (1u << slot)) != 0u {
        let v = visibility[slot * pixels + pixel];
        if v >= 0.0 && v <= 1.0 {
            return vec2<f32>(v, 1.0);
        }
    }
    return fallback;
}

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(pairs);
    if any(id.xy >= size) { return; }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    let fallback = textureLoad(raw, p, layer, 0);
    let pixel = id.y * size.x + id.x;
    let pixels = size.x * size.y;
    let first = id.z * 2u;
    textureStore(pairs, p, layer, vec4<f32>(
        resolved_pair(first, pixel, pixels, fallback.xy),
        resolved_pair(first + 1u, pixel, pixels, fallback.zw),
    ));
}
