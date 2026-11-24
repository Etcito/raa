// raa.real-frame-model/1. FP32 activations and weights, NHWC with channel fastest.
struct ModelParams {
    scene: vec2<u32>,
    history_valid: u32,
    reactive: u32,
    has_reflection: u32,
    has_hit_meta: u32,
    area_slots: u32,
    reserved: u32,
}
struct Guide {
    encoded: array<f32, 12>,
    warped: array<f32, 12>,
    geometry: vec4<f32>, // view normal, metric depth; depth zero means invalid
    gate: f32,
}
struct State {
    hidden: array<f32, 12>,
    geometry: vec4<f32>,
}
@group(0) @binding(0) var<uniform> control: ModelParams;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> frame: RaaTemporalFrame;
@group(0) @binding(3) var depth: texture_depth_2d;
@group(0) @binding(4) var normal: texture_2d<f32>;
@group(0) @binding(5) var motion: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> previous: array<State>;
@group(0) @binding(7) var<storage, read_write> guides: array<Guide>;
@group(0) @binding(8) var<storage, read_write> stem: array<array<f32, 12>>;
@group(0) @binding(9) var pairs: texture_2d_array<f32>;
@group(0) @binding(10) var pairs_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(11) var hdr: texture_2d<f32>;
@group(0) @binding(12) var incident: texture_2d<f32>;
@group(0) @binding(13) var hit_meta: texture_2d<f32>;
@group(0) @binding(14) var<storage, read_write> next: array<State>;
@group(0) @binding(15) var incident_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(16) var<storage, read_write> coefficients: array<array<f32, 8>>;
@group(0) @binding(17) var display_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(18) var<storage, read_write> guide_capture: array<array<f32, 12>>;

const STEM_W: u32 = 0u;
const STEM_B: u32 = 144u;
const SPATIAL_W: u32 = 156u;
const SPATIAL_B: u32 = 264u;
const MIX_W: u32 = 276u;
const MIX_B: u32 = 420u;
const AREA_W: u32 = 432u;
const AREA_B: u32 = 668u;
const RECURRENT_W: u32 = 680u;
const RECURRENT_B: u32 = 1472u;
const REFLECTION_W: u32 = 1496u;
const REFLECTION_B: u32 = 1532u;
const SR_W: u32 = 1536u;
const SR_B: u32 = 1632u;

fn inside(p: vec2<u32>) -> bool { return all(p < control.scene); }
fn clamp_scene(p: vec2<i32>) -> vec2<i32> {
    return clamp(p, vec2<i32>(0), vec2<i32>(control.scene) - 1);
}
fn index(p: vec2<i32>) -> u32 {
    let q = vec2<u32>(clamp_scene(p));
    return q.y * control.scene.x + q.x;
}
fn view_depth(p: vec2<i32>, z: f32) -> f32 {
    if z >= 1.0 { return 0.0; }
    let xy = (vec2<f32>(p) + 0.5) / vec2<f32>(control.scene) * 2.0 - 1.0;
    let h = frame.inv_proj * vec4<f32>(xy.x, -xy.y, z, 1.0);
    return max(0.0, -h.z / h.w);
}
fn jitter_px() -> vec4<f32> {
    let scale = vec2<f32>(control.scene) * vec2<f32>(0.5, -0.5);
    return frame.camera_jitter * scale.xyxy;
}
fn saturating_log(value: f32) -> f32 { return log(1.0 + max(0.0, value)); }
fn history_sample(q: vec2<f32>) -> array<f32, 12> {
    let a = vec2<i32>(floor(q));
    let f = fract(q);
    let h00 = previous[index(a)].hidden;
    let h10 = previous[index(a + vec2<i32>(1, 0))].hidden;
    let h01 = previous[index(a + vec2<i32>(0, 1))].hidden;
    let h11 = previous[index(a + 1)].hidden;
    var result: array<f32, 12>;
    for (var c = 0u; c < 12u; c++) {
        result[c] = mix(mix(h00[c], h10[c], f.x), mix(h01[c], h11[c], f.x), f.y);
    }
    return result;
}
fn pack(p: vec2<i32>) -> array<f32, 12> {
    let g = textureLoad(normal, p, 0);
    let mv = textureLoad(motion, p, 0);
    let z = view_depth(p, textureLoad(depth, p, 0));
    let n = g.xyz;
    let valid = z > 0.0 && dot(n, n) > 0.25;
    let jit = jitter_px();
    let q = vec2<f32>(p) - mv.xy * vec2<f32>(control.scene) + jit.zw - jit.xy;
    var accepted = false;
    var warped: array<f32, 12>;
    if valid && control.history_valid != 0u && frame.temporal.z != 0u && mv.z < 0.5
        && all(q >= vec2<f32>(-0.5)) && all(q < vec2<f32>(control.scene) - 0.5) {
        let prior = previous[index(vec2<i32>(floor(q + 0.5)))].geometry;
        accepted = prior.w > 0.0 && abs(prior.w - z) <= max(0.02, 0.025 * z)
            && dot(prior.xyz, n) >= 0.8;
        if accepted && control.reactive == 0u { warped = history_sample(q); }
    }
    let reactive = f32(control.reactive);
    let i = index(p);
    guides[i].warped = warped;
    guides[i].geometry = select(vec4<f32>(0.0), vec4<f32>(n, z), valid);
    guides[i].gate = f32(accepted) * (1.0 - reactive);
    let nn = select(vec3<f32>(0.0), n, valid);
    return array<f32, 12>(nn.x, nn.y, nn.z, saturating_log(z) / 8.0, g.w,
        mv.x, mv.y, f32(accepted), reactive, jit.x, jit.y, f32(valid));
}
fn stem_at(i: u32, input: array<f32, 12>) {
    for (var out = 0u; out < 12u; out++) {
        var sum = weights[STEM_B + out];
        for (var c = 0u; c < 12u; c++) {
            sum = fma(weights[STEM_W + out * 12u + c], input[c], sum);
        }
        stem[i][out] = max(0.0, sum);
    }
}
@compute @workgroup_size(8, 8)
fn encode_stem(@builtin(global_invocation_id) id: vec3<u32>) {
    if !inside(id.xy) { return; }
    let p = vec2<i32>(id.xy);
    stem_at(index(p), pack(p));
}
@compute @workgroup_size(8, 8)
fn capture_guides(@builtin(global_invocation_id) id: vec3<u32>) {
    if !inside(id.xy) { return; }
    let p = vec2<i32>(id.xy);
    guide_capture[index(p)] = pack(p);
}
@compute @workgroup_size(8, 8)
fn encode_spatial(@builtin(global_invocation_id) id: vec3<u32>) {
    if !inside(id.xy) { return; }
    let p = vec2<i32>(id.xy);
    let i = index(p);
    var dw: array<f32, 12>;
    for (var c = 0u; c < 12u; c++) { dw[c] = weights[SPATIAL_B + c]; }
    for (var tap = 0u; tap < 9u; tap++) {
        let q = p + vec2<i32>(i32(tap % 3u), i32(tap / 3u)) - 1;
        let source = stem[index(q)];
        for (var c = 0u; c < 12u; c++) {
            dw[c] = fma(weights[SPATIAL_W + c * 9u + tap], source[c], dw[c]);
        }
    }
    for (var out = 0u; out < 12u; out++) {
        var sum = weights[MIX_B + out];
        for (var c = 0u; c < 12u; c++) {
            sum = fma(weights[MIX_W + out * 12u + c], dw[c], sum);
        }
        guides[i].encoded[out] = max(0.0, stem[i][out] + sum);
    }
}
fn pair_at(p: vec2<i32>, layer: i32, second: bool) -> vec2<f32> {
    let sample = textureLoad(pairs, clamp_scene(p), layer, 0);
    return select(sample.xy, sample.zw, second);
}
fn filter_pair(p: vec2<i32>, layer: u32, second: bool, guide: Guide) -> vec2<f32> {
    let center = pair_at(p, i32(layer), second);
    let slot = layer * 2u + u32(second);
    if (control.area_slots & (1u << slot)) == 0u { return center; }
    let ratio = select(1.0, clamp(center.x / max(center.y, 1e-8), 0.0, 1.0), center.y > 1e-8);
    var logits: array<f32, 9>;
    var largest = -3.4e38;
    for (var tap = 0u; tap < 9u; tap++) {
        let row = AREA_W + tap * 26u;
        var sum = weights[AREA_B + tap];
        for (var c = 0u; c < 12u; c++) {
            sum = fma(weights[row + c], guide.encoded[c], sum);
        }
        for (var c = 0u; c < 12u; c++) {
            sum = fma(weights[row + 12u + c], guide.warped[c] * guide.gate, sum);
        }
        sum = fma(weights[row + 24u], ratio, sum);
        sum = fma(weights[row + 25u], saturating_log(center.y), sum);
        logits[tap] = sum;
        largest = max(largest, sum);
    }
    var result = vec2<f32>(0.0);
    var total = 0.0;
    for (var tap = 0u; tap < 9u; tap++) {
        let w = exp(logits[tap] - largest);
        let q = p + vec2<i32>(i32(tap % 3u), i32(tap / 3u)) - 1;
        result += w * pair_at(q, i32(layer), second);
        total += w;
    }
    return result / total;
}
@compute @workgroup_size(8, 8)
fn reconstruct_area(@builtin(global_invocation_id) id: vec3<u32>) {
    if !inside(id.xy) { return; }
    let p = vec2<i32>(id.xy);
    let g = guides[index(p)];
    textureStore(pairs_out, p, i32(id.z),
        vec4<f32>(filter_pair(p, id.z, false, g), filter_pair(p, id.z, true, g)));
}
fn recur_dot(row: u32, guide: Guide, late: array<f32, 9>) -> f32 {
    let offset = RECURRENT_W + row * 33u;
    var sum = weights[RECURRENT_B + row];
    for (var c = 0u; c < 12u; c++) {
        sum = fma(weights[offset + c], guide.encoded[c], sum);
    }
    for (var c = 0u; c < 12u; c++) {
        sum = fma(weights[offset + 12u + c], guide.warped[c] * guide.gate, sum);
    }
    for (var c = 0u; c < 9u; c++) {
        sum = fma(weights[offset + 24u + c], late[c], sum);
    }
    return sum;
}
@compute @workgroup_size(8, 8)
fn recurrent(@builtin(global_invocation_id) id: vec3<u32>) {
    if !inside(id.xy) { return; }
    let p = vec2<i32>(id.xy);
    let i = index(p);
    let g = guides[i];
    let color = textureLoad(hdr, p, 0).rgb;
    var r = vec4<f32>(0.0);
    var hit_info = vec4<f32>(0.0);
    if control.has_reflection != 0u {
        r = textureLoad(incident, p, 0);
        if control.has_hit_meta != 0u { hit_info = textureLoad(hit_meta, p, 0); }
        else { hit_info = vec4<f32>(1.0, f32(r.a > 0.0), 0.0, 0.0); }
    }
    let reflection_valid = r.a > 0.0 && hit_info.y > 0.5;
    let rgb = select(vec3<f32>(0.0), r.rgb, reflection_valid);
    let late = array<f32, 9>(saturating_log(color.x), saturating_log(color.y),
        saturating_log(color.z), saturating_log(rgb.x), saturating_log(rgb.y),
        saturating_log(rgb.z), f32(reflection_valid), hit_info.x / 4.0, clamp(hit_info.w, 0.0, 32.0) / 32.0);
    next[i].geometry = g.geometry;
    for (var c = 0u; c < 12u; c++) {
        let candidate = tanh(recur_dot(c, g, late));
        let retain = 1.0 / (1.0 + exp(-recur_dot(c + 12u, g, late)));
        next[i].hidden[c] = mix(candidate, g.warped[c], retain * g.gate);
    }
}
fn project(hidden: array<f32, 12>, start: u32, bias: u32, row: u32) -> f32 {
    var sum = weights[bias + row];
    for (var c = 0u; c < 12u; c++) {
        sum = fma(weights[start + row * 12u + c], hidden[c], sum);
    }
    return sum;
}
@compute @workgroup_size(8, 8)
fn reconstruct_reflection(@builtin(global_invocation_id) id: vec3<u32>) {
    if !inside(id.xy) { return; }
    let p = vec2<i32>(id.xy);
    let original = textureLoad(incident, p, 0);
    var value = original;
    if original.a > 0.0 {
        let h = next[index(p)].hidden;
        for (var c = 0u; c < 3u; c++) {
            value[c] = max(0.0, exp(saturating_log(original[c])
                + 0.5 * tanh(project(h, REFLECTION_W, REFLECTION_B, c))) - 1.0);
        }
    }
    textureStore(incident_out, p, value);
}
@compute @workgroup_size(8, 8)
fn sr_head(@builtin(global_invocation_id) id: vec3<u32>) {
    if !inside(id.xy) { return; }
    let i = index(vec2<i32>(id.xy));
    let h = next[i].hidden;
    for (var c = 0u; c < 8u; c++) {
        coefficients[i][c] = project(h, SR_W, SR_B, c);
    }
}
@compute @workgroup_size(8, 8)
fn resolve_display(@builtin(global_invocation_id) id: vec3<u32>) {
    let display = textureDimensions(display_out);
    if any(id.xy >= display) { return; }
    let ratio = vec2<f32>(control.scene) / vec2<f32>(display);
    let source = (vec2<f32>(id.xy) + 0.5) * ratio - 0.5 + jitter_px().xy;
    let a = vec2<i32>(floor(source));
    let f = fract(source);
    var color = vec4<f32>(0.0);
    var residual = vec3<f32>(0.0);
    var total = 0.0;
    for (var tap = 0u; tap < 4u; tap++) {
        let step = vec2<u32>(tap & 1u, tap >> 1u);
        let q = a + vec2<i32>(step);
        let base = select(vec2<f32>(1.0) - f, f, step != vec2<u32>(0u));
        let c = coefficients[index(q)];
        let delta = source - vec2<f32>(q);
        let k = c[0] + dot(vec2<f32>(c[1], c[2]), delta) + dot(vec2<f32>(c[3], c[4]), ratio);
        let w = base.x * base.y * exp(0.5 * tanh(k));
        color += w * textureLoad(hdr, clamp_scene(q), 0);
        residual += w * vec3<f32>(c[5], c[6], c[7]);
        total += w;
    }
    color /= total;
    let rgb = max(vec3<f32>(0.0), exp(log(1.0 + max(color.rgb, vec3<f32>(0.0)))
        + 0.25 * tanh(residual / total)) - 1.0);
    textureStore(display_out, vec2<i32>(id.xy), vec4<f32>(min(rgb, vec3<f32>(65504.0)), color.a));
}
