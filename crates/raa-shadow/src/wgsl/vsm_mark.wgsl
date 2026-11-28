// VSM page marking (crates/raa-shadow/src/vsm_gpu.rs): every pixel of the prepass depth (one thread per 4 x 4 block) picks
// its clipmap level (as raa_csm_vsm_level in vsm.wgsl) and sets the request bits of the pages its
// filter footprint touches (the page of each corner of a MARGIN-texel square), at its level and
// at the filter level FILTER_UP levels coarser. Bit layout: level
// l, toroidal entry (y & 127) * 128 + (x & 127), word l * 512 + entry / 32.

struct VsmMark {
    inv_view_proj: mat4x4<f32>,
    light: mat4x4<f32>,
    origins: array<vec4<f32>, 4>,
    // View-matrix row 2 (view z of a world point).
    view_z: vec4<f32>,
    // texel0, levels, pixel angle, level bias.
    params: vec4<f32>,
    // margin texels, constant footprint m (orthographic camera), 0, 0.
    params2: vec4<f32>,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<uniform> mark: VsmMark;
@group(0) @binding(2) var<storage, read_write> requests: array<atomic<u32>>;

fn origin_of(level: u32) -> vec2<i32> {
    let c = mark.origins[level >> 1u];
    return vec2<i32>(select(c.xy, c.zw, (level & 1u) == 1u));
}

fn request(level: u32, page: vec2<i32>) {
    let rel = page - origin_of(level);
    if any(rel < vec2<i32>(0)) || any(rel >= vec2<i32>(128)) {
        return;
    }
    let t = page & vec2<i32>(127);
    let bit = u32(t.y * 128 + t.x);
    let word = level * 512u + (bit >> 5u);
    let mask = 1u << (bit & 31u);
    if ((atomicLoad(&requests[word]) & mask) == 0u) {
        atomicOr(&requests[word], mask);
    }
}

// Request of one pixel: its level and the page box of its MARGIN square (p0 .. p1).
struct Req {
    level: u32,
    p0: vec2<i32>,
    p1: vec2<i32>,
    // The same square at the filter level (level + FILTER_UP, see vsm.wgsl): wide penumbrae
    // are filtered there.
    q0: vec2<i32>,
    q1: vec2<i32>,
}

const FILTER_UP: u32 = 2u;

fn pixel_request(px: vec2<u32>, size: vec2<u32>) -> Req {
    var r = Req(0xffffffffu, vec2<i32>(0), vec2<i32>(0), vec2<i32>(0), vec2<i32>(0));
    let d = textureLoad(depth_tex, vec2<i32>(px), 0);
    if (d >= 1.0) {
        return r;
    }
    let ndc = vec2<f32>(
        (f32(px.x) + 0.5) / f32(size.x) * 2.0 - 1.0,
        1.0 - (f32(px.y) + 0.5) / f32(size.y) * 2.0,
    );
    let h = mark.inv_view_proj * vec4<f32>(ndc, d, 1.0);
    let world = h.xyz / h.w;
    let view_depth = max(-dot(mark.view_z, vec4<f32>(world, 1.0)), 1e-4);
    let footprint = view_depth * mark.params.z + mark.params2.y;
    let lf = floor(log2(max(footprint / mark.params.x, 1e-8)) + 0.5 + mark.params.w);
    r.level = u32(clamp(lf, 0.0, mark.params.y - 1.0));
    let texel = mark.params.x * exp2(f32(r.level));
    let lp = mark.light * vec4<f32>(world, 1.0);
    let uv = vec2<f32>(lp.x, -lp.y) / texel;
    let m = mark.params2.x;
    r.p0 = vec2<i32>(floor((uv - vec2<f32>(m)) / 128.0));
    r.p1 = vec2<i32>(floor((uv + vec2<f32>(m)) / 128.0));
    let quv = uv / f32(1u << FILTER_UP);
    r.q0 = vec2<i32>(floor((quv - vec2<f32>(m)) / 128.0));
    r.q1 = vec2<i32>(floor((quv + vec2<f32>(m)) / 128.0));
    return r;
}

fn request_box(level: u32, p0: vec2<i32>, p1: vec2<i32>) {
    request(level, p0);
    if (p1.x != p0.x) { request(level, vec2<i32>(p1.x, p0.y)); }
    if (p1.y != p0.y) { request(level, vec2<i32>(p0.x, p1.y)); }
    if (p1.x != p0.x && p1.y != p0.y) { request(level, p1); }
}

// One thread per BLOCK x BLOCK pixels; a request equal to the thread's previous one is
// skipped, so neighbouring pixels on the same pages issue one set of atomics.
const BLOCK: u32 = 4u;

@compute @workgroup_size(8, 8, 1)
fn cs_mark(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(depth_tex);
    var last = Req(0xffffffffu, vec2<i32>(0), vec2<i32>(0), vec2<i32>(0), vec2<i32>(0));
    for (var k = 0u; k < BLOCK * BLOCK; k = k + 1u) {
        let px = id.xy * BLOCK + vec2<u32>(k % BLOCK, k / BLOCK);
        if (px.x >= size.x || px.y >= size.y) {
            continue;
        }
        let r = pixel_request(px, size);
        if (r.level == 0xffffffffu || (r.level == last.level && all(r.p0 == last.p0) && all(r.p1 == last.p1))) {
            continue;
        }
        last = r;
        request_box(r.level, r.p0, r.p1);
        let up = r.level + FILTER_UP;
        if (up < u32(mark.params.y)) {
            request_box(up, r.q0, r.q1);
        }
    }
}
