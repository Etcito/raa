// Volumetric sun shafts (S12 `sunshafts`, order 69): a half-resolution single-scattering
// ray march of the view ray through the CSM shadow map (32 steps, interleaved-gradient
// jitter by pixel and frame), with the local density scaled by the height-fog factor of
// EnvGpu.fog and a Henyey-Greenstein phase (g 0.7) toward the sun, then a depth-aware
// bilateral 4-tap upsample added to the HDR source. No PRELUDE: group 0 is private, group 1
// declares only the lighting bindings it reads, verbatim from raa-shader's prelude.

struct ShaftsParams {
    // clip -> world of the scene camera
    inv_view_proj: mat4x4<f32>,
    // world -> view of the scene camera (cascade selection by view depth)
    view: mat4x4<f32>,
    // xyz world camera position, w 0
    camera_pos: vec4<f32>,
    // xyz normalised sun travel direction, w intensity
    sun_dir: vec4<f32>,
    // linear rgb, w 0
    sun_color: vec4<f32>,
    // half-res w, h, full-res w, h
    viewport: vec4<f32>,
    // density, max_distance_m, frame index, 0
    params: vec4<f32>,
}

struct ShadowGpu {
    cascade_view_proj: array<mat4x4<f32>, 4>,
    splits: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    reserved: vec4<f32>,
}

struct EnvGpu {
    sun_disc: vec4<f32>,
    zenith: vec4<f32>,
    horizon: vec4<f32>,
    ground: vec4<f32>,
    sky_params: vec4<f32>,
    background: vec4<f32>,
    clouds: vec4<f32>,
    clouds2: vec4<f32>,
    cloud_color: vec4<f32>,
    fog: vec4<f32>,
    fog_color: vec4<f32>,
    aerial: vec4<f32>,
    reserved: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> shafts: ShaftsParams;
// Full-resolution scene depth (Depth32Float, clip z [0, 1], 1 = background).
@group(0) @binding(1) var depth_tex: texture_depth_2d;
// fs_composite only: the half-res march result (rgb in-scatter, a view depth).
@group(0) @binding(2) var march_tex: texture_2d<f32>;
// fs_composite only: hdr.src().
@group(0) @binding(3) var hdr_src: texture_2d<f32>;
// fs_composite only: linear clamp.
@group(0) @binding(4) var hdr_samp: sampler;

@group(1) @binding(0) var raa_shadow_map: texture_depth_2d_array;
@group(1) @binding(1) var raa_shadow_cmp: sampler_comparison;
@group(1) @binding(2) var<uniform> raa_shadow: ShadowGpu;
@group(1) @binding(11) var<uniform> raa_env: EnvGpu;

const STEPS: u32 = 32u;
const HG_G: f32 = 0.7;
const PI: f32 = 3.14159265359;
// View depth given to background pixels (depth 1), so the bilateral weights still match.
const SKY_DEPTH: f32 = 1e6;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    // Texture coordinates of the target, (0, 0) top-left.
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle (3 vertices, no buffers).
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VsOut;
    out.position = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

// World position of target coordinate `uv` at clip depth `z`.
fn unproject(uv: vec2<f32>, z: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    let w = shafts.inv_view_proj * ndc;
    return w.xyz / w.w;
}

fn view_depth_of(world: vec3<f32>) -> f32 {
    return -(shafts.view * vec4<f32>(world, 1.0)).z;
}

// Full-res pixel under target coordinate `uv`.
fn full_pixel(uv: vec2<f32>) -> vec2<i32> {
    let size = vec2<i32>(shafts.viewport.zw);
    return clamp(vec2<i32>(uv * shafts.viewport.zw), vec2<i32>(0), size - vec2<i32>(1));
}

fn scene_view_depth(uv: vec2<f32>) -> f32 {
    let d = textureLoad(depth_tex, full_pixel(uv), 0);
    if (d >= 1.0) {
        return SKY_DEPTH;
    }
    return view_depth_of(unproject(uv, d));
}

// Interleaved gradient noise (Jimenez 2014) in [0, 1), animated by the frame index.
fn ign(frag: vec2<f32>, frame: f32) -> f32 {
    let p = frag + 5.588238 * (frame % 64.0);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// Henyey-Greenstein phase for the cosine between the view ray and the direction to the sun.
fn hg_phase(cos_theta: f32) -> f32 {
    let g2 = HG_G * HG_G;
    let denom = max(1.0 + g2 - 2.0 * HG_G * cos_theta, 1e-4);
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

// Height-fog factor of EnvGpu.fog (density, falloff, base): 1 at and below the base
// height, exp(-falloff x height above it). The fog density itself is raa-fog's
// extinction and does not gate the shafts (fog off = uniform params.density).
fn height_factor(y: f32) -> f32 {
    return exp(-max(raa_env.fog.y, 0.0) * max(y - raa_env.fog.z, 0.0));
}

// Sun visibility at world `p`: one hardware comparison in the cascade picked by view
// depth; 1 past the last split or outside the cascade.
fn sun_visibility(p: vec3<f32>) -> f32 {
    let vd = view_depth_of(p);
    let s = raa_shadow.splits;
    if (vd > s.w) {
        return 1.0;
    }
    var idx = 0u;
    if (vd > s.x) { idx = 1u; }
    if (vd > s.y) { idx = 2u; }
    if (vd > s.z) { idx = 3u; }
    let clip = raa_shadow.cascade_view_proj[idx] * vec4<f32>(p, 1.0);
    let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    if (clip.z <= 0.0 || clip.z >= 1.0
        || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }
    return textureSampleCompareLevel(raa_shadow_map, raa_shadow_cmp, uv, i32(idx), clip.z);
}

// Half-res march (Rgba16Float): rgb = in-scattered sun radiance, a = view depth of the
// scene under the pixel (for the bilateral upsample). 0 when shadows are disabled.
@fragment
fn fs_march(in: VsOut) -> @location(0) vec4<f32> {
    if (raa_shadow.params.w < 0.5) {
        return vec4<f32>(0.0);
    }
    let cam = shafts.camera_pos.xyz;
    let d = textureLoad(depth_tex, full_pixel(in.uv), 0);
    var dir: vec3<f32>;
    var scene_dist: f32;
    var vd: f32;
    if (d >= 1.0) {
        dir = normalize(unproject(in.uv, 0.999) - cam);
        scene_dist = shafts.params.y;
        vd = SKY_DEPTH;
    } else {
        let world = unproject(in.uv, d);
        dir = normalize(world - cam);
        scene_dist = length(world - cam);
        vd = view_depth_of(world);
    }
    let dist = min(scene_dist, shafts.params.y);
    let step_len = dist / f32(STEPS);
    let jitter = ign(in.position.xy, shafts.params.z);
    let phase = hg_phase(dot(dir, -shafts.sun_dir.xyz));
    var sum = 0.0;
    for (var i = 0u; i < STEPS; i = i + 1u) {
        let p = cam + dir * ((f32(i) + jitter) * step_len);
        let density = shafts.params.x * height_factor(p.y);
        sum = sum + sun_visibility(p) * density;
    }
    let radiance = shafts.sun_color.rgb * shafts.sun_dir.w * (sum * phase * step_len);
    return vec4<f32>(radiance, vd);
}

// Full res: depth-aware bilateral 4-tap upsample of the march result (bilinear weights x
// relative view-depth similarity) added to hdr.src(), written to hdr.dst().
@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureSampleLevel(hdr_src, hdr_samp, in.uv, 0.0);
    let depth = scene_view_depth(in.uv);
    let half_size = vec2<i32>(shafts.viewport.xy);
    let h = in.uv * shafts.viewport.xy - vec2<f32>(0.5);
    let base = vec2<i32>(floor(h));
    let f = fract(h);
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let o = vec2<i32>(i32(i & 1u), i32(i >> 1u));
        let coord = clamp(base + o, vec2<i32>(0), half_size - vec2<i32>(1));
        let tap = textureLoad(march_tex, coord, 0);
        let bx = select(1.0 - f.x, f.x, o.x == 1);
        let by = select(1.0 - f.y, f.y, o.y == 1);
        let rel = abs(tap.a - depth) / max(depth, 1e-3);
        let w = max(bx * by, 1e-3) / (1e-2 + rel);
        sum = sum + tap.rgb * w;
        weight = weight + w;
    }
    return vec4<f32>(hdr.rgb + sum / max(weight, 1e-6), hdr.a);
}
