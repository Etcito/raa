// SSR stages (order 65). Trace the reflected ray through the Hi-Z pyramid, sample current HDR
// at the hit and cone mip, then replace the forward IBL specular. The legacy composite runs the
// last two steps together. Group 0 is private. Group 1 declares the lighting bindings used here,
// verbatim from raa-shader's prelude.

struct SsrParams {
    // view -> clip of the scene camera (clip z in [0, 1], 1 = far)
    proj: mat4x4<f32>,
    // clip -> view
    inv_proj: mat4x4<f32>,
    // world -> view
    view: mat4x4<f32>,
    // view -> world
    inv_view: mat4x4<f32>,
    // full-res w, h, 1 / w, 1 / h
    viewport: vec4<f32>,
    // intensity, max_roughness, thickness_m, Hi-Z mip count
    params: vec4<f32>,
    // HDR mip-chain mip count, 0, 0, 0
    params2: vec4<f32>,
}

struct EnvSh {
    sh: array<vec4<f32>, 9>,
    info: vec4<f32>,
    gi: vec4<f32>,
}

@group(0) @binding(0) var<uniform> ssr: SsrParams;
// Prepass depth (Depth32Float, single-sample, 1 = background).
@group(0) @binding(1) var depth_tex: texture_depth_2d;
// S12 `normal` (Rgba16Float): view-space normal xyz, roughness.
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
// fs_composite only: S12 `material` (Rgba8Unorm): F0 rgb sRGB-encoded, metallic.
@group(0) @binding(3) var material_tex: texture_2d<f32>;
// fs_trace only: the Hi-Z min pyramid (r32float, all mips, textureLoad only).
@group(0) @binding(4) var hiz_tex: texture_2d<f32>;
// fs_composite only: the HDR mip chain (HDR_FORMAT, all mips; mip 0 = copy of hdr.src()).
@group(0) @binding(5) var hdr_mips: texture_2d<f32>;
// fs_composite only: the fs_trace result (Rgba16Float: hit uv, confidence, cone mip).
@group(0) @binding(6) var trace_tex: texture_2d<f32>;
// fs_composite only: linear clamp, for the HDR mip chain.
@group(0) @binding(7) var hdr_samp: sampler;

@group(1) @binding(3) var raa_env_specular: texture_cube<f32>;
@group(1) @binding(4) var<uniform> raa_env_sh: EnvSh;
@group(1) @binding(5) var raa_brdf_lut: texture_2d<f32>;
@group(1) @binding(6) var raa_linear_clamp: sampler;
struct RaaGiVolumeParams { levels: array<vec4<f32>, 5>, resolution: f32, intensity: f32, level_count: u32, flags: u32 }
const RAA_GI_MAX_LEVELS: u32 = 5u;
const RAA_GI_FLAG_VALID: u32 = 1u;
const RAA_GI_FLAG_REFLECTIONS: u32 = 2u;
const RAA_GI_FLAG_LOCAL_SHADOWS: u32 = 4u;
// S38 GI screen irradiance (binding 14) and volume (fs_composite's S39 fallback);
// GI_VOLUME_WGSL is appended to this module.
@group(1) @binding(14) var raa_gi_screen: texture_2d<f32>;
@group(1) @binding(17) var raa_gi_volume: texture_3d<f32>;
@group(1) @binding(18) var<uniform> raa_gi_volume_params: RaaGiVolumeParams;

override MAX_STEPS: u32 = 64u;
override REFINE_STEPS: u32 = 8u;
// Width of the screen border over which the confidence fades to 0, in uv.
override EDGE_FADE: f32 = 0.1;
// Step past a cell boundary, in pixels, so the next cell lookup lands in the next cell.
const CELL_EPS: f32 = 0.01;
const NO_T: f32 = 3.4e38;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    // Texture coordinates of the target, (0, 0) top-left.
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle (3 vertices, no buffers).
@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> VsOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VsOut;
    out.position = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

// View position of target coordinate `uv` at clip depth `z`.
fn view_pos(uv: vec2<f32>, z: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    let v = ssr.inv_proj * ndc;
    return v.xyz / v.w;
}

// Screen point (pixel x, pixel y, clip depth) of view position `p` (in front of the camera).
// Clip depth is affine along a projected line, so a view ray maps to a straight screen ray.
fn to_screen(p: vec3<f32>) -> vec3<f32> {
    let c = ssr.proj * vec4<f32>(p, 1.0);
    let ndc = c.xyz / c.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return vec3<f32>(uv * ssr.viewport.xy, ndc.z);
}

// Positive view distance of pixel position `xy` at clip depth `z`.
fn linear_depth(xy: vec2<f32>, z: f32) -> f32 {
    return -view_pos(xy * ssr.viewport.zw, z).z;
}

// Hi-Z cell of pixel position `xy` at `level`, by the footprint rule of src/hiz.rs: cell i
// covers pixels [i 2^level, (i + 1) 2^level), and the last cell of a row or column extends
// to the target edge (odd-size edge inclusion).
fn hiz_cell(xy: vec2<f32>, level: i32) -> vec2<i32> {
    let dims = vec2<i32>(textureDimensions(hiz_tex, level));
    let c = vec2<i32>(floor(xy / f32(1 << u32(level))));
    return clamp(c, vec2<i32>(0), dims - vec2<i32>(1));
}

// Ray parameter at which the screen ray o + d t leaves cell `c` of `level` (xy only).
fn cell_exit(o: vec3<f32>, d: vec3<f32>, c: vec2<i32>, level: i32) -> f32 {
    let s = f32(1 << u32(level));
    let last = vec2<i32>(textureDimensions(hiz_tex, level)) - vec2<i32>(1);
    let lo = vec2<f32>(c) * s;
    let hi = select(vec2<f32>(c + vec2<i32>(1)) * s, ssr.viewport.xy, c == last);
    let bound = select(lo, hi, d.xy > vec2<f32>(0.0));
    var t = NO_T;
    if (d.x != 0.0) {
        t = min(t, (bound.x - o.x) / d.x);
    }
    if (d.y != 0.0) {
        t = min(t, (bound.y - o.y) / d.y);
    }
    return t;
}

// Ray parameter at which the screen ray leaves [0, w] x [0, h] x [0, 1].
fn ray_t_max(o: vec3<f32>, d: vec3<f32>) -> f32 {
    let bound = select(vec3<f32>(0.0), vec3<f32>(ssr.viewport.xy, 1.0), d > vec3<f32>(0.0));
    let moving = d != vec3<f32>(0.0);
    let tt = select(vec3<f32>(NO_T), (bound - o) / select(vec3<f32>(1.0), d, moving), moving);
    return min(tt.x, min(tt.y, tt.z));
}

// Full res (Rgba16Float): hit uv, confidence, cone mip; 0 where no reflection is traced
// (background, roughness > max_roughness, no hit within MAX_STEPS).
// Metadata lanes are source class, validity, metric distance, and real-frame age.
const SOURCE_SCREEN: f32 = 1.0;
const SOURCE_VOLUME: f32 = 2.0;
const SOURCE_SKY: f32 = 3.0;

struct TraceOutput {
    @location(0) hit: vec4<f32>,
    @location(1) metadata: vec4<f32>,
}

fn trace_hit(in: VsOut) -> TraceOutput {
    let px = vec2<i32>(in.position.xy);
    let d = textureLoad(depth_tex, px, 0);
    let nr = textureLoad(normal_tex, px, 0);
    let roughness = clamp(nr.w, 0.0, 1.0);
    if (d >= 1.0 || roughness > ssr.params.y) {
        return TraceOutput(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    let p = view_pos(in.position.xy * ssr.viewport.zw, d);
    let r = reflect(normalize(p), normalize(nr.xyz));
    // A second point of the view ray, kept in front of the near plane so it projects.
    let near_z = view_pos(vec2<f32>(0.5), 0.0).z;
    var len = max(-p.z, 1.0);
    if (r.z > 0.0) {
        len = min(len, 0.999 * (near_z - p.z) / r.z);
    }
    let o = vec3<f32>(in.position.xy, d);
    let delta = to_screen(p + r * len) - o;
    let len_xy = length(delta.xy);
    if (len_xy < 1e-3) {
        // Along the view direction: the ray never leaves its pixel.
        return TraceOutput(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    // t is the distance along the screen, in pixels.
    let dir = delta / len_xy;
    let t_max = ray_t_max(o, dir);
    let max_level = max(i32(ssr.params.w), 1) - 1;
    var level = 0;
    var t = cell_exit(o, dir, hiz_cell(o.xy, 0), 0) + CELL_EPS;
    // Start of the last advance: the refinement interval is [t_prev, t_hit].
    var t_prev = t;
    var t_hit = -1.0;
    for (var i = 0u; i < MAX_STEPS; i = i + 1u) {
        if (t >= t_max) {
            break;
        }
        let q = o + dir * t;
        let cell = hiz_cell(q.xy, level);
        // Clamped to t so a rounding error never moves the ray backwards.
        let t_exit = clamp(cell_exit(o, dir, cell, level), t, t_max);
        let min_z = textureLoad(hiz_tex, cell, level).r;
        if (max(q.z, o.z + dir.z * t_exit) < min_z) {
            // The ray stays in front of every surface of the cell: skip it, ascend.
            t_prev = t;
            t = t_exit + CELL_EPS;
            level = min(level + 1, max_level);
            continue;
        }
        // Entering in front, no surface of the cell is reached before the depth plane
        // z = min_z (dir.z > 0 then, since the ray leaves the cell at or behind it).
        var t_plane = t;
        if (q.z < min_z) {
            t_plane = (min_z - o.z) / dir.z;
        }
        if (level > 0) {
            if (t_plane > t) {
                t_prev = t;
                t = t_plane;
            }
            level = level - 1;
            continue;
        }
        // One pixel: a hit when the ray is at most thickness_m behind its surface.
        let hit_xy = o.xy + dir.xy * t_plane;
        let behind = linear_depth(hit_xy, o.z + dir.z * t_plane) - linear_depth(hit_xy, min_z);
        if (min_z < 1.0 && behind <= ssr.params.z) {
            t_hit = t_plane;
            break;
        }
        // Behind a thicker surface: go on pixel by pixel.
        t_prev = t;
        t = t_exit + CELL_EPS;
    }
    if (t_hit < 0.0) {
        return TraceOutput(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    // Binary refinement of the crossing against the full-res depth (Hi-Z mip 0).
    var lo = t_prev;
    var hi = t_hit;
    for (var j = 0u; j < REFINE_STEPS; j = j + 1u) {
        let mid = 0.5 * (lo + hi);
        let q = o + dir * mid;
        if (q.z >= textureLoad(hiz_tex, hiz_cell(q.xy, 0), 0).r) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let hit = o + dir * hi;
    let hit_uv = hit.xy * ssr.viewport.zw;
    let edge = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
    let edge_fade = saturate(edge / EDGE_FADE);
    // Ray length as a fraction of the screen diagonal.
    let travel = saturate(hi / length(ssr.viewport.xy));
    let length_fade = 1.0 - travel * travel;
    // Back-facing hit: the surface hit faces away from the ray.
    let last_px = vec2<i32>(ssr.viewport.xy) - vec2<i32>(1);
    let hit_px = clamp(vec2<i32>(hit.xy), vec2<i32>(0), last_px);
    let hit_n = normalize(textureLoad(normal_tex, hit_px, 0).xyz);
    let facing = smoothstep(0.0, 0.05, -dot(hit_n, r));
    let confidence = edge_fade * length_fade * facing;
    // Cone footprint: the GGX lobe (half-maximum tan = 0.6436 alpha around the half vector,
    // doubled for the reflected ray) widened over the ray length, as a mip of the chain.
    let alpha = roughness * roughness;
    let cone = min(2.0 * atan(0.6436 * alpha), 1.4);
    let footprint = 2.0 * hi * tan(cone);
    let distance_m = distance(p, view_pos(hit_uv, hit.z));
    return TraceOutput(
        vec4<f32>(hit_uv, confidence, log2(max(footprint, 1.0))),
        vec4<f32>(SOURCE_SCREEN, 1.0, distance_m, 0.0),
    );
}

@fragment
fn fs_trace(in: VsOut) -> @location(0) vec4<f32> {
    return trace_hit(in).hit;
}

@fragment
fn fs_hit(in: VsOut) -> TraceOutput {
    return trace_hit(in);
}

// Piecewise sRGB EOTF (the material target stores F0 sRGB-encoded).
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn fresnel_schlick(f0: vec3<f32>, n_dot_v: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - n_dot_v, 5.0);
}

// The open-sky IBL specular of the forward pass along the world reflection `r`: prefiltered
// cube at roughness x max mip times the split-sum DFG of the BRDF LUT, at the specular
// intensity; 0 without an environment (info.w < 0.5).
fn sky_specular(r: vec3<f32>, roughness: f32, f0: vec3<f32>, n_dot_v: f32) -> vec3<f32> {
    if (raa_env_sh.info.w < 0.5) {
        return vec3<f32>(0.0);
    }
    // BRDF region of the 128 x 192 atlas (contract brdf-ltc-atlas), as raa_brdf_lut_uv.
    let lut_uv = clamp(vec2<f32>(n_dot_v, roughness) * 128.0, vec2<f32>(0.5), vec2<f32>(127.5)) / vec2<f32>(128.0, 192.0);
    let lut = textureSampleLevel(raa_brdf_lut, raa_linear_clamp, lut_uv, 0.0).rg;
    let lod = roughness * raa_env_sh.info.z;
    let prefiltered = textureSampleLevel(raa_env_specular, raa_linear_clamp, r, lod).rgb;
    return prefiltered * (f0 * lut.x + vec3<f32>(lut.y)) * raa_env_sh.info.x;
}

// The specular the forward ambient adds along `r`: the open sky blended toward the GI
// irradiance estimate E_gi / pi by EnvSh.gi.x (S45), at the pixel `px`; 0 without an
// environment (info.w < 0.5).
fn env_specular(r: vec3<f32>, roughness: f32, f0: vec3<f32>, n_dot_v: f32, px: vec2<i32>) -> vec3<f32> {
    if (raa_env_sh.info.w < 0.5) {
        return vec3<f32>(0.0);
    }
    let lut_uv = clamp(vec2<f32>(n_dot_v, roughness) * 128.0, vec2<f32>(0.5), vec2<f32>(127.5)) / vec2<f32>(128.0, 192.0);
    let lut = textureSampleLevel(raa_brdf_lut, raa_linear_clamp, lut_uv, 0.0).rg;
    let size = vec2<i32>(textureDimensions(raa_gi_screen));
    let pixel = clamp(px, vec2<i32>(0), size - vec2<i32>(1));
    let e_gi = textureLoad(raa_gi_screen, pixel, 0).rgb * raa_gi_volume_params.intensity;
    let gi = e_gi * (1.0 / 3.14159265) * (f0 * lut.x + vec3<f32>(lut.y));
    return mix(sky_specular(r, roughness, f0, n_dot_v), gi, clamp(raa_env_sh.gi.x, 0.0, 1.0));
}

// Incident screen radiance in scene-linear HDR, before material response.
fn screen_incident(px: vec2<i32>) -> vec4<f32> {
    let trace = textureLoad(trace_tex, px, 0);
    if (trace.z <= 0.0) {
        return vec4<f32>(0.0);
    }
    let lod = clamp(trace.w, 0.0, max(ssr.params2.x - 1.0, 0.0));
    return vec4<f32>(textureSampleLevel(hdr_mips, hdr_samp, trace.xy, lod).rgb, trace.z);
}

@fragment
fn fs_incident(in: VsOut) -> @location(0) vec4<f32> {
    return screen_incident(vec2<i32>(in.position.xy));
}

struct Replacement {
    @location(0) color: vec4<f32>,
    @location(1) fallback: vec4<f32>,
    @location(2) metadata: vec4<f32>,
}

fn replace_environment(in: VsOut, screen: vec4<f32>) -> Replacement {
    let px = vec2<i32>(in.position.xy);
    let hdr = textureLoad(hdr_mips, px, 0);
    var result = Replacement(hdr, vec4<f32>(0.0), vec4<f32>(0.0));
    let use_volume = (raa_gi_volume_params.flags & RAA_GI_FLAG_REFLECTIONS) != 0u;
    var weight = screen.a * ssr.params.x;
    if (!use_volume && weight <= 0.0) {
        return result;
    }
    let nr = textureLoad(normal_tex, px, 0);
    let roughness = clamp(nr.w, 0.0, 1.0);
    let depth = textureLoad(depth_tex, px, 0);
    if (use_volume && (roughness > ssr.params.y || depth >= 1.0)) {
        return result;
    }
    let n = normalize(nr.xyz);
    let f0 = srgb_to_linear(textureLoad(material_tex, px, 0).rgb);
    let p = view_pos(in.position.xy * ssr.viewport.zw, depth);
    let v = normalize(-p);
    let n_dot_v = saturate(dot(n, v));
    let r = (ssr.inv_view * vec4<f32>(reflect(-v, n), 0.0)).xyz;
    let env = env_specular(r, roughness, f0, n_dot_v, px);
    var reflected = screen.rgb;
    if (use_volume) {
        let p_w = (ssr.inv_view * vec4<f32>(p, 1.0)).xyz;
        let n_w = (ssr.inv_view * vec4<f32>(n, 0.0)).xyz;
        let v0 = raa_gi_finest_voxel();
        let origin_offset = n_w * 1.5 * v0;
        let hit = raa_gi_trace(p_w + origin_offset, r, v0, 160.0, 48u, 0.5);
        // Keep the legacy split-sum sky term on a miss. Its distance is undefined (zero).
        let fallback = select(sky_specular(r, roughness, f0, n_dot_v), hit.radiance, hit.hit);
        result.fallback = vec4<f32>(fallback, 1.0 - screen.a);
        if (hit.hit) {
            // The volume interface has no real-frame timestamp.
            result.metadata = vec4<f32>(SOURCE_VOLUME, 1.0, length(origin_offset + r * hit.t), -1.0);
        } else if (raa_env_sh.info.w >= 0.5) {
            result.metadata = vec4<f32>(SOURCE_SKY, 1.0, 0.0, 0.0);
        }
        reflected = mix(fallback, select(fallback, reflected, screen.a > 0.0), screen.a);
        weight = ssr.params.x;
    }
    let rgb = hdr.rgb + (reflected - env) * fresnel_schlick(f0, n_dot_v) * weight;
    result.color = vec4<f32>(max(rgb, vec3<f32>(0.0)), hdr.a);
    return result;
}

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    return replace_environment(in, screen_incident(vec2<i32>(in.position.xy))).color;
}

// Binding 6 is incident RGB and confidence for this entry, and raw hit data for fs_incident.
@fragment
fn fs_replace(in: VsOut) -> Replacement {
    return replace_environment(in, textureLoad(trace_tex, vec2<i32>(in.position.xy), 0));
}
