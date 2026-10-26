// SSR trace and composite (S12 `ssr`, order 65). fs_trace marches the reflected view ray of
// each glossy pixel in screen space through the Hi-Z min pyramid of ssr_hiz.wgsl (64 steps
// that ascend and descend mips, then 8 binary refinement steps against mip 0); fs_composite
// samples the HDR mip chain at the hit, by roughness and cone footprint, and swaps the IBL
// specular it covers for it. No PRELUDE: group 0 is private, group 1 declares only the
// lighting bindings it reads, verbatim from raa-shader's prelude.

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

const MAX_STEPS: u32 = 64u;
const REFINE_STEPS: u32 = 8u;
// Width of the screen border over which the confidence fades to 0, in uv.
const EDGE_FADE: f32 = 0.1;
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
@fragment
fn fs_trace(in: VsOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.position.xy);
    let d = textureLoad(depth_tex, px, 0);
    let nr = textureLoad(normal_tex, px, 0);
    let roughness = clamp(nr.w, 0.0, 1.0);
    if (d >= 1.0 || roughness > ssr.params.y) {
        return vec4<f32>(0.0);
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
        return vec4<f32>(0.0);
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
        return vec4<f32>(0.0);
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
    return vec4<f32>(hit_uv, confidence, log2(max(footprint, 1.0)));
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

// The IBL specular of the forward pass along the world reflection `r`: prefiltered cube at
// roughness x max mip times the split-sum DFG of the BRDF LUT, at the specular intensity;
// 0 without an environment (info.w < 0.5).
fn env_specular(r: vec3<f32>, roughness: f32, f0: vec3<f32>, n_dot_v: f32) -> vec3<f32> {
    if (raa_env_sh.info.w < 0.5) {
        return vec3<f32>(0.0);
    }
    let lut_uv = vec2<f32>(n_dot_v, roughness);
    let lut = textureSampleLevel(raa_brdf_lut, raa_linear_clamp, lut_uv, 0.0).rg;
    let lod = roughness * raa_env_sh.info.z;
    let prefiltered = textureSampleLevel(raa_env_specular, raa_linear_clamp, r, lod).rgb;
    return prefiltered * (f0 * lut.x + vec3<f32>(lut.y)) * raa_env_sh.info.x;
}

// Full res, into hdr.dst(): hdr + (ssr - env_specular) x F(F0) x confidence x intensity, so
// the reflection replaces the IBL specular it covers.
@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.position.xy);
    let hdr = textureLoad(hdr_mips, px, 0);
    let trace = textureLoad(trace_tex, px, 0);
    let weight = trace.z * ssr.params.x;
    if (weight <= 0.0) {
        return hdr;
    }
    let nr = textureLoad(normal_tex, px, 0);
    let n = normalize(nr.xyz);
    let roughness = clamp(nr.w, 0.0, 1.0);
    let f0 = srgb_to_linear(textureLoad(material_tex, px, 0).rgb);
    let p = view_pos(in.position.xy * ssr.viewport.zw, textureLoad(depth_tex, px, 0));
    let v = normalize(-p);
    let n_dot_v = saturate(dot(n, v));
    let r = (ssr.inv_view * vec4<f32>(reflect(-v, n), 0.0)).xyz;
    let lod = clamp(trace.w, 0.0, max(ssr.params2.x - 1.0, 0.0));
    let reflected = textureSampleLevel(hdr_mips, hdr_samp, trace.xy, lod).rgb;
    let env = env_specular(r, roughness, f0, n_dot_v);
    let rgb = hdr.rgb + (reflected - env) * fresnel_schlick(f0, n_dot_v) * weight;
    return vec4<f32>(max(rgb, vec3<f32>(0.0)), hdr.a);
}
