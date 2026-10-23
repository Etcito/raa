// Display transform (S12 `tonemap`): exposure, AgX / ACES / linear, sRGB OETF, dither.
// WGSL ports of src/curves.rs; keep the constants in step with it.

struct TonemapParams {
    // x = 2^ev, y = ev
    exposure: vec4<f32>,
    // x = mapper (0 agx, 1 aces, 2 linear), y = look (0 none, 1 punchy, 2 golden),
    // z = dither on, w = surface is *Srgb (the hardware encodes)
    flags: vec4<u32>,
}

// S10 EnvGpu, uploaded by the pass from LightingResources::env().
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

@group(0) @binding(0) var<uniform> params: TonemapParams;
@group(0) @binding(1) var hdr_src: texture_2d<f32>;
// Depth32Float bound as unfilterable float: GL backends cannot textureLoad depth textures.
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> env: EnvGpu;

const AGX_MIN_EV: f32 = -12.47393;
const AGX_MAX_EV: f32 = 4.026069;
const ACES_PRE_SCALE: f32 = 1.0 / 0.6;

// Column-major (the CPU port lists rows).
const AGX_INSET = mat3x3<f32>(
    vec3<f32>(0.84247906, 0.042328242, 0.042375655),
    vec3<f32>(0.0784336, 0.8784686, 0.0784336),
    vec3<f32>(0.079223745, 0.07916613, 0.879143),
);
const AGX_OUTSET = mat3x3<f32>(
    vec3<f32>(1.196879, -0.05289685, -0.052971635),
    vec3<f32>(-0.09802088, 1.1519031, -0.09804345),
    vec3<f32>(-0.09902974, -0.09896118, 1.1510737),
);
const ACES_INPUT = mat3x3<f32>(
    vec3<f32>(0.59719, 0.07600, 0.02840),
    vec3<f32>(0.35458, 0.90834, 0.13383),
    vec3<f32>(0.04823, 0.01566, 0.83777),
);
const ACES_OUTPUT = mat3x3<f32>(
    vec3<f32>(1.60475, -0.10208, -0.00327),
    vec3<f32>(-0.53108, 1.10813, -0.07276),
    vec3<f32>(-0.07367, -0.00605, 1.07602),
);

fn tm_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn agx_sigmoid(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2
        + 0.1191 * x - 0.00232;
}

fn agx_look(encoded: vec3<f32>, look: u32) -> vec3<f32> {
    var slope = vec3<f32>(1.0);
    var power = vec3<f32>(1.0);
    var saturation = 1.0;
    if (look == 1u) {
        power = vec3<f32>(1.35);
        saturation = 1.4;
    } else if (look == 2u) {
        slope = vec3<f32>(1.0, 0.9, 0.5);
        power = vec3<f32>(0.8);
        saturation = 0.8;
    } else {
        return encoded;
    }
    let luma = tm_luma(encoded);
    let graded = pow(max(encoded * slope, vec3<f32>(0.0)), power);
    return vec3<f32>(luma) + saturation * (graded - vec3<f32>(luma));
}

fn tm_agx(rgb: vec3<f32>, look: u32) -> vec3<f32> {
    let inset = AGX_INSET * rgb;
    let ev = clamp(log2(max(inset, vec3<f32>(1e-10))), vec3<f32>(AGX_MIN_EV), vec3<f32>(AGX_MAX_EV));
    let encoded = clamp(
        agx_sigmoid((ev - vec3<f32>(AGX_MIN_EV)) / (AGX_MAX_EV - AGX_MIN_EV)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let looked = agx_look(encoded, look);
    return pow(clamp(AGX_OUTSET * looked, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
}

fn rrt_and_odt_fit(v: vec3<f32>) -> vec3<f32> {
    let a = v * (v + vec3<f32>(0.0245786)) - vec3<f32>(0.000090537);
    let b = v * (0.983729 * v + vec3<f32>(0.432951)) + vec3<f32>(0.238081);
    return a / b;
}

fn tm_aces(rgb: vec3<f32>) -> vec3<f32> {
    let v = ACES_INPUT * (max(rgb, vec3<f32>(0.0)) * ACES_PRE_SCALE);
    return clamp(ACES_OUTPUT * rrt_and_odt_fit(v), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn tm_linear(rgb: vec3<f32>) -> vec3<f32> {
    return clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn srgb_oetf(c: vec3<f32>) -> vec3<f32> {
    let v = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let low = v * 12.92;
    let high = 1.055 * pow(v, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(high, low, v <= vec3<f32>(0.0031308));
}

// 8x8 Bayer index (0..63) of a pixel.
fn bayer8(p: vec2<u32>) -> u32 {
    let a = p.x ^ p.y;
    return ((a & 1u) << 5u) | ((p.y & 1u) << 4u) | ((a & 2u) << 2u) | ((p.y & 2u) << 1u)
        | ((a & 4u) >> 1u) | ((p.y & 4u) >> 2u);
}

struct VsOut {
    @builtin(position) position: vec4<f32>,
}

// Fullscreen triangle.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VsOut;
    out.position = vec4<f32>(uv * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(frag.xy);
    // S12 background rule: no geometry and a constant background -> unchanged.
    let depth = textureLoad(depth_tex, px, 0).x;
    if (depth >= 1.0 && env.sky_params.z < 0.5) {
        return vec4<f32>(env.background.rgb, 1.0);
    }

    let hdr = max(textureLoad(hdr_src, px, 0).rgb, vec3<f32>(0.0)) * params.exposure.x;
    var color: vec3<f32>;
    if (params.flags.x == 0u) {
        color = tm_agx(hdr, params.flags.y);
    } else if (params.flags.x == 1u) {
        color = tm_aces(hdr);
    } else {
        color = tm_linear(hdr);
    }
    if (params.flags.w == 0u) {
        color = srgb_oetf(color);
    }
    if (params.flags.z != 0u) {
        let b = (f32(bayer8(vec2<u32>(px))) + 0.5) / 64.0 - 0.5;
        color = color + vec3<f32>(b / 255.0);
    }
    return vec4<f32>(color, 1.0);
}
