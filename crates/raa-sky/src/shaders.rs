//! Full WGSL modules of the `sky.env` pass, the `sky.background` contributor and the
//! atmosphere LUT compute passes.
//!
//! The legacy render modules are `PRELUDE_WGSL + LIB_WGSL + SKY_WGSL +` their entry file, so
//! groups 0 (frame) and 1 (lighting) are declared exactly as in the forward pass. The physical
//! builders replace `SKY_WGSL` with the atmosphere/celestial/cloud evaluator and append the
//! private S10 group-2 declarations. The atmosphere LUT module is the S23 helper source plus
//! its compute entries and deliberately omits the prelude: groups 0/1 stay empty and every
//! resource is an explicit group-2 binding. Self-contained (no `crate::` paths) so
//! `tests/variants.rs` can include it with `#[path]`.

use raa_shader::{LIB_WGSL, PRELUDE_WGSL};

/// The same file as `SKY_WGSL`.
const SKY_SRC: &str = include_str!("wgsl/sky.wgsl");
/// Entry points of the env cube pass (one face per instance).
const SKY_ENV_SRC: &str = include_str!("wgsl/sky_env.wgsl");
/// Entry points of the background contributor.
const SKY_BACKGROUND_SRC: &str = include_str!("wgsl/sky_background.wgsl");
/// The S23 atmosphere helpers and the S10 `PhysicalSkyGpu` record, without bindings.
const ATMOSPHERE_SRC: &str = include_str!("wgsl/atmosphere.wgsl");
/// The three LUT compute entries (static transmittance, static multiple scattering, dynamic
/// sky view) of the physical atmosphere.
const ATMOSPHERE_LUTS_SRC: &str = include_str!("wgsl/atmosphere_luts.wgsl");
/// Shared galactic basis, noise and celestial-map coordinates; no bindings.
const MILKY_WAY_SRC: &str = include_str!("wgsl/milky_way.wgsl");
const MILKY_WAY_BAKE_SRC: &str = include_str!("wgsl/milky_way_bake.wgsl");

/// The unchanged S10 group-2 layout of both physical common-sky consumers: the private
/// `PhysicalSkyGpu` uniform, the transmittance and per-view sky-view LUTs, the cloud
/// panorama/screen results, the Milky Way map and the two filtering samplers.
///
/// Binding 7 repeats U and clamps V (celestial-map seams and the panorama azimuth);
/// binding 6 clamps both axes (LUT and screen lookups). Every unused texture is a valid
/// initialized dummy bound by the Rust consumers, and allocation never means readiness.
const PHYSICAL_SKY_BINDINGS: &str = "
@group(2) @binding(0) var<uniform> raa_physical_sky: PhysicalSkyGpu;
@group(2) @binding(1) var raa_sky_transmittance: texture_2d<f32>;
@group(2) @binding(2) var raa_sky_view: texture_2d<f32>;
@group(2) @binding(3) var raa_sky_cloud_panorama: texture_2d<f32>;
@group(2) @binding(4) var raa_sky_cloud_screen: texture_2d<f32>;
@group(2) @binding(5) var raa_sky_milky_way: texture_2d<f32>;
@group(2) @binding(6) var raa_sky_clamp_sampler: sampler;
@group(2) @binding(7) var raa_sky_repeat_sampler: sampler;
";

/// Common celestial evaluator with the canonical S10 evaluation bindings, without entries.
/// Card 7436 folds it into both physical consumers. Bind the map through
/// `AtmosphereResources::milky_way()`; allocation is not submitted-bake readiness.
pub(crate) fn celestial_module_wgsl() -> String {
    let bodies = include_str!("wgsl/celestial.wgsl");
    format!(
        "{PRELUDE_WGSL}\n{LIB_WGSL}\n{ATMOSPHERE_SRC}\n{MILKY_WAY_SRC}\n\
         {PHYSICAL_SKY_BINDINGS}\n{bodies}"
    )
}

// The physical consumers below consume this evaluator; the pin keeps its signature stable.
const _: fn() -> String = celestial_module_wgsl;

/// The physical common-sky evaluation added to the legacy entry points.
///
/// Clear sky = the per-view atmosphere LUT times the raw daylight baseline and the
/// profile-relative artistic controls (the `sky.env` prepare writes ratios, so preset
/// defaults are 1), plus the night glow and the complete celestial evaluator. Clouds are
/// composed exactly as `shaders/common/environment.wgsl:93-113` at the pinned Solstice
/// revision: `clear * cloud.a + cloud.rgb`, with the negative un-marched screen alpha
/// rejected in favour of a valid panorama or the neutral cloud contribution.
const PHYSICAL_SKY_SRC: &str = r#"
const raa_physical_sky_night_zenith: vec3<f32> = vec3<f32>(0.3, 0.45, 1.0);
const raa_physical_sky_night_horizon: vec3<f32> = vec3<f32>(0.62, 0.7, 0.9);

// Profile-relative artistic tint: the `sky.env` prepare stores authored/preset ratios in the
// legacy gradient lanes, so an unchanged preset contributes exactly one and an explicit
// override still acts. The physical atmosphere supplies the colour itself.
fn raa_physical_sky_tint(direction: vec3<f32>, env: EnvGpu) -> vec3<f32> {
    let softness = max(env.sky_params.y, 1.0e-3);
    let t = pow(clamp(1.0 - max(direction.y, 0.0), 0.0, 1.0), softness);
    let above = mix(env.zenith.rgb, env.horizon.rgb, t);
    let below_mix = smoothstep(0.0, 0.06, max(-direction.y, 0.0));
    let below = mix(env.horizon.rgb, env.ground.rgb, below_mix);
    return max(select(below, above, direction.y >= 0.0), vec3<f32>(0.0));
}

// Radiance of the clear sky plus the celestial bodies; clouds are kept separate so the
// composition shape stays the source's.
fn raa_physical_sky_clear(direction: vec3<f32>, env: EnvGpu) -> vec3<f32> {
    let uv = raa_atmosphere_sky_view_uv(direction);
    let scattered = textureSampleLevel(raa_sky_view, raa_sky_clamp_sampler, uv, 0.0).rgb;
    let intensity = max(env.sky_params.x, 0.0);
    let daylight = max(raa_physical_sky.daylight.w, 0.0);
    let horizon = exp(-clamp(direction.y, 0.0, 1.0) * 5.0);
    let glow = mix(raa_physical_sky_night_zenith, raa_physical_sky_night_horizon, horizon)
        * max(raa_physical_sky.equinox.w, 0.0) * (0.4 + 2.0 * horizon);
    let clear = max(scattered * daylight * raa_physical_sky_tint(direction, env) + glow, vec3<f32>(0.0))
        * intensity;
    return clear + raa_celestial_radiance(direction);
}

// Panorama clouds use the atmosphere sky-view direction mapping; validity is the exact
// `CloudViews::panorama_valid` the prepare stored in `flags.z`.
fn raa_physical_sky_cloud_panorama(direction: vec3<f32>) -> vec4<f32> {
    if (raa_physical_sky.flags.z == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return textureSampleLevel(
        raa_sky_cloud_panorama,
        raa_sky_repeat_sampler,
        raa_atmosphere_sky_view_uv(direction),
        0.0,
    );
}

// The dome uv of a world direction (twin of `clouds/dome.rs` `dome_uv`): hemi-octahedral over
// the direction shifted up by 0.05, so the dome reaches 0.05 below the horizon.
fn raa_physical_sky_dome_uv(direction: vec3<f32>) -> vec2<f32> {
    let v = vec3<f32>(direction.x, max(direction.y + 0.05, 0.0), direction.z);
    let length_v = length(v);
    var u = vec3<f32>(1.0, 0.0, 0.0);
    if (length_v > 1.0e-6) {
        u = v / length_v;
    }
    let p = u.xz / max(abs(u.x) + u.y + abs(u.z), 1.0e-6);
    return vec2<f32>(p.x + p.y, p.x - p.y) * 0.5 + 0.5;
}

// The cloud dome sampled by world direction with the bilinear clamping sampler (binding 4 holds
// the dome while `flags.w` is 1).
fn raa_physical_sky_cloud_dome(direction: vec3<f32>) -> vec4<f32> {
    let dome = textureSampleLevel(
        raa_sky_cloud_screen,
        raa_sky_clamp_sampler,
        raa_physical_sky_dome_uv(direction),
        0.0,
    );
    return vec4<f32>(max(dome.rgb, vec3<f32>(0.0)), clamp(dome.a, 0.0, 1.0));
}

// Screen clouds use viewport UV. Texels M2 did not march hold the reserved negative alpha;
// they never enter composition, falling back to a valid panorama or the neutral result.
// With the cloud dome on (`flags.w`), binding 4 holds the dome instead and is sampled by world
// direction with the bilinear clamping sampler: the viewport UV is not used.
fn raa_physical_sky_cloud_screen(direction: vec3<f32>, uv: vec2<f32>) -> vec4<f32> {
    if (raa_physical_sky.flags.w != 0u) {
        return raa_physical_sky_cloud_dome(direction);
    }
    if (raa_physical_sky.flags.y != 0u) {
        let cloud = textureSampleLevel(raa_sky_cloud_screen, raa_sky_repeat_sampler, uv, 0.0);
        if (cloud.a >= 0.0) {
            return cloud;
        }
    }
    return raa_physical_sky_cloud_panorama(direction);
}

fn raa_physical_sky_composite(clear: vec3<f32>, cloud: vec4<f32>) -> vec3<f32> {
    return max(clear * clamp(cloud.a, 0.0, 1.0) + max(cloud.rgb, vec3<f32>(0.0)), vec3<f32>(0.0));
}

// The environment (cube, reflections, image-based lighting): with the cloud dome on its clouds
// come from the dome, not from the 192x108 panorama, which looked pixelated in reflections.
fn raa_physical_sky_radiance(direction: vec3<f32>, env: EnvGpu) -> vec3<f32> {
    var cloud = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (raa_physical_sky.flags.w != 0u) {
        cloud = raa_physical_sky_cloud_dome(direction);
    } else {
        cloud = raa_physical_sky_cloud_panorama(direction);
    }
    return raa_physical_sky_composite(raa_physical_sky_clear(direction, env), cloud);
}

fn raa_physical_sky_radiance_screen(direction: vec3<f32>, uv: vec2<f32>, env: EnvGpu) -> vec3<f32> {
    return raa_physical_sky_composite(
        raa_physical_sky_clear(direction, env),
        raa_physical_sky_cloud_screen(direction, uv),
    );
}

// The S11 common seam keeps its signature; only the implementation changes with the model.
fn raa_sky_radiance(dir: vec3<f32>, env: EnvGpu, sun_dir: vec3<f32>, sun_color: vec3<f32>, time: f32) -> vec3<f32> {
    return raa_physical_sky_radiance(dir, env);
}
"#;

/// The physical background entry: the same depth-1 fullscreen triangle as the legacy one,
/// but the cloud result is sampled in viewport UV while the world ray comes from the frame.
const PHYSICAL_BACKGROUND_SRC: &str = r#"
struct RaaPhysicalSkyBackgroundVaryings {
    @builtin(position) clip: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> RaaPhysicalSkyBackgroundVaryings {
    let xy = vec2<f32>(f32((vertex << 1u) & 2u), f32(vertex & 2u)) * 2.0 - vec2<f32>(1.0);
    var out: RaaPhysicalSkyBackgroundVaryings;
    out.clip = vec4<f32>(xy, 1.0, 1.0);
    return out;
}

fn raa_physical_sky_bg_unproject(ndc: vec3<f32>) -> vec3<f32> {
    let p = raa_frame.inv_proj * vec4<f32>(ndc, 1.0);
    let w = select(p.w, 1e-6, abs(p.w) < 1e-6);
    return p.xyz / w;
}

fn raa_physical_sky_bg_world_dir(frag: vec4<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(
        frag.x * raa_frame.viewport.z * 2.0 - 1.0,
        1.0 - frag.y * raa_frame.viewport.w * 2.0,
    );
    let near = raa_physical_sky_bg_unproject(vec3<f32>(ndc, 0.0));
    let mid = raa_physical_sky_bg_unproject(vec3<f32>(ndc, 0.5));
    let world = (raa_frame.inv_view * vec4<f32>(mid - near, 0.0)).xyz;
    let len = length(world);
    if (!(len > 1e-12)) {
        return vec3<f32>(0.0, 0.0, -1.0);
    }
    return world / len;
}

@fragment
fn fs_main(in: RaaPhysicalSkyBackgroundVaryings) -> @location(0) vec4<f32> {
    let dir = raa_physical_sky_bg_world_dir(in.clip);
    let uv = in.clip.xy * raa_frame.viewport.zw;
    let c = raa_physical_sky_radiance_screen(dir, uv, raa_env);
    return vec4<f32>(c, 1.0);
}
"#;

/// The landed common celestial evaluator plus the physical clear-sky/cloud composition.
fn physical_sky_module() -> String {
    let mut s = celestial_module_wgsl();
    s.push_str(PHYSICAL_SKY_SRC);
    s.push('\n');
    s
}

/// The `sky.env` module for the physical model: the same face entry and S11 seam as the
/// legacy builder, composed over the shared celestial evaluator and the explicit S10
/// group-2 resources. The legacy [`env_module_wgsl`] stays standalone.
pub(crate) fn physical_env_module_wgsl() -> String {
    let mut s = physical_sky_module();
    s.push_str(SKY_ENV_SRC);
    s.push('\n');
    s
}

/// The `sky.background` module for the physical model: the same depth-1 fullscreen triangle
/// and seam, with the screen-cloud viewport-UV composition. The legacy builder stays
/// standalone.
pub(crate) fn physical_background_module_wgsl() -> String {
    let mut s = physical_sky_module();
    s.push_str(PHYSICAL_BACKGROUND_SRC);
    s.push('\n');
    s
}

/// Immutable celestial-map bake, entry `raa_milky_way_bake`, with empty groups 0/1
/// and one write-only Rgba8Unorm D2 storage output at group 2 binding 0.
pub(crate) fn milky_way_bake_module_wgsl() -> String {
    format!("{MILKY_WAY_SRC}\n{MILKY_WAY_BAKE_SRC}")
}

fn assemble(entry: &str) -> String {
    let mut s = String::with_capacity(
        PRELUDE_WGSL.len() + LIB_WGSL.len() + SKY_SRC.len() + entry.len() + 3,
    );
    for part in [PRELUDE_WGSL, LIB_WGSL, SKY_SRC, entry] {
        s.push_str(part);
        s.push('\n');
    }
    s
}

/// The `sky.env` module: `vs_main` draws a fullscreen triangle per cube face
/// (`instance_index` = face), `fs_main` writes the sky radiance to an HDR_FORMAT target.
#[allow(dead_code)] // consumed by SkyFeature (sky.env, sky.background) and tests/variants.rs
pub(crate) fn env_module_wgsl() -> String {
    assemble(SKY_ENV_SRC)
}

/// The `sky.background` module: fullscreen triangle at clip z = 1, per-pixel sky radiance
/// along the reconstructed world view ray.
#[allow(dead_code)] // consumed by SkyFeature (sky.env, sky.background) and tests/variants.rs
pub(crate) fn background_module_wgsl() -> String {
    assemble(SKY_BACKGROUND_SRC)
}

/// The three compute entries of [`atmosphere_luts_module_wgsl`], in build order: the static
/// transmittance table, the static multiple-scattering table and the per-view sky view.
pub(crate) const ATMOSPHERE_LUT_ENTRIES: [&str; 3] = [
    "raa_atmosphere_build_transmittance",
    "raa_atmosphere_build_multi_scatter",
    "raa_atmosphere_build_sky_view",
];

/// The fourth compute entry: the 16-byte directional transmission toward the dominant light,
/// written to the S10 `STORAGE | COPY_SRC` result after the table dispatches.
pub(crate) const ATMOSPHERE_DIRECTIONAL_ENTRY: &str =
    "raa_atmosphere_build_directional_transmission";

/// The atmosphere LUT compute module: the S23 helper source plus the four compute entries.
///
/// It deliberately composes without `PRELUDE_WGSL`/`LIB_WGSL`, so groups 0 and 1 are empty
/// and a pipeline layout only needs an explicit group 2 with the S10 private compute
/// resources: binding 0 `PhysicalSkyGpu` uniform (256 bytes), 1 transmittance LUT, 2
/// multiple-scattering LUT, 3 filtering sampler, 4 Rgba16Float storage output and 5 the
/// directional `STORAGE` result. The dispatched entry selects the table written to binding 4
/// or the 16-byte result written to binding 5.
pub(crate) fn atmosphere_luts_module_wgsl() -> String {
    let mut s = String::with_capacity(ATMOSPHERE_SRC.len() + ATMOSPHERE_LUTS_SRC.len() + 2);
    for part in [ATMOSPHERE_SRC, ATMOSPHERE_LUTS_SRC] {
        s.push_str(part);
        s.push('\n');
    }
    s
}

/// CPU port of `raa_sky_env_face_dir`: unit direction of face coordinates (u, v) in [-1, 1]
/// (u grows with the texel column, v with the row) on cube face `face` in wgpu order
/// +X, -X, +Y, -Y, +Z, -Z. Same mapping as raa-ibl's SH projection.
#[allow(dead_code)] // consumed by SkyFeature (sky.env, sky.background) and tests/variants.rs
pub(crate) fn cube_face_direction(face: u32, u: f32, v: f32) -> [f32; 3] {
    let d = match face {
        0 => [1.0, -v, -u],
        1 => [-1.0, -v, u],
        2 => [u, 1.0, v],
        3 => [u, -1.0, -v],
        4 => [u, -v, 1.0],
        _ => [-u, -v, -1.0],
    };
    // One component is +-1, so the length is >= 1 for finite (u, v).
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    [d[0] / l, d[1] / l, d[2] / l]
}
