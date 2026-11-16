// M7 ReSTIR DI for the area-light shadows (S12 order 43), common share 1 of contract
// 'area-restir-wgsl': the group-2 bindings 12-19, the `RestirParamsGpu` and
// `RaaRestirReservoir` records, and the reservoir codec and reuse helpers every ReSTIR stage
// calls. This is RAA's own code from Bitterli et al. 2020 (ReSTIR), Wyman and Panteleev,
// HPG 2021, and Lin et al. 2022 (GRIS). Bevy Solari's restir.wesl is not among the supplied
// checkouts and no URL for it is given, so nothing is adapted or copied from it and no
// licence header applies. NVIDIA RTXDI is never read or copied.
//
// Every function mirrors its CPU twin of contract 'area-restir-cpu' (module
// raa_gi::parts::area_shadow::restir_cpu); the WGSL names keep the same names with the
// `raa_restir_` prefix. `RaaTemporalFrame` comes from raa_taa::REPROJECT_WGSL, which every
// composed stage module includes; share 2 (after the share-1 helpers) holds the surface,
// spherical-rectangle sampling, target and LTC luminance functions.
//
// Group-2 binding registry of contract 'area-restir-wgsl' (binding number to resource):
// 12 raa_restir_motion (S12 motion), 13 raa_restir_frame: RaaTemporalFrame (only temporal.z is
// read), 14 raa_restir_previous_depth (the depth pair's read view), 15 raa_restir_params:
// RestirParamsGpu (64 B), 16 raa_restir_current (reservoir storage, read_write), 17
// raa_restir_history (reservoir history, read_write), 18 raa_restir_ltc (filterable atlas),
// 19 raa_restir_material (S12 material). Bindings 0-11 belong to the composed stage files
// (restir_initial.wgsl, restir_temporal.wgsl, restir_spatial.wgsl): 0 is the 432-byte
// GiFrameGpu this file reads through raa_gi_frame; share 2 also reads 1 raa_gi_depth, 2
// raa_gi_normal and 8 raa_area_table. Binding 9 is not in this layout.

// The per-frame ReSTIR parameters (64 B), little-endian. `frame_index` is the ReSTIR frame
// counter; `history_usable` is 1 when raa_restir_history holds a committed frame of this
// allocation; `remap` lane s is the current slot of previous slot s, or 0xFFFFFFFF.
struct RestirParamsGpu {
    frame_index: u32,
    history_usable: u32,
    candidates: u32,
    spatial_samples: u32,
    remap: array<vec4<u32>, 2>,
    spatial_radius: f32,
    m_cap: f32,
    boiling_factor: f32,
    normal_cos: f32,
}

// One reservoir in registers: the stable area-stochastic slot (0..7, 0xFF empty, bits 8-31 of
// the packed word are zero), the point in the light rectangle's unit square, the unbiased
// contribution weight W, the confidence M and the octahedral world normal of the owner pixel.
struct RaaRestirReservoir {
    slot: u32,
    uv: vec2<f32>,
    weight: f32,
    confidence: f32,
    normal_oct: vec2<f32>,
}

// The S12 motion target of the pixel (the lit frame's reprojection input).
@group(2) @binding(12) var raa_restir_motion: texture_2d<f32>;
// The lit FrameGpu (S10) of this frame; only temporal.z (camera history valid) is read.
@group(2) @binding(13) var<uniform> raa_restir_frame: RaaTemporalFrame;
// The previous frame's linear view depth (the depth pair's read view).
@group(2) @binding(14) var raa_restir_previous_depth: texture_2d<f32>;
// The per-frame ReSTIR parameters (64 B).
@group(2) @binding(15) var<uniform> raa_restir_params: RestirParamsGpu;
// This frame's reservoir records, one vec4<u32> per pixel at index y * width + x.
@group(2) @binding(16) var<storage, read_write> raa_restir_current: array<vec4<u32>>;
// The previous committed frame's reservoir records (next frame's history).
@group(2) @binding(17) var<storage, read_write> raa_restir_history: array<vec4<u32>>;
// The part's own BRDF/LTC atlas from raa_pipeline::create_brdf_atlas, read through
// `raa_ltc_uv` with the group-1 `raa_linear_clamp` sampler.
@group(2) @binding(18) var raa_restir_ltc: texture_2d<f32>;
// The S12 material target: sRGB-encoded F0 rgb and metallic a.
@group(2) @binding(19) var raa_restir_material: texture_2d<f32>;

// The empty reservoir: slot 0xFF and every other field zero.
fn raa_restir_empty() -> RaaRestirReservoir {
    return RaaRestirReservoir(0xFFu, vec2<f32>(0.0), 0.0, 0.0, vec2<f32>(0.0));
}

// Packs a reservoir into its four WGSL words: x = slot & 0xFF, y = pack2x16float(uv in the
// unit square), z = pack2x16float((W, M) in [0, 65504], NaN -> 0) and w =
// pack2x16float(normal_oct).
fn raa_restir_pack(r: RaaRestirReservoir) -> vec4<u32> {
    return vec4<u32>(
        r.slot & 0xFFu,
        pack2x16float(vec2<f32>(
            raa_restir_uv_clamped(r.uv.x),
            raa_restir_uv_clamped(r.uv.y),
        )),
        pack2x16float(vec2<f32>(
            raa_restir_half_clamped(r.weight),
            raa_restir_half_clamped(r.confidence),
        )),
        pack2x16float(r.normal_oct),
    );
}

// Unpacks the four WGSL words of `raa_restir_pack`; the slot keeps its low eight bits.
fn raa_restir_unpack(v: vec4<u32>) -> RaaRestirReservoir {
    let weight_confidence = unpack2x16float(v.z);
    return RaaRestirReservoir(
        v.x & 0xFFu,
        unpack2x16float(v.y),
        weight_confidence.x,
        weight_confidence.y,
        unpack2x16float(v.w),
    );
}

// The reservoir storage index of `pixel`: y * width + x, width = u32(viewport.x).
fn raa_restir_index(pixel: vec2<u32>) -> u32 {
    return pixel.y * u32(raa_gi_frame.viewport.x) + pixel.x;
}

// The current slot of the previous slot `slot` from raa_restir_params.remap; 0xFFFFFFFF for a
// slot >= 8 or a lane that did not survive.
fn raa_restir_remap(slot: u32) -> u32 {
    if slot >= 8u {
        return 0xFFFFFFFFu;
    }
    return raa_restir_params.remap[slot / 4u][slot % 4u];
}

// ---------------------------------------------------------------------------------------------
// Share 2 of contract 'area-restir-wgsl': the receiver surface, the slot-table light lookup,
// Urena et al. EGSR 2013 area-preserving spherical-rectangle sampling, the area PDF, the
// Lambert + GGX target and the unshadowed LTC luminance U_i. The stage-less validation module
// that composes this file is crates/raa-gi/src/parts/area_shadow/restir.rs:80
// `restir_common_module_wgsl`, the Handoff of card
// 'raa-gi: ReSTIR group-2 layout, params and reservoir codec WGSL (restir_common share 1)'.
// No share-2 function reads group 0 (the ReSTIR entries run with an empty group 0); group 1
// use is 6 raa_linear_clamp and 12 raa_local_lights. Every division is guarded so no NaN or
// infinity is produced.

const RAA_RESTIR_PI: f32 = 3.14159265359;
// The largest finite f32: `x <= RAA_RESTIR_F32_MAX` is false for +infinity and NaN. Written
// exactly as a hex float: the decimal 3.40282347e38 lies above it, which naga rounds down but
// Tint (Chrome) refuses ("cannot be represented as 'f32'"), invalidating every ReSTIR module.
const RAA_RESTIR_F32_MAX: f32 = 0x1.fffffep+127f;
// The GGX roughness floor of the target, ROUGHNESS_MIN of contract 'area-restir-cpu'.
const RAA_RESTIR_ROUGHNESS_MIN: f32 = 0.0525;

// One receiver as the prepass publishes it: world position, world unit normal, unit vector to
// the camera, roughness (normal target w), linear F0 (decoded material rgb), metallic
// (material a) and the linear view depth. `valid` is false outside the viewport or on a far
// depth; every other field is then zero.
struct RaaRestirSurface {
    position: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
    roughness: f32,
    f0: vec3<f32>,
    metallic: f32,
    view_depth: f32,
    valid: bool,
}

// One spherical-rectangle sample: the point in the light rectangle's unit square (area
// measure, (0, 0) at the corner position - T*w/2 - B*h/2) and the rectangle's solid angle in
// steradians seen from the sampling origin (0 for a degenerate rectangle).
struct RaaRestirRectSample {
    uv: vec2<f32>,
    solid_angle: f32,
}

// The local frame of a spherical rectangle (Urena et al. EGSR 2013), the twin of the CPU
// `RectFrame`: edge lengths, the corner coordinates in the rectangle's (x, y, z) frame, the z
// components of the edge normals n0 and n2, the angle sum k and the solid angle. `valid` is
// false for a degenerate rectangle or an origin in its plane.
struct RaaRestirRectFrame {
    exl: f32,
    eyl: f32,
    z0: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    b0: f32,
    b1: f32,
    k: f32,
    solid_angle: f32,
    valid: bool,
}

// `num / den`, or 0 when `den` is not positive and finite (the CPU twin's `ratio`).
fn raa_restir_ratio(num: f32, den: f32) -> f32 {
    if den > 0.0 && den <= RAA_RESTIR_F32_MAX {
        return num / den;
    }
    return 0.0;
}

// `a` normalised, or the zero vector when it has no positive finite length.
fn raa_restir_normalized(a: vec3<f32>) -> vec3<f32> {
    let len = length(a);
    if len > 0.0 && len <= RAA_RESTIR_F32_MAX {
        return a / len;
    }
    return vec3<f32>(0.0);
}

// The receiver of `pixel` from the S12 depth, normal and material targets; invalid outside
// the viewport, on a far depth (raa_depth_is_far) or on a zero normal.
//
// Handoff: card 'raa-gi: ReSTIR temporal reuse kernel with slot remap, M clamp and boiling filter'
// takes `raa_restir_surface` and `raa_restir_target` from this file: today
// raa_restir_surface returns the receiver of a pixel with its view_depth (valid false outside
// the viewport, on a far depth or a zero normal) and raa_restir_target returns the Lambert +
// GGX target of a light point at that receiver.
fn raa_restir_surface(pixel: vec2<i32>) -> RaaRestirSurface {
    var surface: RaaRestirSurface;
    surface.valid = false;
    let viewport = vec2<i32>(raa_gi_frame.viewport.xy);
    if any(pixel < vec2<i32>(0)) || any(pixel >= viewport) {
        return surface;
    }
    let depth = textureLoad(raa_gi_depth, pixel, 0);
    if raa_depth_is_far(depth) {
        return surface;
    }
    let normal_target = textureLoad(raa_gi_normal, pixel, 0);
    if !(dot(normal_target.xyz, normal_target.xyz) > 0.0) {
        return surface;
    }
    let material = textureLoad(raa_restir_material, pixel, 0);
    let centre = vec2<f32>(pixel) + vec2<f32>(0.5);
    surface.position = raa_gi_world_position(centre, depth);
    surface.normal = raa_gi_world_normal(pixel);
    surface.roughness = normal_target.w;
    surface.f0 = raa_restir_srgb_decode(material.rgb);
    surface.metallic = material.a;
    let to_camera = raa_restir_normalized(raa_gi_frame.camera_position.xyz - surface.position);
    surface.view = select(surface.normal, to_camera, dot(to_camera, to_camera) > 0.0);
    surface.view_depth = -raa_gi_view_position(centre, depth).z;
    surface.valid = true;
    return surface;
}

// The packed raa_local_lights index of mid-tier slot `slot`: the slot-table lane when the slot
// is below min(raa_area_table.params.x, 8), the lane is not 0xFFFFFFFF and the light is an
// area light (kind 3); else 0xFFFFFFFF.
fn raa_restir_slot_light(slot: u32) -> u32 {
    if slot >= min(raa_area_table.params.x, 8u) {
        return 0xFFFFFFFFu;
    }
    let index = raa_area_table.lights[slot / 4u][slot % 4u];
    if index == 0xFFFFFFFFu || index >= MAX_LOCAL_LIGHTS {
        return 0xFFFFFFFFu;
    }
    if raa_local_lights[index].kind != 3u {
        return 0xFFFFFFFFu;
    }
    return index;
}

// The world point of `uv` in the light rectangle: corner + T*w*u + B*h*v with the corner
// position - T*w/2 - B*h/2, T and B the raa_area_basis tangent and bitangent.
//
// Handoff: card
// 'raa-gi: ReSTIR spatial reuse and resolve kernel (GRIS, second visibility ray, raw S and U)'
// takes `raa_restir_light_point`, `raa_restir_slot_light`, `raa_restir_target` and
// `raa_restir_ltc_luminance` from this file: today raa_restir_light_point returns the world
// point of a reservoir uv on the light of the packed index raa_restir_slot_light returns, and
// raa_restir_ltc_luminance returns the unshadowed LTC luminance U_i of that light.
fn raa_restir_light_point(light: LightGpu, uv: vec2<f32>) -> vec3<f32> {
    let basis = raa_area_basis(light.direction_cos_outer.xyz);
    let size = vec2<f32>(max(light.width, 0.0), max(light.height, 0.0));
    return light.position_range.xyz
        + basis[0] * ((uv.x - 0.5) * size.x)
        + basis[1] * ((uv.y - 0.5) * size.y);
}

// The Karis representative point: the rectangle point nearest `position`. The 6-line clamp
// of raa_area_representative, crates/raa-shader/src/wgsl/local_lighting.wgsl:25-32, which is
// not composed into GI compute modules (LOCAL_LIGHTING_WGSL is a forward-pass library).
fn raa_restir_representative(light: LightGpu, position: vec3<f32>) -> vec3<f32> {
    let basis = raa_area_basis(light.direction_cos_outer.xyz);
    let offset = position - light.position_range.xyz;
    let half_extent = vec2<f32>(max(light.width, 0.0), max(light.height, 0.0)) * 0.5;
    let along_t = clamp(dot(offset, basis[0]), -half_extent.x, half_extent.x);
    let along_b = clamp(dot(offset, basis[1]), -half_extent.y, half_extent.y);
    return light.position_range.xyz + basis[0] * along_t + basis[1] * along_b;
}

// The rectangle frame of `corner + [0, 1] edge_x + [0, 1] edge_y` seen from `origin`, the twin
// of the CPU `rect_frame`: invalid for a zero or non-finite edge, a non-finite plane distance,
// an origin in the rectangle's plane (z0 == 0) or a solid angle that is not positive.
fn raa_restir_rect_frame(
    origin: vec3<f32>,
    corner: vec3<f32>,
    edge_x: vec3<f32>,
    edge_y: vec3<f32>,
) -> RaaRestirRectFrame {
    var frame: RaaRestirRectFrame;
    frame.valid = false;
    let exl = length(edge_x);
    let eyl = length(edge_y);
    if !(exl > 0.0 && exl <= RAA_RESTIR_F32_MAX && eyl > 0.0 && eyl <= RAA_RESTIR_F32_MAX) {
        return frame;
    }
    let x = edge_x / exl;
    let y = edge_y / eyl;
    let z = cross(x, y);
    let d = corner - origin;
    var z0 = dot(d, z);
    if !(abs(z0) <= RAA_RESTIR_F32_MAX) {
        return frame;
    }
    if z0 > 0.0 {
        z0 = -z0;
    }
    if z0 == 0.0 {
        return frame;
    }
    let x0 = dot(d, x);
    let y0 = dot(d, y);
    let x1 = x0 + exl;
    let y1 = y0 + eyl;
    let n0 = raa_restir_normalized(cross(vec3<f32>(x0, y0, z0), vec3<f32>(x1, y0, z0)));
    let n1 = raa_restir_normalized(cross(vec3<f32>(x1, y0, z0), vec3<f32>(x1, y1, z0)));
    let n2 = raa_restir_normalized(cross(vec3<f32>(x1, y1, z0), vec3<f32>(x0, y1, z0)));
    let n3 = raa_restir_normalized(cross(vec3<f32>(x0, y1, z0), vec3<f32>(x0, y0, z0)));
    let g0 = acos(clamp(-dot(n0, n1), -1.0, 1.0));
    let g1 = acos(clamp(-dot(n1, n2), -1.0, 1.0));
    let g2 = acos(clamp(-dot(n2, n3), -1.0, 1.0));
    let g3 = acos(clamp(-dot(n3, n0), -1.0, 1.0));
    let k = RAA_RESTIR_TAU - g2 - g3;
    let solid_angle = g0 + g1 - k;
    if !(solid_angle > 0.0 && solid_angle <= RAA_RESTIR_F32_MAX) {
        return frame;
    }
    return RaaRestirRectFrame(exl, eyl, z0, x0, y0, x1, y1, n0.z, n2.z, k, solid_angle, true);
}

// Area-preserving spherical-rectangle sampling (Urena et al. EGSR 2013) of the light
// rectangle seen from `origin`: corner = position - T*w/2 - B*h/2, edge_x = T*w, edge_y = B*h
// (T, B the raa_area_basis tangent and bitangent). Returns the unit-square uv of `xi` (each
// lane clamped to [0, 1]) and the solid angle; a degenerate rectangle or an origin in its
// plane returns uv (0.5, 0.5) and solid angle 0. The steps are those of the CPU twin
// `sample_rect`; every division is guarded, so the result is never NaN.
//
// Handoff: card 'raa-gi: ReSTIR initial candidates kernel with visibility reuse' takes
// `raa_restir_sample_rect`, `raa_restir_area_pdf`, `raa_restir_ltc_luminance`,
// `raa_restir_slot_light`, `raa_restir_surface` and `raa_restir_target` from this file: today
// raa_restir_sample_rect returns one area-preserving sample (uv, solid angle) of a slot light
// and raa_restir_area_pdf converts its solid angle to the area measure of the reservoir uv.
// This file's composition is the Handoff at crates/raa-gi/src/parts/area_shadow/restir.rs:80
// `restir_common_module_wgsl`.
fn raa_restir_sample_rect(
    light: LightGpu,
    origin: vec3<f32>,
    xi: vec2<f32>,
) -> RaaRestirRectSample {
    let basis = raa_area_basis(light.direction_cos_outer.xyz);
    let edge_x = basis[0] * max(light.width, 0.0);
    let edge_y = basis[1] * max(light.height, 0.0);
    let corner = light.position_range.xyz - 0.5 * edge_x - 0.5 * edge_y;
    let frame = raa_restir_rect_frame(origin, corner, edge_x, edge_y);
    if !frame.valid {
        return RaaRestirRectSample(vec2<f32>(0.5), 0.0);
    }
    let u = clamp(xi.x, 0.0, 1.0);
    let v = clamp(xi.y, 0.0, 1.0);
    let au = u * frame.solid_angle + frame.k;
    var sin_au = sin(au);
    if abs(sin_au) < 1e-9 {
        sin_au = select(1e-9f, -1e-9f, sin_au < 0.0);
    }
    let fu = (cos(au) * frame.b0 - frame.b1) / sin_au;
    // sign(fu) / sqrt(fu^2 + b0^2) clamped to [-1, 1]: a denominator <= 1 gives +-1 after the
    // clamp, so dividing by max(denominator, 1) is the same value and never divides by 0.
    let cu = raa_restir_sign_one(fu) / max(sqrt(fu * fu + frame.b0 * frame.b0), 1.0);
    let root = sqrt(max(1.0 - cu * cu, 0.0));
    var xu = select(frame.x0, frame.x1, cu >= 0.0);
    if root > 0.0 {
        xu = -(cu * frame.z0) / root;
    }
    xu = clamp(xu, frame.x0, frame.x1);
    let dd = sqrt(xu * xu + frame.z0 * frame.z0);
    let h0 = raa_restir_ratio(frame.y0, sqrt(dd * dd + frame.y0 * frame.y0));
    let h1 = raa_restir_ratio(frame.y1, sqrt(dd * dd + frame.y1 * frame.y1));
    let hv = h0 + v * (h1 - h0);
    var yv = frame.y1;
    if hv * hv < 1.0 - 1e-6 {
        yv = hv * dd / sqrt(1.0 - hv * hv);
    }
    let uv = vec2<f32>(
        clamp((xu - frame.x0) / frame.exl, 0.0, 1.0),
        clamp((yv - frame.y0) / frame.eyl, 0.0, 1.0),
    );
    return RaaRestirRectSample(uv, frame.solid_angle);
}

// The area PDF of a light point sampled by solid angle: cos_light / (distance^2 *
// solid_angle); 0 when any input is not positive or the result is not finite.
fn raa_restir_area_pdf(solid_angle: f32, cos_light: f32, distance: f32) -> f32 {
    if !(solid_angle > 0.0 && cos_light > 0.0 && distance > 0.0) {
        return 0.0;
    }
    let pdf = raa_restir_ratio(cos_light, distance * distance * solid_angle);
    if !(pdf <= RAA_RESTIR_F32_MAX) {
        return 0.0;
    }
    return max(pdf, 0.0);
}

// The ReSTIR target of the light point `uv` at `surface`, the CPU twin `target_luminance`:
// luminance(radiance * (white Lambert (1 - metallic) / pi + GGX D * Vis * F)) * nl * cos_y /
// r^2 * window, with radiance = raa_area_radiance(light), travel = the light direction and
// window = raa_area_window(r, range). 0 for an invalid surface, a coincident point, a receiver
// facing away (nl <= 0) or a point behind the emitter (cos_y <= 0).
fn raa_restir_target(surface: RaaRestirSurface, light: LightGpu, uv: vec2<f32>) -> f32 {
    if !surface.valid {
        return 0.0;
    }
    let d = raa_restir_light_point(light, uv) - surface.position;
    let r = length(d);
    if !(r > 0.0 && r <= RAA_RESTIR_F32_MAX) {
        return 0.0;
    }
    let l = d / r;
    let nl = dot(surface.normal, l);
    let cos_y = dot(light.direction_cos_outer.xyz, -l);
    if !(nl > 0.0 && cos_y > 0.0) {
        return 0.0;
    }
    let nv = max(dot(surface.normal, surface.view), 1e-4);
    let h = raa_restir_normalized(l + surface.view);
    let nh = max(dot(surface.normal, h), 0.0);
    let vh = max(dot(surface.view, h), 0.0);
    let rough = max(surface.roughness, RAA_RESTIR_ROUGHNESS_MIN);
    let alpha = rough * rough;
    let alpha2 = alpha * alpha;
    let spec_base = nh * nh * (alpha2 - 1.0) + 1.0;
    let distribution = raa_restir_ratio(alpha2, RAA_RESTIR_PI * spec_base * spec_base);
    let vis_den = nl * sqrt(nv * nv * (1.0 - alpha2) + alpha2)
        + nv * sqrt(nl * nl * (1.0 - alpha2) + alpha2);
    let visibility = raa_restir_ratio(0.5, vis_den);
    let diffuse = (1.0 - surface.metallic) / RAA_RESTIR_PI;
    let f1 = clamp(1.0 - vh, 0.0, 1.0);
    let fresnel = f1 * f1 * f1 * f1 * f1;
    let spec = distribution * visibility * (surface.f0 + (vec3<f32>(1.0) - surface.f0) * fresnel);
    let scaled = raa_area_radiance(light) * (vec3<f32>(diffuse) + spec);
    let window = raa_area_window(r, light.position_range.w);
    let value = raa_restir_ratio(raa_restir_luminance(scaled) * nl * cos_y, r * r) * window;
    if !(abs(value) <= RAA_RESTIR_F32_MAX) {
        return 0.0;
    }
    return value;
}

// U_i, the unshadowed LTC luminance of `light` at `surface`: the forward raa_area_ltc
// (crates/raa-shader/src/wgsl/local_lighting.wgsl:45-69, Heitz, Dupuy, Hill, Neubelt,
// SIGGRAPH 2016) evaluated from the prepass data. (1 - metallic) white Lambert times
// raa_area_form_factor plus the LTC GGX term with the fresnel f0 * b.x + (1 - f0) * b.y, both
// LTC tables read from raa_restir_ltc (group 2 binding 18) with raa_linear_clamp through
// raa_ltc_uv and textureSampleLevel only, times raa_area_radiance and the raa_area_window of
// the distance to the rectangle point nearest the surface, as luminance. 0 for an invalid
// surface or a surface behind the emitter.
fn raa_restir_ltc_luminance(surface: RaaRestirSurface, light: LightGpu) -> f32 {
    if !surface.valid {
        return 0.0;
    }
    if !(dot(surface.position - light.position_range.xyz, light.direction_cos_outer.xyz) > 0.0) {
        return 0.0;
    }
    let n = surface.normal;
    let v = surface.view;
    let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
    let roughness = max(surface.roughness, RAA_RESTIR_ROUGHNESS_MIN);
    var t1 = v - n * dot(v, n);
    if dot(t1, t1) < 1e-8 {
        t1 = raa_area_basis(n)[0];
    }
    t1 = normalize(t1);
    let t2 = cross(n, t1);
    let world_to_local = transpose(mat3x3<f32>(t1, t2, n));
    let a = textureSampleLevel(raa_restir_ltc, raa_linear_clamp,
        raa_ltc_uv(0u, roughness, n_dot_v), 0.0);
    let b = textureSampleLevel(raa_restir_ltc, raa_linear_clamp,
        raa_ltc_uv(1u, roughness, n_dot_v), 0.0);
    let m_inv = mat3x3<f32>(vec3<f32>(a.x, 0.0, a.y), vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(a.z, 0.0, a.w));
    let corners = raa_area_corners(light, surface.position);
    let fs = raa_area_vector_form_factor(corners, m_inv * world_to_local);
    let specular = raa_area_clipped(length(fs), fs.z);
    let diffuse = raa_area_form_factor(light, surface.position, n);
    let fresnel = surface.f0 * b.x + (vec3<f32>(1.0) - surface.f0) * b.y;
    let nearest = raa_restir_representative(light, surface.position);
    let window = raa_area_window(distance(nearest, surface.position), light.position_range.w);
    let shaded = raa_area_radiance(light)
        * (vec3<f32>(1.0 - surface.metallic) * diffuse + fresnel * specular) * window;
    let value = raa_restir_luminance(shaded);
    if !(value > 0.0 && value <= RAA_RESTIR_F32_MAX) {
        return 0.0;
    }
    return value;
}
