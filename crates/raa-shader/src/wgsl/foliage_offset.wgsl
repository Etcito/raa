// M7 foliage offset and single-shell sampling (contract `m7-impostor-sampling`).
//
// `foliage_offset_hit` resolves one source view from the Rust-provided group-2 view table
// and returns the baked source-root hit, its independent baked normal, the linear unlit
// albedo and coverage, and the standard-Z depth in that source view. Mode 2 (offset)
// performs exactly one depth-based parallax offset: the baked depth at the hull UV selects
// a shell plane and one clip-space step lands on it. Mode 3 (single-shell) intersects one
// shell plane through the fitted volume centre and shifts the UV by the view. Both modes
// keep the convex hull geometry; no path builds a camera-facing quad or reads the hull face
// normal. A ray with no usable source view (grazing), a cleared or empty layer, or a shifted
// UV outside the layer returns the zero-coverage miss. Albedo stays linear and unlit: the
// packed layer is sampled as-is, with no sRGB conversion and no light computation.

/// Mode rank `offset` of `m7-foliage-resources`.
const RAA_FOLIAGE_MODE_OFFSET: u32 = 2u;
/// Mode rank `shell` of `m7-foliage-resources`.
const RAA_FOLIAGE_MODE_SHELL: u32 = 3u;
/// Records of the group-2 uniform view table (only the first `view_count` are read).
const RAA_FOLIAGE_VIEW_MAX: u32 = 128u;
const RAA_FOLIAGE_EPS: f32 = 1e-6;
/// A source-camera cosine below this (about 88.9 degrees) is a grazing miss.
const RAA_FOLIAGE_GRAZING_COS: f32 = 0.02;

/// The impostor sample result. `position` and `normal` are source-root; `depth` is the
/// standard-Z depth in the selected source view; `coverage` 0 denotes a miss.
struct FoliageHit {
    position: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    coverage: f32,
    depth: f32,
}

/// Binding 0 of group 2: the 64-byte record of `FoliageParams::to_bytes`.
struct RaaFoliageParams {
    min: vec3<f32>,
    resolution: u32,
    max: vec3<f32>,
    mode: u32,
    translucency: f32,
    generation: u32,
    cutoff: f32,
    view_count: u32,
    texture_resolution: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

/// One 80-byte view-table record: column-major root-to-clip matrix and near/far.
struct RaaFoliageView {
    matrix: mat4x4<f32>,
    near_far: vec4<f32>,
}

@group(2) @binding(0) var<uniform> raa_foliage: RaaFoliageParams;
@group(2) @binding(1) var raa_foliage_volume: texture_3d<f32>;
@group(2) @binding(2) var raa_foliage_sampler: sampler;
@group(2) @binding(3) var raa_foliage_albedo_coverage: texture_2d_array<f32>;
@group(2) @binding(4) var raa_foliage_normal_depth: texture_2d_array<f32>;
@group(2) @binding(5) var<uniform> raa_foliage_views: array<RaaFoliageView, 128>;

/// The zero-coverage miss: no position, normal, albedo or depth.
fn raa_foliage_miss() -> FoliageHit {
    return FoliageHit(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0.0);
}

/// The coverage cutout: a texel cuts when it is empty or not above `cutoff`.
fn raa_foliage_covers(coverage: f32) -> bool {
    return coverage > 0.0 && coverage > raa_foliage.cutoff;
}

fn raa_foliage_clip(matrix: mat4x4<f32>, p: vec3<f32>) -> vec4<f32> {
    return matrix * vec4<f32>(p, 1.0);
}

/// Framebuffer UV of a clip position: x right, y down (the clear row is the top).
fn raa_foliage_uv(clip: vec4<f32>) -> vec2<f32> {
    let inv_w = 1.0 / clip.w;
    return vec2<f32>(clip.x * inv_w * 0.5 + 0.5, 0.5 - clip.y * inv_w * 0.5);
}

fn raa_foliage_depth(clip: vec4<f32>) -> f32 {
    return clamp(clip.z / max(clip.w, RAA_FOLIAGE_EPS), 0.0, 1.0);
}

/// The source camera's viewing direction in source-root metres: the null direction of the
/// clip x and y rows, signed so that the clip z gradient grows into the scene.
fn raa_foliage_forward(matrix: mat4x4<f32>) -> vec3<f32> {
    let x_row = vec3<f32>(matrix[0][0], matrix[1][0], matrix[2][0]);
    let y_row = vec3<f32>(matrix[0][1], matrix[1][1], matrix[2][1]);
    let z_row = vec3<f32>(matrix[0][2], matrix[1][2], matrix[2][2]);
    var forward = cross(x_row, y_row);
    if (dot(forward, z_row) < 0.0) {
        forward = -forward;
    }
    let len = length(forward);
    if (len <= RAA_FOLIAGE_EPS) {
        return vec3<f32>(0.0);
    }
    return forward / len;
}

/// The source view whose camera looks most nearly along `direction` while covering
/// `position`; records are scanned in table order and the first of equal scores wins.
struct RaaFoliageViewChoice {
    index: i32,
    score: f32,
    hit: bool,
}

fn raa_foliage_choose_view(position: vec3<f32>, direction: vec3<f32>) -> RaaFoliageViewChoice {
    var best = RaaFoliageViewChoice(-1, -2.0, false);
    let count = min(raa_foliage.view_count, RAA_FOLIAGE_VIEW_MAX);
    for (var i = 0u; i < count; i = i + 1u) {
        let view = raa_foliage_views[i];
        let score = dot(raa_foliage_forward(view.matrix), direction);
        if (score <= RAA_FOLIAGE_GRAZING_COS || score <= best.score) {
            continue;
        }
        let clip = raa_foliage_clip(view.matrix, position);
        if (clip.w <= RAA_FOLIAGE_EPS) {
            continue;
        }
        let uv = raa_foliage_uv(clip);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            continue;
        }
        best = RaaFoliageViewChoice(i32(i), score, true);
    }
    return best;
}

/// The one plane intersection shared by both modes: the clip-space line through
/// `position` along `direction` meets the standard-Z plane `plane_depth`.
struct RaaFoliagePlaneHit {
    uv: vec2<f32>,
    distance: f32,
    hit: bool,
}

fn raa_foliage_plane_hit(view: RaaFoliageView, position: vec3<f32>, direction: vec3<f32>, plane_depth: f32) -> RaaFoliagePlaneHit {
    let clip0 = raa_foliage_clip(view.matrix, position);
    let clip1 = raa_foliage_clip(view.matrix, position + direction);
    let denom = (clip1.z - clip0.z) - plane_depth * (clip1.w - clip0.w);
    let scale = max(abs(clip0.w), 1.0);
    if (abs(denom) <= RAA_FOLIAGE_EPS * scale) {
        return RaaFoliagePlaneHit(vec2<f32>(0.0), 0.0, false);
    }
    let distance = (plane_depth * clip0.w - clip0.z) / denom;
    let clip = clip0 + (clip1 - clip0) * distance;
    if (clip.w <= RAA_FOLIAGE_EPS) {
        return RaaFoliagePlaneHit(vec2<f32>(0.0), 0.0, false);
    }
    return RaaFoliagePlaneHit(raa_foliage_uv(clip), distance, true);
}

/// Handoff: 'Compose relightable foliage materials' calls `foliage_offset_hit`, which today
/// resolves one source view from the group-2 table and returns the offset (mode 2) or
/// single-shell (mode 3) sample with the baked normal, linear unlit albedo/coverage and the
/// standard-Z source-view depth.
///
/// Samples the packed impostor of the convex hull at `position` along `direction`.
/// `direction` is normalized here before use. `mode` 2 offsets the layer UV once from the
/// baked depth at the hull UV; `mode` 3 intersects one shell plane through the fitted
/// volume centre. Any other mode, a grazing ray, an empty or cleared layer, a parallax step
/// before the hull or beyond the fitted volume, and a shifted UV outside the layer all
/// return `raa_foliage_miss()`.
fn foliage_offset_hit(position: vec3<f32>, direction: vec3<f32>, mode: u32) -> FoliageHit {
    if (mode != RAA_FOLIAGE_MODE_OFFSET && mode != RAA_FOLIAGE_MODE_SHELL) {
        return raa_foliage_miss();
    }
    let dir_len = length(direction);
    if (dir_len <= RAA_FOLIAGE_EPS) {
        return raa_foliage_miss();
    }
    let dir = direction / dir_len;
    let choice = raa_foliage_choose_view(position, dir);
    if (!choice.hit) {
        return raa_foliage_miss();
    }
    let view = raa_foliage_views[u32(choice.index)];
    let uv0 = raa_foliage_uv(raa_foliage_clip(view.matrix, position));
    let base = textureSampleLevel(raa_foliage_albedo_coverage, raa_foliage_sampler, uv0, choice.index, 0.0);
    if (base.a <= 0.0) {
        return raa_foliage_miss();
    }
    var plane_depth = textureSampleLevel(raa_foliage_normal_depth, raa_foliage_sampler, uv0, choice.index, 0.0).w;
    if (mode == RAA_FOLIAGE_MODE_SHELL) {
        let centre = 0.5 * (raa_foliage.min + raa_foliage.max);
        plane_depth = raa_foliage_depth(raa_foliage_clip(view.matrix, centre));
    }
    if (plane_depth >= 1.0 - RAA_FOLIAGE_EPS) {
        return raa_foliage_miss();
    }
    let plane = raa_foliage_plane_hit(view, position, dir, plane_depth);
    if (!plane.hit) {
        return raa_foliage_miss();
    }
    let reach = length(raa_foliage.max - raa_foliage.min);
    if (plane.distance < 0.0 || plane.distance > reach) {
        return raa_foliage_miss();
    }
    if (plane.uv.x < 0.0 || plane.uv.x > 1.0 || plane.uv.y < 0.0 || plane.uv.y > 1.0) {
        return raa_foliage_miss();
    }
    let albedo_coverage = textureSampleLevel(raa_foliage_albedo_coverage, raa_foliage_sampler, plane.uv, choice.index, 0.0);
    if (!raa_foliage_covers(albedo_coverage.a)) {
        return raa_foliage_miss();
    }
    let normal_depth = textureSampleLevel(raa_foliage_normal_depth, raa_foliage_sampler, plane.uv, choice.index, 0.0);
    let normal_len = length(normal_depth.xyz);
    if (normal_len <= RAA_FOLIAGE_EPS) {
        return raa_foliage_miss();
    }
    return FoliageHit(
        position + dir * plane.distance,
        normal_depth.xyz / normal_len,
        albedo_coverage.rgb,
        albedo_coverage.a,
        plane_depth,
    );
}
