//! M7 offset and single-shell foliage sampling (contract `m7-impostor-sampling`).
//!
//! `foliage_offset_hit` validates after the foundation prelude with a fragment caller and
//! keeps the declared group-2 layout and result shape. The hand-derived CPU twins of its
//! projection, source-view choice and one-plane intersection pin the zero-offset frontal
//! case, the oblique plane UV shift, the coverage cutout and the defined grazing and
//! empty-layer misses. No GPU is needed.

use raa_shader::{Composer, PRELUDE_WGSL};

const OFFSET_WGSL: &str = include_str!("../src/wgsl/foliage_offset.wgsl");

const FRAGMENT_CALLER: &str = "
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let offset = foliage_offset_hit(vec3<f32>(1.0, 0.0, -3.0), vec3<f32>(0.0, 0.0, -1.0), 2u);
    let shell = foliage_offset_hit(vec3<f32>(1.0, 0.0, -3.0), vec3<f32>(0.0, 0.0, -1.0), 3u);
    return vec4<f32>(offset.albedo + shell.albedo, max(offset.coverage, shell.coverage));
}
";

/// Column-major 4x4 matrix, as the WGSL `mat4x4<f32>` stores it.
type Mat4 = [[f32; 4]; 4];

/// A 90-degree perspective camera looking down -Z from the origin, with the standard-Z
/// mapping `clip.z = far / (near - far) * z + far * near / (near - far)` and `clip.w = -z`.
fn perspective(near: f32, far: f32) -> Mat4 {
    let a = far / (near - far);
    let b = far * near / (near - far);
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, a, -1.0],
        [0.0, 0.0, b, 0.0],
    ]
}

fn mul(m: &Mat4, v: [f32; 4]) -> [f32; 4] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3],
        m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

/// CPU twin of `raa_foliage_forward`.
fn forward(m: &Mat4) -> [f32; 3] {
    let x_row = [m[0][0], m[1][0], m[2][0]];
    let y_row = [m[0][1], m[1][1], m[2][1]];
    let z_row = [m[0][2], m[1][2], m[2][2]];
    let mut f = cross(x_row, y_row);
    if dot(f, z_row) < 0.0 {
        f = [-f[0], -f[1], -f[2]];
    }
    normalize(f)
}

/// CPU twin of `raa_foliage_uv` and `raa_foliage_depth`.
fn project(m: &Mat4, p: [f32; 3]) -> Option<([f32; 2], f32)> {
    let c = mul(m, [p[0], p[1], p[2], 1.0]);
    if c[3] <= 1e-6 {
        return None;
    }
    let uv = [c[0] / c[3] * 0.5 + 0.5, 0.5 - c[1] / c[3] * 0.5];
    Some((uv, (c[2] / c[3]).clamp(0.0, 1.0)))
}

/// CPU twin of `raa_foliage_plane_hit`: one clip-space step to `plane_depth`.
fn plane_hit(
    m: &Mat4,
    p: [f32; 3],
    direction: [f32; 3],
    plane_depth: f32,
) -> Option<([f32; 2], f32)> {
    let c0 = mul(m, [p[0], p[1], p[2], 1.0]);
    let c1 = mul(
        m,
        [
            p[0] + direction[0],
            p[1] + direction[1],
            p[2] + direction[2],
            1.0,
        ],
    );
    let denom = (c1[2] - c0[2]) - plane_depth * (c1[3] - c0[3]);
    if denom.abs() <= 1e-6 * c0[3].abs().max(1.0) {
        return None;
    }
    let distance = (plane_depth * c0[3] - c0[2]) / denom;
    let c: [f32; 4] = std::array::from_fn(|i| c0[i] + (c1[i] - c0[i]) * distance);
    if c[3] <= 1e-6 {
        return None;
    }
    Some(([c[0] / c[3] * 0.5 + 0.5, 0.5 - c[1] / c[3] * 0.5], distance))
}

fn close(actual: f32, expected: f32, what: &str) {
    assert!(
        (actual - expected).abs() <= 1e-5,
        "{what}: {actual} != {expected}"
    );
}

#[test]
fn module_validates_after_the_foundation_and_keeps_the_contract() {
    let wgsl = format!("{PRELUDE_WGSL}\n{OFFSET_WGSL}\n{FRAGMENT_CALLER}");
    Composer::validate(&wgsl).expect("prelude + foliage_offset.wgsl + fragment caller");

    assert!(OFFSET_WGSL.contains(
        "fn foliage_offset_hit(position: vec3<f32>, direction: vec3<f32>, mode: u32) -> FoliageHit"
    ));
    for declaration in [
        "@group(2) @binding(0) var<uniform> raa_foliage: RaaFoliageParams;",
        "@group(2) @binding(1) var raa_foliage_volume: texture_3d<f32>;",
        "@group(2) @binding(2) var raa_foliage_sampler: sampler;",
        "@group(2) @binding(3) var raa_foliage_albedo_coverage: texture_2d_array<f32>;",
        "@group(2) @binding(4) var raa_foliage_normal_depth: texture_2d_array<f32>;",
        "@group(2) @binding(5) var<uniform> raa_foliage_views: array<RaaFoliageView, 128>;",
    ] {
        assert!(OFFSET_WGSL.contains(declaration), "missing {declaration}");
    }

    let fields: Vec<&str> = OFFSET_WGSL
        .split_once("struct FoliageHit {")
        .expect("FoliageHit")
        .1
        .split_once('}')
        .expect("closing brace")
        .0
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.split(':').next().expect("field name"))
        .collect();
    assert_eq!(
        fields,
        ["position", "normal", "albedo", "coverage", "depth"]
    );
    for field in [
        "position: vec3<f32>",
        "normal: vec3<f32>",
        "albedo: vec3<f32>",
        "coverage: f32",
        "depth: f32",
    ] {
        assert!(OFFSET_WGSL.contains(field), "missing {field}");
    }

    // Mode 2 uses one baked-depth offset; mode 3 one shell plane through the fitted centre.
    assert!(OFFSET_WGSL.contains("const RAA_FOLIAGE_MODE_OFFSET: u32 = 2u;"));
    assert!(OFFSET_WGSL.contains("const RAA_FOLIAGE_MODE_SHELL: u32 = 3u;"));
    assert!(OFFSET_WGSL.contains(
        "var plane_depth = textureSampleLevel(raa_foliage_normal_depth, raa_foliage_sampler, uv0, choice.index, 0.0).w;"
    ));
    assert!(OFFSET_WGSL.contains("let centre = 0.5 * (raa_foliage.min + raa_foliage.max);"));
    assert!(
        OFFSET_WGSL
            .contains("plane_depth = raa_foliage_depth(raa_foliage_clip(view.matrix, centre));")
    );
    // The baked normal is independent of the hull face normal and is never derived from the ray.
    assert!(OFFSET_WGSL.contains("normal_depth.xyz / normal_len"));
    assert!(!OFFSET_WGSL.contains("@vertex"));
    // Explicit LOD only, and the linear unlit layer colour is used as-is.
    assert!(OFFSET_WGSL.contains("textureSampleLevel(raa_foliage_albedo_coverage"));
    assert!(!OFFSET_WGSL.contains("textureSample("));
    assert!(!OFFSET_WGSL.contains("pow("));
}

#[test]
fn frontal_incidence_offsets_nothing() {
    let view = perspective(1.0, 11.0);
    // On the source view axis, a fronto-parallel ray keeps the centre texel.
    let p = [0.0, 0.0, -3.0];
    let (uv0, _) = project(&view, p).expect("hull UV");
    let (uv, distance) = plane_hit(&view, p, [0.0, 0.0, -1.0], 0.88).expect("plane hit");
    close(distance, 2.0, "frontal distance");
    close(uv[0], uv0[0], "frontal u");
    close(uv[1], uv0[1], "frontal v");
    let hit = [p[0], p[1], p[2] - distance];
    close(hit[2], -5.0, "frontal hit z");
    close(project(&view, hit).expect("hit depth").1, 0.88, "hit depth");

    // Off axis, the ray through p from the source camera keeps the same projected UV too.
    let p = [1.0, 0.0, -3.0];
    let ray = normalize([1.0, 0.0, -3.0]);
    let (uv, distance) = plane_hit(&view, p, ray, 0.88).expect("camera-ray hit");
    close(
        distance,
        2.0 / (9.0f32 / 10.0f32).sqrt(),
        "camera-ray distance",
    );
    close(
        uv[0],
        project(&view, p).expect("hull UV").0[0],
        "camera-ray u",
    );
    close(
        uv[1],
        project(&view, p).expect("hull UV").0[1],
        "camera-ray v",
    );
}

#[test]
fn oblique_plane_shift_matches_the_hand_derivation() {
    let view = perspective(1.0, 11.0);
    let p = [1.0, 0.0, -3.0];
    let direction = normalize([0.6, 0.0, -0.8]);
    // From z = -3 to the plane z = -5 along -0.8 z per metre: s = 2 / 0.8 = 2.5.
    // The hit x is 1 + 0.6 * 2.5 = 2.5 at w = 5, so u = 2.5 / 5 * 0.5 + 0.5 = 0.75.
    let (uv, distance) = plane_hit(&view, p, direction, 0.88).expect("oblique hit");
    close(distance, 2.5, "oblique distance");
    close(uv[0], 0.75, "oblique u");
    close(uv[1], 0.5, "oblique v");
    let hit = [
        p[0] + direction[0] * distance,
        0.0,
        p[2] + direction[2] * distance,
    ];
    close(hit[0], 2.5, "oblique hit x");
    close(hit[2], -5.0, "oblique hit z");
    close(
        project(&view, hit).expect("hit depth").1,
        0.88,
        "oblique hit depth",
    );
    // The shift from the hull UV is the parallax the shifted sample resolves.
    close(
        uv[0] - project(&view, p).expect("hull UV").0[0],
        1.0 / 12.0,
        "u shift",
    );
}

#[test]
fn single_shell_uses_the_fitted_volume_centre_plane() {
    let min = [-2.0, -2.0, -7.0];
    let max = [2.0, 2.0, -3.0];
    let centre = [
        0.5 * (min[0] + max[0]),
        0.5 * (min[1] + max[1]),
        0.5 * (min[2] + max[2]),
    ];
    let view = perspective(1.0, 11.0);
    let centre_depth = project(&view, centre).expect("centre depth").1;
    close(centre_depth, 0.88, "centre shell depth");
    close(dot(forward(&view), [0.0, 0.0, -1.0]), 1.0, "camera forward");
    let (uv, _) =
        plane_hit(&view, [0.0, 0.0, -3.0], [0.0, 0.0, -1.0], centre_depth).expect("shell hit");
    close(uv[0], 0.5, "shell frontal u");
}

#[test]
fn grazing_rays_and_empty_layers_miss() {
    let view = perspective(1.0, 11.0);
    let grazing = [1.0, 0.0, 0.0];
    assert!(dot(forward(&view), grazing) <= 0.02, "grazing cosine");
    // The ray is parallel to the depth plane: the intersection is singular.
    assert!(plane_hit(&view, [0.0, 0.0, -3.0], grazing, 0.88).is_none());
    // The coverage cutout also cuts an empty texel with cutoff 0.
    let covers = |coverage: f32, cutoff: f32| coverage > 0.0 && coverage > cutoff;
    assert!(!covers(0.0, 0.0), "empty layer");
    assert!(!covers(0.25, 0.25), "at the cutoff");
    assert!(covers(0.5, 0.25), "above the cutoff");
    assert!(!covers(0.1, 0.25), "below the cutoff");
    assert!(OFFSET_WGSL.contains("return coverage > 0.0 && coverage > raa_foliage.cutoff;"));
    assert!(OFFSET_WGSL.contains("if (base.a <= 0.0) {"));
}
