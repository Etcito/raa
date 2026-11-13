//! Q11: SKY_WGSL composes after the S11 prelude and helper library and passes naga validation,
//! every function it defines is the contract function or a `raa_sky_` helper, the `sky.env` and
//! `sky.background` modules validate, and the env cube faces follow the raa-ibl convention.

use raa_shader::{Composer, LIB_WGSL, PRELUDE_WGSL};
use raa_sky::SKY_WGSL;

/// Minimal fragment entry evaluating the sky along a direction derived from the pixel, with
/// the frame's sun and the environment uniform, as the background contributor does.
const ENTRY: &str = "
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag.xy * raa_frame.viewport.zw * 2.0 - vec2<f32>(1.0);
    let dir = normalize(vec3<f32>(uv.x, uv.y, -1.0));
    let c = raa_sky_radiance(dir, raa_env, raa_frame.sun_dir.xyz, raa_frame.sun_color.rgb, raa_frame.camera_pos.w);
    return vec4<f32>(c, 1.0);
}
";

#[test]
fn sky_wgsl_validates() {
    let wgsl = format!("{PRELUDE_WGSL}\n{LIB_WGSL}\n{SKY_WGSL}\n{ENTRY}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("sky shader refused: {e}");
    }
}

#[test]
fn sky_helpers_are_prefixed() {
    let mut names = Vec::new();
    for line in SKY_WGSL.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("fn ") {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            names.push(name);
        }
    }
    assert!(
        names.iter().any(|n| n == "raa_sky_radiance"),
        "raa_sky_radiance missing: {names:?}"
    );
    for name in &names {
        assert!(
            name == "raa_sky_radiance" || name.starts_with("raa_sky_"),
            "helper {name} is not prefixed raa_sky_"
        );
    }
}

#[path = "../src/shaders.rs"]
mod shaders;

#[test]
fn env_and_background_modules_validate() {
    for (name, wgsl) in [
        ("sky.env", shaders::env_module_wgsl()),
        ("sky.background", shaders::background_module_wgsl()),
    ] {
        assert!(wgsl.contains(SKY_WGSL), "{name} does not embed SKY_WGSL");
        assert!(
            wgsl.starts_with(PRELUDE_WGSL),
            "{name} does not start with the prelude"
        );
        if let Err(e) = Composer::validate(&wgsl) {
            panic!("{name} shader refused: {e}");
        }
    }
}

#[test]
fn env_face_directions_match_ibl_convention() {
    let centres = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    for (face, want) in centres.iter().enumerate() {
        let got = shaders::cube_face_direction(face as u32, 0.0, 0.0);
        for c in 0..3 {
            assert!(
                (got[c] - want[c]).abs() < 1e-6,
                "face {face} centre {got:?}, expected {want:?}"
            );
        }
    }
    // Off-centre orientation of the wgpu cube convention (raa-ibl sh.rs): on +X, u grows
    // toward -Z and v (rows, downwards) toward -Y; on +Y, v grows toward +Z.
    let d = shaders::cube_face_direction(0, 1.0, 1.0);
    let s = 1.0 / 3.0f32.sqrt();
    for (c, want) in [s, -s, -s].iter().enumerate() {
        assert!((d[c] - want).abs() < 1e-6, "+X corner {d:?}");
    }
    let d = shaders::cube_face_direction(2, 0.0, 1.0);
    assert!(d[2] > 0.7 && d[1] > 0.7, "+Y bottom edge {d:?}");
    // Unit length everywhere on every face.
    for face in 0..6 {
        for (u, v) in [(-1.0, -1.0), (0.3, -0.7), (1.0, 0.5)] {
            let d = shaders::cube_face_direction(face, u, v);
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-6,
                "face {face} ({u}, {v}) length {len}"
            );
        }
    }
}

/// The adapted physical-atmosphere helper source (S23); no bindings, functions prefixed
/// `raa_atmosphere_`, and the S10 `PhysicalSkyGpu` record.
const ATMOSPHERE_WGSL: &str = include_str!("../src/wgsl/atmosphere.wgsl");

/// Minimal fragment entry binding the S10 group-2 record and exercising every helper.
const ATMOSPHERE_ENTRY: &str = "
@group(2) @binding(0) var<uniform> raa_sky: PhysicalSkyGpu;

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    let direction = raa_atmosphere_sky_view_direction(vec2<f32>(0.5, 0.5));
    let medium = raa_atmosphere_sample_medium(1.0, raa_sky);
    let transmittance = raa_atmosphere_transmittance_uv(raa_sky.viewer.w, 1.0, raa_sky);
    let params = raa_atmosphere_transmittance_params(transmittance, raa_sky);
    let hit = raa_atmosphere_ray_sphere(
        vec3<f32>(0.0, raa_sky.viewer.w, 0.0), direction, raa_sky.planet.y,
    );
    let phase = raa_atmosphere_rayleigh_phase(0.5) + raa_atmosphere_mie_phase(0.5, raa_sky.mie.w);
    let uv = raa_atmosphere_sky_view_uv(direction);
    return vec4<f32>(
        medium.extinction.x + medium.mie + transmittance.x + params.y + hit + phase + uv.x + uv.y,
        0.0, 0.0, 1.0,
    );
}
";

#[test]
fn atmosphere_wgsl_validates_and_helpers_are_prefixed() {
    let wgsl = format!("{PRELUDE_WGSL}\n{LIB_WGSL}\n{ATMOSPHERE_WGSL}\n{ATMOSPHERE_ENTRY}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("atmosphere shader refused: {e}");
    }
    for name in [
        "sample_medium",
        "rayleigh_phase",
        "mie_phase",
        "ray_sphere",
        "transmittance_uv",
        "transmittance_params",
        "sky_view_uv",
        "sky_view_direction",
    ] {
        assert!(
            ATMOSPHERE_WGSL.contains(&format!("fn raa_atmosphere_{name}(")),
            "raa_atmosphere_{name} missing"
        );
    }
    let mut names = Vec::new();
    for line in ATMOSPHERE_WGSL.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("fn ") {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            names.push(name);
        }
    }
    assert!(!names.is_empty(), "atmosphere helpers are empty");
    for name in names {
        assert!(
            name.starts_with("raa_atmosphere_"),
            "helper {name} is not prefixed raa_atmosphere_"
        );
    }
    // The source constants stay in the helper file, parameterised by the S10 record.
    for constant in [
        "raa_atmosphere_ground_radius_km: f32 = 6360.0",
        "raa_atmosphere_top_radius_km: f32 = 6460.0",
        "vec3<f32>(5.802e-3, 13.558e-3, 33.1e-3)",
        "raa_atmosphere_mie_scale_height_km: f32 = 1.6",
        "vec3<f32>(0.650e-3, 1.881e-3, 0.085e-3)",
    ] {
        assert!(
            ATMOSPHERE_WGSL.contains(constant),
            "source constant missing: {constant}"
        );
    }
}

#[test]
fn legacy_sky_remains_standalone() {
    // The physical helper is a separate source: the legacy constant composition must
    // neither embed the S10 record nor the atmosphere helpers.
    assert!(!SKY_WGSL.contains("PhysicalSkyGpu"));
    assert!(!SKY_WGSL.contains("raa_atmosphere_"));
    let wgsl = format!("{PRELUDE_WGSL}\n{LIB_WGSL}\n{SKY_WGSL}\n{ENTRY}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("legacy sky shader refused: {e}");
    }
}

/// The adapted LUT source (S23): the static transmittance and multiple-scattering tables and
/// the dynamic sky-view table, as three explicit group-2 compute entries.
const ATMOSPHERE_LUTS_WGSL: &str = include_str!("../src/wgsl/atmosphere_luts.wgsl");

#[test]
fn atmosphere_lut_module_is_explicit_group_two_compute() {
    let wgsl = shaders::atmosphere_luts_module_wgsl();
    assert!(
        wgsl.contains(ATMOSPHERE_WGSL),
        "the LUT module omits the atmosphere helpers and record"
    );
    assert!(
        wgsl.contains(ATMOSPHERE_LUTS_WGSL),
        "the LUT module omits its compute source"
    );
    // The lit prelude/library are not composed: the compute pipeline has empty groups 0/1
    // and binds everything explicitly in group 2.
    assert!(
        !wgsl.contains("@group(0)") && !wgsl.contains("@group(1)"),
        "the LUT module must not declare group 0/1 resources"
    );
    assert!(
        !wgsl.contains("raa_frame"),
        "the LUT module must not depend on the frame ABI"
    );
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("atmosphere LUT shader refused: {e}");
    }
    for entry in shaders::ATMOSPHERE_LUT_ENTRIES {
        assert!(
            wgsl.contains(&format!("@compute @workgroup_size(8, 8)\nfn {entry}(")),
            "compute entry {entry} missing"
        );
    }
    assert_eq!(
        shaders::ATMOSPHERE_DIRECTIONAL_ENTRY,
        "raa_atmosphere_build_directional_transmission"
    );
    assert!(
        wgsl.contains(&format!(
            "@compute @workgroup_size(1, 1)\nfn {}()",
            shaders::ATMOSPHERE_DIRECTIONAL_ENTRY
        )),
        "the one-texel directional entry is missing or has the wrong workgroup"
    );
    // Build order: the static tables, then the sky view, then the directional result that
    // the recorder copies into the weather uniform.
    let order = [
        shaders::ATMOSPHERE_LUT_ENTRIES[0],
        shaders::ATMOSPHERE_LUT_ENTRIES[1],
        shaders::ATMOSPHERE_LUT_ENTRIES[2],
        shaders::ATMOSPHERE_DIRECTIONAL_ENTRY,
    ];
    let mut last = 0;
    for entry in order {
        let at = ATMOSPHERE_LUTS_WGSL
            .find(&format!("fn {entry}("))
            .unwrap_or_else(|| panic!("entry {entry} source missing"));
        assert!(at >= last, "entry {entry} is out of build order");
        last = at;
    }
}

#[test]
fn atmosphere_luts_preserve_source_steps_and_authorable_ground() {
    for text in [
        "const STEPS: i32 = 40;",
        "const SQRT_DIRECTIONS: u32 = 8u;",
        "const STEPS: i32 = 20;",
        "const STEPS: i32 = 32;",
        "let t = f * f * t_max;",
        "fn raa_atmosphere_lut_optical_depth_to_top(",
        "fn raa_atmosphere_lut_integrate(",
        "sky.ground_albedo.rgb",
        "raa_atmosphere_min_scale",
    ] {
        assert!(
            ATMOSPHERE_LUTS_WGSL.contains(text),
            "source integration detail missing: {text}"
        );
    }
    // The ground albedo is authored through the S10 lane, never a hidden shader constant.
    assert!(
        !ATMOSPHERE_LUTS_WGSL.contains("GROUND_ALBEDO"),
        "ground albedo must come from PhysicalSkyGpu.ground_albedo"
    );
    let mut names = Vec::new();
    for line in ATMOSPHERE_LUTS_WGSL.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("fn ") {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            names.push(name);
        }
    }
    assert!(!names.is_empty(), "LUT helpers are empty");
    for name in names {
        assert!(
            name.starts_with("raa_atmosphere_"),
            "LUT function {name} is not prefixed raa_atmosphere_"
        );
    }
}

#[test]
fn directional_transmission_pins_the_horizon_source() {
    // Adapted from Solstice `shaders/common/environment.wgsl:98-113`: the horizon dip blocks a
    // light below it and the 0.006 half-width fades the cut; the result is (rgb, 1).
    for text in [
        "const raa_atmosphere_horizon_fade: f32 = 0.006;",
        "fn raa_atmosphere_above_horizon(",
        "let dip = -sqrt(max(1.0 - ratio * ratio, 0.0));",
        "smoothstep(dip - raa_atmosphere_horizon_fade, dip + raa_atmosphere_horizon_fade, mu)",
        "raa_atmosphere_unit_direction(sky.dominant.xyz)",
        "raa_atmosphere_transmittance_uv(radius, mu, sky)",
        "raa_atmosphere_directional_result.value = vec4<f32>(filtered * horizon, 1.0);",
    ] {
        assert!(
            ATMOSPHERE_LUTS_WGSL.contains(text),
            "directional transmission source detail missing: {text}"
        );
    }
    assert!(
        !ATMOSPHERE_LUTS_WGSL.contains("raa_frame"),
        "directional transmission must not read the frame ABI"
    );
}

#[test]
fn atmosphere_luts_reflect_the_s10_private_compute_layout() {
    use wgpu::naga::{
        AddressSpace, ImageClass, ImageDimension, ScalarKind, ShaderStage, StorageAccess,
        StorageFormat, TypeInner, front::wgsl, valid,
    };
    let source = shaders::atmosphere_luts_module_wgsl();
    let module = wgsl::parse_str(&source).expect("atmosphere LUT module parses");
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .expect("atmosphere LUT module validates");

    let mut bindings = Vec::new();
    for (handle, global) in module.global_variables.iter() {
        if let Some(binding) = &global.binding {
            assert_eq!(binding.group, 2, "every LUT resource lives in group 2");
            bindings.push((binding.group, binding.binding, handle));
        }
    }
    bindings.sort_by_key(|(group, binding, _)| (*group, *binding));
    let points: Vec<(u32, u32)> = bindings.iter().map(|(g, b, _)| (*g, *b)).collect();
    assert_eq!(
        points,
        vec![(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)],
        "atmosphere compute layout is uniform, two sampled LUTs, sampler, storage texture \
         and the directional storage result"
    );
    for (_, binding, handle) in &bindings {
        let global = &module.global_variables[*handle];
        let inner = &module.types[global.ty].inner;
        match *binding {
            0 => {
                assert_eq!(global.space, AddressSpace::Uniform);
                let TypeInner::Struct { span, .. } = *inner else {
                    panic!("binding 0 is not the PhysicalSkyGpu struct");
                };
                assert_eq!(span, 256, "PhysicalSkyGpu is the S10 256-byte record");
            }
            1 | 2 => assert!(
                matches!(
                    inner,
                    TypeInner::Image {
                        dim: ImageDimension::D2,
                        arrayed: false,
                        class: ImageClass::Sampled {
                            kind: ScalarKind::Float,
                            multi: false,
                        },
                    }
                ),
                "binding {binding} is not a sampled 2D f32 LUT"
            ),
            3 => assert!(
                matches!(inner, TypeInner::Sampler { comparison: false }),
                "binding 3 is not a filtering sampler"
            ),
            4 => assert!(
                matches!(
                    inner,
                    TypeInner::Image {
                        dim: ImageDimension::D2,
                        arrayed: false,
                        class: ImageClass::Storage {
                            format: StorageFormat::Rgba16Float,
                            access: StorageAccess::STORE,
                        },
                    }
                ),
                "binding 4 is not an Rgba16Float storage output"
            ),
            5 => {
                let AddressSpace::Storage { access } = global.space else {
                    panic!("binding 5 is not a storage buffer");
                };
                assert!(
                    access.contains(StorageAccess::LOAD | StorageAccess::STORE),
                    "binding 5 is not read_write"
                );
                let TypeInner::Struct { span, .. } = *inner else {
                    panic!("binding 5 is not the directional result struct");
                };
                assert_eq!(span, 16, "the directional result is one vec4<f32>");
            }
            other => panic!("unexpected atmosphere LUT binding {other}"),
        }
    }

    assert_eq!(
        module.entry_points.len(),
        shaders::ATMOSPHERE_LUT_ENTRIES.len() + 1,
        "the LUT module has exactly three table entries and one directional entry"
    );
    let mut used = Vec::new();
    for (index, entry) in module.entry_points.iter().enumerate() {
        assert_eq!(entry.stage, ShaderStage::Compute);
        let expected_workgroup = if entry.name == shaders::ATMOSPHERE_DIRECTIONAL_ENTRY {
            [1, 1, 1]
        } else {
            [8, 8, 1]
        };
        assert_eq!(entry.workgroup_size, expected_workgroup);
        let point = info.get_entry_point(index);
        let mut resources: Vec<u32> = bindings
            .iter()
            .filter(|(_, _, handle)| !point[*handle].is_empty())
            .map(|(_, binding, _)| *binding)
            .collect();
        resources.sort_unstable();
        used.push((entry.name.as_str(), resources));
    }
    used.sort_by_key(|(name, _)| *name);
    assert_eq!(
        used,
        vec![
            (
                "raa_atmosphere_build_directional_transmission",
                vec![0, 1, 3, 5]
            ),
            ("raa_atmosphere_build_multi_scatter", vec![0, 1, 3, 4]),
            ("raa_atmosphere_build_sky_view", vec![0, 1, 2, 3, 4]),
            ("raa_atmosphere_build_transmittance", vec![0, 4]),
        ],
        "reflected per-entry group-2 resource use changed"
    );
}

#[test]
fn milky_way_bake_reflects_one_explicit_group_two_output() {
    use wgpu::naga::{
        ImageClass, ImageDimension, ShaderStage, StorageAccess, StorageFormat, TypeInner,
        front::wgsl, valid,
    };
    let source = shaders::milky_way_bake_module_wgsl();
    Composer::validate(&source).expect("Milky Way bake validates");
    let module = wgsl::parse_str(&source).expect("Milky Way bake parses");
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .expect("Milky Way bake validates");
    let globals: Vec<_> = module.global_variables.iter().collect();
    assert_eq!(
        globals.len(),
        1,
        "no uniforms, sampled textures or extra groups"
    );
    let (handle, output) = globals[0];
    let binding = output.binding.as_ref().expect("explicit output binding");
    assert_eq!((binding.group, binding.binding), (2, 0));
    assert!(matches!(
        module.types[output.ty].inner,
        TypeInner::Image {
            dim: ImageDimension::D2,
            arrayed: false,
            class: ImageClass::Storage {
                format: StorageFormat::Rgba8Unorm,
                access: StorageAccess::STORE
            }
        }
    ));
    assert_eq!(module.entry_points.len(), 1);
    let entry = &module.entry_points[0];
    assert_eq!(entry.name, "raa_milky_way_bake");
    assert_eq!(entry.stage, ShaderStage::Compute);
    assert_eq!(entry.workgroup_size, [8, 8, 1]);
    let usage = info.get_entry_point(0)[handle];
    assert!(usage.contains(valid::GlobalUse::WRITE));
    assert!(!usage.contains(valid::GlobalUse::READ));
    for (_, function) in module.functions.iter() {
        assert!(
            function
                .name
                .as_ref()
                .expect("named helper")
                .starts_with("raa_milky_way_")
        );
    }
    assert!(
        source.contains("if any(id.xy >= size)"),
        "guard rounded-up dispatches"
    );
    assert!(source.contains("const raa_milky_way_range: f32 = 4.0;"));
    assert!(source.contains("sqrt(clamp(galaxy.rgb / raa_milky_way_range"));
}

#[test]
fn milky_way_helpers_compose_with_shared_and_atmosphere_sources() {
    let helpers = include_str!("../src/wgsl/milky_way.wgsl");
    let entry = "
@fragment fn fs_milky_way() -> @location(0) vec4<f32> {
    let direction = raa_milky_way_celestial_direction(vec2f(0.5));
    let uv = raa_milky_way_celestial_uv(direction);
    return raa_milky_way_glow(raa_milky_way_celestial_direction(uv));
}";
    Composer::validate(&format!(
        "{PRELUDE_WGSL}\n{LIB_WGSL}\n{ATMOSPHERE_WGSL}\n{SKY_WGSL}\n{helpers}\n{entry}"
    ))
    .expect("prefixed celestial noise and map coordinates compose without collisions");
    for source in [helpers, include_str!("../src/wgsl/milky_way_bake.wgsl")] {
        assert!(source.contains("Copyright (c) 2026 SamG-Coder"));
        assert!(source.contains("Permission is hereby granted, free of charge"));
        assert!(source.contains("THE SOFTWARE IS PROVIDED \"AS IS\""));
        assert!(source.contains("OUT OF OR IN CONNECTION WITH THE SOFTWARE"));
    }
}

#[test]
fn celestial_body_contracts_are_callable_with_s10_resources() {
    use wgpu::naga::{front::wgsl, valid};
    // Call every handoff helper, including the surface helpers, from a fragment entry.
    let entry = "
@fragment fn fs_bodies(@builtin(position) frag: vec4f) -> @location(0) vec4f {
    let direction = normalize(vec3f(frag.xy * 0.001, 1.0));
    let pixel_angle = raa_physical_sky.discs.w;
    let moon = moon_body(direction, pixel_angle);
    let sun = sun_body(direction, pixel_angle);
    let coords = selenographic(direction);
    let rays = crater_rays(direction, normalize(vec3f(0.2, 0.3, 1.0)), 0.5, 31u);
    let surface = moon_surface(direction, 1.0);
    let coverage = disc_coverage(coords.x, 0.0075, pixel_angle);
    return moon + vec4f(sun + air_transmittance(direction) * surface * rays, coverage);
}";
    let source = format!("{}\n{entry}", shaders::celestial_module_wgsl());
    Composer::validate(&source).expect("body helpers compose with the shared prelude and noise");
    let module = wgsl::parse_str(&source).expect("body module parses");
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .expect("body module validates");
    let used: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(handle, global)| {
            if info.get_entry_point(0)[handle].is_empty() {
                return None;
            }
            global.binding.as_ref().map(|b| (b.group, b.binding))
        })
        .collect();
    assert_eq!(
        used,
        [(2, 0), (2, 1), (2, 6)],
        "only private sky resources are read"
    );
}

#[test]
fn celestial_bodies_retain_surface_and_display_contracts() {
    let source = include_str!("../src/wgsl/celestial.wgsl");
    assert!(!source.contains("@group"), "the composer owns bindings");
    assert!(!source.contains("frame."), "no Solstice combined frame ABI");
    for retained in [
        "const raa_celestial_moon_radius: f32 = 0.0075;",
        "array<vec4f, 17>",
        "Oceanus Procellarum",
        "Sinus Roris",
        "vec3f(-0.1433, -0.6859, 0.7134)", // Tycho
        "vec3f(-0.3385, 0.1671, 0.9260)",  // Copernicus
        "vec3f(-0.6096, 0.1412, 0.7800)",  // Kepler
        "vec3f(-0.6748, 0.4024, 0.6186)",  // Aristarchus
        "raa_physical_sky.discs.x <= 0.0",
        "raa_physical_sky.discs.y",
        "vec3f(0.55, 0.62, 0.72)",
        "let earthshine = 0.035 * earth;",
        "pow(max(cos_phase, 0.0), 60.0)",
        "return vec4f(radiance * cover, cover);",
        "smoothstep(dip - 0.006, dip + 0.006, direction.y)",
        "Copyright (c) 2026 SamG-Coder",
        "Permission is hereby granted, free of charge",
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE",
    ] {
        assert!(
            source.contains(retained),
            "missing source contract: {retained}"
        );
    }
}

#[test]
fn celestial_evaluator_composes_with_canonical_resources() {
    use wgpu::naga::{AddressSpace, TypeInner, front::wgsl, valid};
    let source = format!(
        "{}\n@fragment fn fs_celestial(@builtin(position) p: vec4f) -> @location(0) vec4f {{
            return vec4f(raa_celestial_radiance(vec3f(p.xy, 1.0)), 1.0);
        }}",
        shaders::celestial_module_wgsl()
    );
    Composer::validate(&source).expect("common celestial evaluator validates");
    let module = wgsl::parse_str(&source).unwrap();
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .unwrap();
    let mut private = Vec::new();
    let mut used = Vec::new();
    let mut budget = [0; 4]; // sampled textures, samplers, uniforms, storage buffers
    for (handle, global) in module.global_variables.iter() {
        let Some(binding) = &global.binding else {
            continue;
        };
        assert!(binding.group < 4);
        if binding.group == 2 {
            private.push(binding.binding);
        }
        if !info.get_entry_point(0)[handle].is_empty() {
            used.push((binding.group, binding.binding));
        }
        // S10 previous instances (group 0 binding 8) are VERTEX-only. Include all
        // other declared shared resources in the conservative fragment budget.
        if (binding.group, binding.binding) == (0, 8) {
            assert!(info.get_entry_point(0)[handle].is_empty());
            continue;
        }
        match module.types[global.ty].inner {
            TypeInner::Image { .. } => budget[0] += 1,
            TypeInner::Sampler { .. } => budget[1] += 1,
            _ if global.space == AddressSpace::Uniform => budget[2] += 1,
            _ if matches!(global.space, AddressSpace::Storage { .. }) => budget[3] += 1,
            _ => {}
        }
    }
    assert_eq!(private, (0..8).collect::<Vec<_>>());
    assert_eq!(used, [(2, 0), (2, 1), (2, 5), (2, 6), (2, 7)]);
    for (count, limit) in budget.into_iter().zip([15, 5, 7, 5]) {
        assert!(
            count <= limit,
            "composed resource count {count} exceeds {limit}"
        );
    }
}

#[test]
fn celestial_decode_occlusion_and_finite_input_guards_are_retained() {
    let source = include_str!("../src/wgsl/celestial.wgsl");
    for contract in [
        "baked.rgb * baked.rgb * raa_milky_way_range, baked.a",
        "dot(direction, cross(pole, equinox))",
        "raa_physical_sky.pole.w",
        "raa_physical_sky.planet.z",
        "vec3f(0.610, 0.550, 0.465)",
        "12.0, 0.28, -1.0, 2.5, 1u",
        "28.0, 0.85, 2.5, 4.5, 2u",
        "60.0, min(0.6 * (1.0 + 0.7 * galaxy.a), 1.0)",
        "let footprint = max(pixel_angle, 1.0e-8);",
        "let spread = footprint *",
        "if scale <= 0.0",
        "normalize(direction / scale)",
        "sun_body(ray, pixel_angle) + moon.rgb",
        "raa_celestial_night(ray, pixel_angle) * (1.0 - moon.a)",
    ] {
        assert!(
            source.contains(contract),
            "missing celestial contract: {contract}"
        );
    }
    assert!(!source.contains("moon.rgb *"));
    assert!(!source.contains("raa_directional_radiance"));
    // Exercise the WGSL call boundary for finite extremes, zero and both celestial poles.
    // These are validation fixtures, not a CPU reimplementation or GPU output claim.
    let calls = [
        "vec3f(0.0)",
        "vec3f(0.0, 1.0, 0.0)",
        "vec3f(0.0, -1.0, 0.0)",
        "vec3f(3.0e38, -3.0e38, 3.0e38)",
        "vec3f(1.0e-30, 0.0, 0.0)",
    ];
    let body = calls
        .map(|ray| format!("c += raa_celestial_radiance({ray});"))
        .join("\n");
    let source = format!(
        "{}\n@fragment fn fs_finite() -> @location(0) vec4f {{
            var c = vec3f(0.0); {body} return vec4f(c, 1.0);
        }}",
        shaders::celestial_module_wgsl()
    );
    Composer::validate(&source).expect("finite-input celestial call fixtures validate");
}

/// The physical consumers keep the legacy entry points and the S11 `raa_sky_radiance` seam,
/// but attach the completed atmosphere, celestial and cloud resources through the unchanged
/// S10 group-2 layout. Allocation is not readiness: the Rust prepare supplies validity flags
/// from `CloudViews` and the bind group only after the view's sky-view LUT and the Milky Way
/// bake were carried by actual submissions.
#[test]
fn physical_consumers_attach_the_exact_s10_group_two_layout() {
    use wgpu::naga::{
        AddressSpace, ImageClass, ImageDimension, ScalarKind, ShaderStage, TypeInner, front::wgsl,
        valid,
    };
    for (name, source) in [
        ("sky.env", shaders::physical_env_module_wgsl()),
        ("sky.background", shaders::physical_background_module_wgsl()),
    ] {
        assert!(source.starts_with(PRELUDE_WGSL), "{name}: prelude first");
        assert!(
            source.contains("fn raa_sky_radiance("),
            "{name}: the common seam is replaced, not bypassed"
        );
        assert!(
            source.contains("raa_celestial_radiance"),
            "{name}: the celestial evaluator is folded in"
        );
        if let Err(e) = Composer::validate(&source) {
            panic!("{name} physical shader refused: {e}");
        }
        let module = wgsl::parse_str(&source).expect("physical module parses");
        let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
            .validate(&module)
            .expect("physical module validates");

        let mut bindings = Vec::new();
        for (handle, global) in module.global_variables.iter() {
            let Some(binding) = &global.binding else {
                continue;
            };
            if binding.group == 2 {
                bindings.push((binding.binding, handle));
            }
        }
        bindings.sort_by_key(|(binding, _)| *binding);
        let points: Vec<u32> = bindings.iter().map(|(binding, _)| *binding).collect();
        assert_eq!(points, (0..8).collect::<Vec<_>>(), "{name}: group-2 layout");
        for (binding, handle) in &bindings {
            let global = &module.global_variables[*handle];
            let inner = &module.types[global.ty].inner;
            match *binding {
                0 => {
                    assert_eq!(global.space, AddressSpace::Uniform, "{name}: binding 0");
                    let TypeInner::Struct { span, .. } = *inner else {
                        panic!("{name}: binding 0 is not PhysicalSkyGpu");
                    };
                    assert_eq!(span, 256, "{name}: S10 uniform size");
                }
                1..=5 => assert!(
                    matches!(
                        inner,
                        TypeInner::Image {
                            dim: ImageDimension::D2,
                            arrayed: false,
                            class: ImageClass::Sampled {
                                kind: ScalarKind::Float,
                                multi: false,
                            },
                        }
                    ),
                    "{name}: binding {binding} is not a filterable D2 texture"
                ),
                6 | 7 => assert!(
                    matches!(inner, TypeInner::Sampler { comparison: false }),
                    "{name}: binding {binding} is not a filtering sampler"
                ),
                other => panic!("{name}: unexpected group-2 binding {other}"),
            }
        }

        let fragment = module
            .entry_points
            .iter()
            .position(|entry| entry.name == "fs_main" && entry.stage == ShaderStage::Fragment)
            .expect("fs_main");
        let usage = info.get_entry_point(fragment);
        let mut used: Vec<u32> = bindings
            .iter()
            .filter(|(_, handle)| !usage[*handle].is_empty())
            .map(|(binding, _)| *binding)
            .collect();
        used.sort_unstable();
        // The env reads binding 4 through the cloud dome (`raa_physical_sky_cloud_dome`).
        let expected = vec![0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(used, expected, "{name}: group-2 use");
    }
}

#[test]
fn physical_composition_keeps_clear_sky_and_clouds_separate() {
    let source = shaders::physical_env_module_wgsl();
    for contract in [
        // Solstice environment.wgsl:93-113 at 7e465b5: clear * cloud.a + cloud.rgb.
        "clear * clamp(cloud.a, 0.0, 1.0) + max(cloud.rgb",
        // The raw accepted factors, never a previously attenuated or replayed value.
        "raa_physical_sky.daylight.w",
        "raa_physical_sky.equinox.w",
        "raa_physical_sky.discs.w",
        // Exact CloudViews validity lanes and the reserved negative screen alpha.
        "raa_physical_sky.flags.y",
        "raa_physical_sky.flags.z",
        "cloud.a >= 0.0",
        // Artistic controls are ratios written by the prepare, not a second preset tint.
        "raa_physical_sky_tint(direction, env)",
        "raa_atmosphere_sky_view_uv(direction)",
    ] {
        assert!(
            source.contains(contract),
            "missing physical contract: {contract}"
        );
    }
    assert_eq!(
        source.matches("raa_directional_radiance(").count(),
        1,
        "the prelude defines the material boundary; the sky evaluator must not call it a second time"
    );
    assert!(
        !source.contains("moon.rgb *"),
        "the moon radiance already carries coverage"
    );
    let background = shaders::physical_background_module_wgsl();
    assert!(
        background.contains("in.clip.xy * raa_frame.viewport.zw"),
        "screen clouds use viewport UV"
    );
    assert!(background.contains("raa_physical_sky_radiance_screen"));
}

#[test]
fn legacy_builders_do_not_embed_the_physical_module() {
    for (name, wgsl) in [
        ("sky.env", shaders::env_module_wgsl()),
        ("sky.background", shaders::background_module_wgsl()),
    ] {
        assert!(wgsl.contains(SKY_WGSL), "{name}: legacy sky source");
        assert!(!wgsl.contains("PhysicalSkyGpu"), "{name}: physical uniform");
        assert!(
            !wgsl.contains("raa_physical_sky"),
            "{name}: physical helpers"
        );
        assert!(
            !wgsl.contains("raa_celestial_radiance"),
            "{name}: celestial"
        );
    }
}
