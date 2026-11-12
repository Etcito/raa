//! Reflection of the cloud noise WGSL module and its S10 private compute layout.

use raa_shader::Composer;
use wgpu::naga::{
    AddressSpace, ImageClass, ImageDimension, Scalar, ShaderStage, StorageAccess, StorageFormat,
    TypeInner, VectorSize,
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};

use super::super::resources::CloudGpu;
use super::super::shaders::{
    COVERAGE_ENTRIES, NOISE_ENTRIES, cloud_volume_wgsl, coverage_module_wgsl, noise_module_wgsl,
    dome_module_wgsl, panorama_module_wgsl, screen_module_wgsl, shadow_mean_module_wgsl, shadow_module_wgsl,
};

#[test]
fn noise_module_is_valid_wgsl() {
    let source = noise_module_wgsl();
    let module = wgsl::parse_str(&source).expect("noise module parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("noise module validates");
    Composer::validate(&source).expect("noise module composes");
}

#[test]
fn noise_module_declares_three_compute_functions_and_group2_outputs() {
    let module = wgsl::parse_str(&noise_module_wgsl()).expect("noise module parses");
    let names: Vec<&str> = module
        .entry_points
        .iter()
        .map(|entry| {
            assert_eq!(entry.stage, ShaderStage::Compute, "{}", entry.name);
            entry.name.as_str()
        })
        .collect();
    assert_eq!(names, NOISE_ENTRIES);

    let mut bindings = Vec::new();
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = &global.binding else {
            continue;
        };
        assert!(
            binding.group >= 2,
            "group {} must stay empty",
            binding.group
        );
        bindings.push((
            binding.group,
            binding.binding,
            &module.types[global.ty].inner,
        ));
    }
    bindings.sort_by_key(|(group, binding, _)| (*group, *binding));
    let points: Vec<(u32, u32)> = bindings.iter().map(|(g, b, _)| (*g, *b)).collect();
    assert_eq!(points, [(2, 0), (2, 1)]);

    let storage = |inner: &TypeInner, dim| {
        matches!(
            inner,
            TypeInner::Image {
                dim: actual,
                arrayed: false,
                class: ImageClass::Storage {
                    format: StorageFormat::Rgba8Unorm,
                    access: StorageAccess::STORE,
                },
            } if *actual == dim
        )
    };
    assert!(
        storage(bindings[0].2, ImageDimension::D3),
        "binding (2,0) is not the 3D rgba8unorm storage write"
    );
    assert!(
        storage(bindings[1].2, ImageDimension::D2),
        "binding (2,1) is not the 2D rgba8unorm storage write"
    );
}

#[test]
fn noise_helpers_keep_the_cloud_prefix() {
    let module = wgsl::parse_str(&noise_module_wgsl()).expect("noise module parses");
    let mut helpers = 0;
    for (_, function) in module.functions.iter() {
        let name = function.name.as_deref().expect("named helper");
        assert!(
            name.starts_with("raa_cloud_noise_"),
            "helper {name} is not prefixed raa_cloud_noise_"
        );
        helpers += 1;
    }
    assert!(helpers > 0, "the module has helpers to check");
    for entry in &module.entry_points {
        assert!(entry.name.starts_with("raa_cloud_noise_"));
    }
}

#[test]
fn coverage_module_is_valid_wgsl() {
    let source = coverage_module_wgsl();
    let module = wgsl::parse_str(&source).expect("coverage module parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("coverage module validates");
    Composer::validate(&source).expect("coverage module composes");
}

#[test]
fn coverage_module_declares_two_compute_functions_and_group2_bindings() {
    let module = wgsl::parse_str(&coverage_module_wgsl()).expect("coverage module parses");
    let names: Vec<&str> = module
        .entry_points
        .iter()
        .map(|entry| {
            assert_eq!(entry.stage, ShaderStage::Compute, "{}", entry.name);
            entry.name.as_str()
        })
        .collect();
    assert_eq!(names, COVERAGE_ENTRIES);

    let mut bindings = Vec::new();
    for (_, global) in module.global_variables.iter() {
        if let Some(binding) = &global.binding {
            bindings.push((binding.group, binding.binding, global));
        }
    }
    bindings.sort_by_key(|(group, binding, _)| (*group, *binding));
    let points: Vec<(u32, u32)> = bindings.iter().map(|(g, b, _)| (*g, *b)).collect();
    assert_eq!(points, [(2, 0), (2, 1), (2, 2)]);
    assert!(matches!(
        module.types[bindings[0].2.ty].inner,
        TypeInner::Image {
            dim: ImageDimension::D2,
            arrayed: false,
            class: ImageClass::Sampled { multi: false, .. },
        }
    ));
    for (_, binding, global) in &bindings[1..] {
        assert!(
            matches!(global.space, AddressSpace::Storage { access } if access.contains(StorageAccess::STORE)),
            "binding (2,{binding}) is not a read_write storage buffer"
        );
        assert!(matches!(
            module.types[global.ty].inner,
            TypeInner::Array { .. }
        ));
    }
}

#[test]
fn cloud_volume_text_with_a_probe_function_is_valid_wgsl() {
    let mut source = cloud_volume_wgsl();
    source.push_str(
        "\n@compute @workgroup_size(1)\nfn raa_cloud_probe() {\n    \
         let d = raa_cloud_density(vec2<f32>(0.0, 0.0), 2000.0, true);\n    \
         if d > 1.0 {\n        return;\n    }\n}\n",
    );
    let module = wgsl::parse_str(&source).expect("cloud volume module parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("cloud volume module validates");
    Composer::validate(&source).expect("cloud volume module composes");
}

#[test]
fn cloud_gpu_wgsl_struct_is_192_bytes() {
    let module = wgsl::parse_str(&cloud_volume_wgsl()).expect("cloud volume module parses");
    let (_, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("CloudGpu"))
        .expect("the module declares CloudGpu");
    let TypeInner::Struct { members, span } = &ty.inner else {
        panic!("CloudGpu is not a struct");
    };
    assert_eq!(*span, 192, "the WGSL struct is 192 bytes");
    assert_eq!(std::mem::size_of::<CloudGpu>(), 192);

    let rust_offsets = [
        ("wind_offset", std::mem::offset_of!(CloudGpu, wind_offset)),
        ("evolution", std::mem::offset_of!(CloudGpu, evolution)),
        ("coverage", std::mem::offset_of!(CloudGpu, coverage)),
        ("cloud_type", std::mem::offset_of!(CloudGpu, cloud_type)),
        ("base", std::mem::offset_of!(CloudGpu, base)),
        ("top", std::mem::offset_of!(CloudGpu, top)),
        ("density", std::mem::offset_of!(CloudGpu, density)),
        ("cirrus", std::mem::offset_of!(CloudGpu, cirrus)),
        ("rain", std::mem::offset_of!(CloudGpu, rain)),
        (
            "shadow_extent",
            std::mem::offset_of!(CloudGpu, shadow_extent),
        ),
        ("enabled", std::mem::offset_of!(CloudGpu, enabled)),
        (
            "shadow_origin",
            std::mem::offset_of!(CloudGpu, shadow_origin),
        ),
        (
            "wind_direction",
            std::mem::offset_of!(CloudGpu, wind_direction),
        ),
        ("right", std::mem::offset_of!(CloudGpu, right)),
        ("up", std::mem::offset_of!(CloudGpu, up)),
        ("forward", std::mem::offset_of!(CloudGpu, forward)),
        ("screen", std::mem::offset_of!(CloudGpu, screen)),
        (
            "previous_view_proj",
            std::mem::offset_of!(CloudGpu, previous_view_proj),
        ),
    ];
    assert_eq!(members.len(), rust_offsets.len());
    for (member, (name, offset)) in members.iter().zip(rust_offsets) {
        assert_eq!(member.name.as_deref(), Some(name));
        assert_eq!(member.offset, offset as u32, "member {name}");
    }
}

#[test]
fn shadow_modules_are_valid_wgsl_with_group2_only() {
    for (source, entry) in [
        (shadow_module_wgsl(), "raa_cloud_shadow_main"),
        (shadow_mean_module_wgsl(), "raa_cloud_shadow_mean_main"),
    ] {
        let module = wgsl::parse_str(&source).expect("shadow module parses");
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .expect("shadow module validates");
        Composer::validate(&source).expect("shadow module composes");
        let names: Vec<&str> = module
            .entry_points
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(names, [entry]);
        assert_eq!(module.entry_points[0].stage, ShaderStage::Compute);
        for (_, global) in module.global_variables.iter() {
            if let Some(binding) = &global.binding {
                assert_eq!(binding.group, 2, "{entry}: groups 0 and 1 stay empty");
            }
        }
    }
}

/// Counts of (sampled textures, samplers, uniforms, storage buffers, storage textures).
fn resource_counts(source: &str) -> [u32; 5] {
    let module = wgsl::parse_str(source).expect("module parses");
    let mut counts = [0; 5];
    for (_, global) in module.global_variables.iter() {
        if global.binding.is_none() {
            continue;
        }
        let slot = match (&module.types[global.ty].inner, global.space) {
            (
                TypeInner::Image {
                    class: ImageClass::Storage { .. },
                    ..
                },
                _,
            ) => 4,
            (TypeInner::Image { .. }, _) => 0,
            (TypeInner::Sampler { .. }, _) => 1,
            (_, AddressSpace::Uniform) => 2,
            (_, AddressSpace::Storage { .. }) => 3,
            (inner, space) => panic!("unexpected binding {inner:?} in {space:?}"),
        };
        counts[slot] += 1;
    }
    counts
}

#[test]
fn shadow_module_fits_the_s10_compute_budget() {
    let shadow = resource_counts(&shadow_module_wgsl());
    for (count, max) in shadow.into_iter().zip([5, 2, 2, 1, 1]) {
        assert!(
            count <= max,
            "shadow module {shadow:?} exceeds [5, 2, 2, 1, 1]"
        );
    }
    assert_eq!(shadow[4], 1, "one storage texture output");
    assert_eq!(resource_counts(&shadow_mean_module_wgsl()), [1, 0, 0, 0, 1]);
}

#[test]
fn cloud_march_text_with_a_probe_function_is_valid_wgsl() {
    let mut source = cloud_volume_wgsl();
    source.push_str(
        "\n@group(2) @binding(10) var raa_cloud_probe_output: texture_storage_2d<rgba16float, write>;\n\
         @compute @workgroup_size(1)\nfn raa_cloud_probe() {\n    \
         let sample = raa_cloud_along(vec3<f32>(0.0, 1.0, 0.0), 32, 4, 0.5);\n    \
         textureStore(\n        raa_cloud_probe_output,\n        vec2<i32>(0, 0),\n        \
         vec4<f32>(sample.radiance, sample.transmittance),\n    );\n}\n",
    );
    let module = wgsl::parse_str(&source).expect("cloud march module parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("cloud march module validates");
    Composer::validate(&source).expect("cloud march module composes");
}

#[test]
fn cloud_sample_struct_has_radiance_transmittance_distance() {
    let module = wgsl::parse_str(&cloud_volume_wgsl()).expect("cloud volume module parses");
    let (_, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("RaaCloudSample"))
        .expect("the module declares RaaCloudSample");
    let TypeInner::Struct { members, .. } = &ty.inner else {
        panic!("RaaCloudSample is not a struct");
    };
    let names: Vec<&str> = members
        .iter()
        .map(|member| member.name.as_deref().expect("named member"))
        .collect();
    assert_eq!(names, ["radiance", "transmittance", "distance"]);
    assert!(matches!(
        module.types[members[0].ty].inner,
        TypeInner::Vector {
            size: VectorSize::Tri,
            scalar: Scalar::F32,
        }
    ));
    for member in &members[1..] {
        assert!(matches!(
            module.types[member.ty].inner,
            TypeInner::Scalar(Scalar::F32)
        ));
    }
}

#[test]
fn cloud_volume_has_no_aerial_perspective_or_fog_helper() {
    let source = include_str!("../wgsl/cloud_volume.wgsl");
    let code: String = source
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase();
    let words: Vec<&str> = code
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    for forbidden in ["aerial", "fog"] {
        assert!(
            !words.contains(&forbidden),
            "cloud_volume.wgsl still names {forbidden}"
        );
    }
}

#[test]
fn cloud_volume_names_keep_the_cloud_prefix() {
    let module = wgsl::parse_str(&cloud_volume_wgsl()).expect("cloud volume module parses");
    let mut cloud_functions = 0;
    for (_, function) in module.functions.iter() {
        let name = function.name.as_deref().expect("named helper");
        if name.starts_with("raa_atmosphere_") {
            continue;
        }
        assert!(
            name.starts_with("raa_cloud_"),
            "helper {name} is not prefixed raa_cloud_"
        );
        cloud_functions += 1;
    }
    assert!(
        cloud_functions > 0,
        "the file defines cloud helpers to check"
    );
}

#[test]
fn panorama_module_is_valid_wgsl_and_fits_the_s10_budget() {
    let source = panorama_module_wgsl();
    let module = wgsl::parse_str(&source).expect("panorama module parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("panorama module validates");
    assert!(Composer::validate(&source).is_ok());
    assert_eq!(resource_counts(&source), [5, 2, 2, 1, 1]);
    for (_, global) in module.global_variables.iter() {
        if let Some(binding) = &global.binding {
            assert_eq!(binding.group, 2, "groups 0 and 1 stay empty");
        }
    }
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].name, "raa_cloud_panorama_main");
    assert_eq!(module.entry_points[0].stage, ShaderStage::Compute);
}

fn assert_valid_screen_module(multisampled_depth: bool) {
    let source = screen_module_wgsl(multisampled_depth);
    let module = wgsl::parse_str(&source).expect("screen module parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("screen module validates");
    Composer::validate(&source).expect("screen module composes");
    let (_, depth) = module
        .global_variables
        .iter()
        .find(|(_, global)| global.name.as_deref() == Some("raa_cloud_depth"))
        .expect("the module declares raa_cloud_depth");
    assert_eq!(
        depth.binding.as_ref().map(|b| (b.group, b.binding)),
        Some((2, 12))
    );
    assert!(matches!(
        module.types[depth.ty].inner,
        TypeInner::Image {
            dim: ImageDimension::D2,
            arrayed: false,
            class: ImageClass::Depth { multi },
        } if multi == multisampled_depth
    ));
}

#[test]
fn screen_module_is_valid_wgsl_for_one_sample_depth() {
    assert_valid_screen_module(false);
}

#[test]
fn screen_module_is_valid_wgsl_for_multisampled_depth() {
    assert_valid_screen_module(true);
}

#[test]
fn screen_module_fits_the_s10_budget() {
    for multisampled_depth in [false, true] {
        let source = screen_module_wgsl(multisampled_depth);
        assert_eq!(resource_counts(&source), [7, 2, 2, 1, 1]);
        let module = wgsl::parse_str(&source).expect("screen module parses");
        for (_, global) in module.global_variables.iter() {
            if let Some(binding) = &global.binding {
                assert_eq!(binding.group, 2, "groups 0 and 1 stay empty");
            }
        }
        assert_eq!(module.entry_points.len(), 1);
        assert_eq!(module.entry_points[0].name, "raa_cloud_screen_main");
        assert_eq!(module.entry_points[0].stage, ShaderStage::Compute);
    }
}

#[test]
fn dome_module_is_valid_wgsl_and_fits_the_s10_budget() {
    let source = dome_module_wgsl();
    let module = wgsl::parse_str(&source).expect("dome module parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("dome module validates");
    Composer::validate(&source).expect("dome module composes");
    // The panorama bindings plus the dome history texture.
    assert_eq!(resource_counts(&source), [6, 2, 2, 1, 1]);
    for (_, global) in module.global_variables.iter() {
        if let Some(binding) = &global.binding {
            assert_eq!(binding.group, 2, "groups 0 and 1 stay empty");
        }
    }
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].name, "raa_cloud_dome_main");
    assert_eq!(module.entry_points[0].stage, ShaderStage::Compute);
}
