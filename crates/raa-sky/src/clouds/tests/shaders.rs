//! Reflection of the cloud noise WGSL module and its S10 private compute layout.

use raa_shader::Composer;
use wgpu::naga::{
    AddressSpace, ImageClass, ImageDimension, ShaderStage, StorageAccess, StorageFormat, TypeInner,
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};

use super::super::resources::CloudGpu;
use super::super::shaders::{
    COVERAGE_ENTRIES, NOISE_ENTRIES, cloud_volume_wgsl, coverage_module_wgsl, noise_module_wgsl,
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
