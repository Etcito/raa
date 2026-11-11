//! Reflection of the cloud noise WGSL module and its S10 private compute layout.

use raa_shader::Composer;
use wgpu::naga::{
    ImageClass, ImageDimension, ShaderStage, StorageAccess, StorageFormat, TypeInner,
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};

use super::super::shaders::{NOISE_ENTRIES, noise_module_wgsl};

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
