//! Contract `gi-virtual-kernel`: the driver B module validates with its one entry point, its
//! group 2 holds eight storage buffers and one uniform, the material twin is 256 bytes and the
//! level view selects the GI members. Follows tests/variants.rs. Consumes the Handoff of
//! `voxelize_virtual_module_wgsl` (crates/raa-gi/src/parts/virtual_driver.rs:42).

#![cfg(not(target_arch = "wasm32"))]

use raa_gi::clipmap::level_view;
use raa_gi::parts::virtual_driver::{virtual_group2_entries, voxelize_virtual_module_wgsl};
use raa_shader::Composer;
use wgpu::naga::{self, valid};
use wgpu::{BindingType, BufferBindingType, ShaderStages};

fn analyse(wgsl: &str) -> (naga::Module, valid::ModuleInfo) {
    let module = naga::front::wgsl::parse_str(wgsl).expect("the module parses");
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .expect("the module validates");
    (module, info)
}

#[test]
fn virtual_module_validates_with_one_entry() {
    let wgsl = voxelize_virtual_module_wgsl();
    assert_eq!(Composer::validate(&wgsl), Ok(()));
    let (module, _) = analyse(&wgsl);
    let names: Vec<&str> = module
        .entry_points
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(names, ["raa_gi_voxelize_virtual"]);
    assert_eq!(module.entry_points[0].workgroup_size, [128, 1, 1]);
}

#[test]
fn virtual_group_holds_eight_storage_buffers_and_one_uniform() {
    let (module, info) = analyse(&voxelize_virtual_module_wgsl());
    let usage = info.get_entry_point(0);
    let mut bound: Vec<(u32, u32, naga::AddressSpace)> = module
        .global_variables
        .iter()
        .filter(|(handle, _)| !usage[*handle].is_empty())
        .filter_map(|(_, g)| g.binding.map(|b| (b.group, b.binding, g.space)))
        .collect();
    bound.sort_by_key(|b| (b.0, b.1));
    let slots: Vec<(u32, u32)> = bound.iter().map(|b| (b.0, b.1)).collect();
    assert_eq!(slots, (1..=9).map(|b| (2, b)).collect::<Vec<_>>());
    for (_, binding, space) in &bound {
        let expected = match binding {
            1..=7 => naga::AddressSpace::Storage {
                access: naga::StorageAccess::LOAD,
            },
            8 => naga::AddressSpace::Storage {
                access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE,
            },
            _ => naga::AddressSpace::Uniform,
        };
        assert_eq!(*space, expected, "binding {binding}");
    }

    let entries = virtual_group2_entries();
    let bindings: Vec<u32> = entries.iter().map(|e| e.binding).collect();
    assert_eq!(bindings, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let (mut storage, mut uniform) = (0, 0);
    for e in &entries {
        assert_eq!(e.visibility, ShaderStages::COMPUTE);
        match e.ty {
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                ..
            } => {
                assert_eq!(read_only, e.binding != 8, "binding {}", e.binding);
                storage += 1;
            }
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset,
                min_binding_size,
            } => {
                assert_eq!(e.binding, 9);
                assert!(!has_dynamic_offset);
                assert_eq!(min_binding_size.map(|s| s.get()), Some(32));
                uniform += 1;
            }
            ref other => panic!("binding {}: {other:?}", e.binding),
        }
    }
    assert_eq!((storage, uniform), (8, 1));
}

#[test]
fn vg_material_twin_is_256_bytes() {
    // SurfaceMaterialGpu: crates/raa-shader/src/wgsl/prelude.wgsl:40-50, 256 bytes.
    let (module, _) = analyse(&voxelize_virtual_module_wgsl());
    let (_, ty) = module
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some("RaaGiVgMaterial"))
        .expect("RaaGiVgMaterial is declared");
    let naga::TypeInner::Struct { span, .. } = ty.inner else {
        panic!("RaaGiVgMaterial is a struct");
    };
    assert_eq!(span, 256);
}

#[test]
fn level_view_selects_gi_members_with_software_lists() {
    let view = level_view([-48, -8, -48], 0.2, 96);
    assert_eq!(view.instance_filter, raa_vg::InstanceFilter::Gi as u32);
    assert_eq!(view.instance_filter, 3);
    assert_eq!(view.allow_software, 2);
    assert_eq!(view.lod_threshold, 0.5);
    assert_eq!(view.is_ortho, 1);
    assert_eq!(view.occlusion_enabled, 0);
    // 1 / 0.2.
    assert_eq!(view.lod_scale, 5.0);
    // centre = (origin + 48) x 0.2 = (0, 40 x 0.2, 0).
    let c = view.camera_position;
    assert_eq!((c.x, c.y, c.z), (0.0, 8.0, 0.0));
    // Half extent 0.2 x 96 / 2 = 9.6 m.
    let inside = |p: [f32; 3]| {
        view.planes
            .iter()
            .all(|n| n.x * p[0] + n.y * p[1] + n.z * p[2] + n.w >= 0.0)
    };
    for p in [[0.0, 8.0, 9.5], [9.5, 8.0, 0.0], [0.0, 17.5, 0.0]] {
        assert!(inside(p), "{p:?} is inside");
    }
    for p in [[0.0, 8.0, 9.7], [9.7, 8.0, 0.0], [0.0, 17.7, 0.0]] {
        assert!(!inside(p), "{p:?} is outside");
    }
}
