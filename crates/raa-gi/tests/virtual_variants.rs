//! Contract `gi-virtual-kernel`: the driver B module validates with its one entry point, its
//! group 2 holds eight storage buffers and one uniform, the material twin is 256 bytes and the
//! level view selects the GI members. Follows tests/variants.rs. Consumes the Handoff of
//! `voxelize_virtual_module_wgsl` (crates/raa-gi/src/parts/virtual_driver.rs:42). The last test
//! pins the shared-grid rows of the GI page's virtual stage (lit, off and voxels) inside level 0,
//! the level the voxel debug view draws and the finest one the bounce is traced in.

#![cfg(not(target_arch = "wasm32"))]

use raa_gi::clipmap::{level_view, snapped_origin};
use raa_gi::parts::virtual_driver::{virtual_group2_entries, voxelize_virtual_module_wgsl};
use raa_gi::settings::FINEST_VOXEL_METRES;
use raa_gi::shaders::debug_view_module_wgsl;
use raa_math::Vec3;
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

/// The GI page's virtual stage, read as text: the shared-grid camera must stay where the test
/// below finds its subjects in level 0.
const VIRTUAL_SCENE_JS: &str =
    include_str!("../../../examples/global-illumination/scenes/virtual.js");

/// The three numbers of `const <name> = { eye: [x, y, z]` in `VIRTUAL_SCENE_JS`.
fn scene_eye(name: &str) -> Vec3 {
    let head = format!("const {name} = {{ eye: [");
    let start = VIRTUAL_SCENE_JS
        .find(&head)
        .map(|at| at + head.len())
        .unwrap_or_else(|| panic!("scenes/virtual.js declares {head}"));
    let end = start
        + VIRTUAL_SCENE_JS[start..]
            .find(']')
            .expect("the eye array closes");
    let v: Vec<f32> = VIRTUAL_SCENE_JS[start..end]
        .split(',')
        .map(|c| c.trim().parse().expect("an eye coordinate is a number"))
        .collect();
    assert_eq!(v.len(), 3, "{name} eye has three coordinates");
    Vec3::new(v[0], v[1], v[2])
}

/// Level 0 of the default 96 quality around `eye`: its world min and max corners.
fn level_zero(eye: Vec3) -> ([f32; 3], [f32; 3]) {
    let origin = snapped_origin(eye, FINEST_VOXEL_METRES, 96);
    let min = origin.map(|c| c as f32 * FINEST_VOXEL_METRES);
    (min, min.map(|c| c + 96.0 * FINEST_VOXEL_METRES))
}

/// Whether the box `min..max` is traced in `level`: `raa_gi_level_at` (raa-shader
/// gi_volume.wgsl) keeps a point in a level only more than 1.5 voxels inside its faces.
fn holds(level: &([f32; 3], [f32; 3]), min: [f32; 3], max: [f32; 3]) -> bool {
    let margin = 1.5 * FINEST_VOXEL_METRES;
    (0..3).all(|a| level.0[a] + margin < min[a] && max[a] < level.1[a] - margin)
}

// Subjects of the shared-grid rows (scenes/virtual.js): the wall box at x -6, the sixteen spheres
// (radius 1.06 with their bumps) at x -1.5..6, y 1.1, z -3.75..3.75, and the two boxes.
const WALL: ([f32; 3], [f32; 3]) = ([-6.25, 0.0, -6.0], [-5.75, 4.0, 6.0]);
const FIELD: ([f32; 3], [f32; 3]) = ([-2.56, 0.04, -4.81], [7.06, 2.16, 4.81]);
const BOXES: ([f32; 3], [f32; 3]) = ([-4.8, 0.0, -3.1], [-3.6, 1.2, 3.1]);

#[test]
fn shared_grid_camera_keeps_its_subjects_in_level_zero() {
    // The voxel debug view walks level 0 only: a surface outside it shows smooth (run 14892).
    let debug = debug_view_module_wgsl();
    assert!(debug.contains("let grid = raa_gi_volume_params.levels[0];"));

    // From the field camera (z 15), level 0 spans z 4.8..24: the field, the boxes and the wall
    // lie in level 1, which is what the failed shared-grid images (voxels and lit) were taken in.
    let far = level_zero(scene_eye("CAMERA"));
    assert!(
        (far.0[2] - 4.8).abs() < 1.0e-4,
        "level 0 of CAMERA starts at z {}",
        far.0[2]
    );
    for (min, max) in [FIELD, BOXES, WALL] {
        assert!(
            !holds(&far, min, max),
            "{min:?}..{max:?} is outside level 0 of CAMERA"
        );
    }

    // Every shared-grid row (lit, off and voxels) takes the content's one selection.
    assert!(VIRTUAL_SCENE_JS.contains("'shared-grid': { camera: GRID_CAMERA,"));
    assert!(VIRTUAL_SCENE_JS.contains("selection: SELECTIONS[content],"));

    // GRID_CAMERA: eye (0.4, 9, 4.5) snaps to (0, 8, 3.2), so level 0 spans x -9.6..9.6,
    // y -1.6..17.6 and z -6.4..12.8 and holds every subject whole, 1.5 voxels inside.
    let grid = level_zero(scene_eye("GRID_CAMERA"));
    let expected = ([-9.6, -1.6, -6.4], [9.6, 17.6, 12.8]);
    for a in 0..3 {
        assert!(
            (grid.0[a] - expected.0[a]).abs() < 1.0e-4
                && (grid.1[a] - expected.1[a]).abs() < 1.0e-4,
            "level 0 of GRID_CAMERA is {grid:?}"
        );
    }
    for (min, max) in [FIELD, BOXES, WALL] {
        assert!(
            holds(&grid, min, max),
            "{min:?}..{max:?} is inside level 0 of GRID_CAMERA, {grid:?}"
        );
    }
}
