//! The S32 resolve module (`resolve.wgsl`): naga validation of the composed module, its
//! `main` compute entry point and workgroup size, the group-2 bindings 0-12, the per-stage
//! storage-buffer and storage-texture budget, the one depth helper it calls, the removed
//! wetness/waterline/tier text and the 320-byte `VgFrame` record.

#![cfg(not(target_arch = "wasm32"))]

use raa_core::testing::{block_on, skip, software_gpu};
use raa_math::{Mat4, Vec3};
use raa_shader::{Composer, DEPTH_WGSL};
use raa_vg::culling::{CullPipelines, CullWork, PASS_VIEW_WGSL};
use raa_vg::raster::VisibilityBuffer;
use raa_vg::resolve::{
    Resolve, VG_FRAME_GPU_SIZE, VgFrameParams, VgSurface, resolve_module_wgsl, surface_bind_group,
    surface_layout, write_vg_frame,
};
use raa_vg::scene::VgScene;
use std::collections::BTreeSet;
use wgpu::naga;

/// Every `raa_depth_` identifier in `src`, unique and in first-appearance order.
fn depth_calls(src: &str) -> Vec<&str> {
    const PREFIX: &str = "raa_depth_";
    let mut names: Vec<&str> = Vec::new();
    for (start, _) in src.match_indices(PREFIX) {
        let rest = &src[start..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let name = &rest[..end];
        if name.len() > PREFIX.len() && !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

fn parse(source: &str) -> naga::Module {
    naga::front::wgsl::parse_str(source).expect("the module parses")
}

/// The CPU twin of the WGSL `lod_ramp`: the DAG level through blue (0), green (6) and red
/// (12), linear between the stops and clamped above 12.
fn lod_ramp(level: u32) -> [f32; 3] {
    const BLUE: [f32; 3] = [0.1, 0.3, 1.0];
    const GREEN: [f32; 3] = [0.2, 0.9, 0.3];
    const RED: [f32; 3] = [1.0, 0.2, 0.1];
    let t = level.min(12) as f32;
    if t <= 6.0 {
        mix3(BLUE, GREEN, t / 6.0)
    } else {
        mix3(GREEN, RED, (t - 6.0) / 6.0)
    }
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn assert_rgb(actual: [f32; 3], expected: [f32; 3]) {
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e).abs() < 1e-6,
            "expected {expected:?}, got {actual:?}"
        );
    }
}

/// Every `(group, binding)` the entry point reaches, following the functions it calls.
fn used_bindings(module: &naga::Module, entry: &naga::EntryPoint) -> BTreeSet<(u32, u32)> {
    fn walk(
        module: &naga::Module,
        function: &naga::Function,
        seen: &mut BTreeSet<naga::Handle<naga::Function>>,
        out: &mut BTreeSet<(u32, u32)>,
    ) {
        for (_, expression) in function.expressions.iter() {
            if let naga::Expression::GlobalVariable(handle) = *expression
                && let Some(binding) = module.global_variables[handle].binding
            {
                out.insert((binding.group, binding.binding));
            }
        }
        let mut blocks = vec![&function.body];
        while let Some(block) = blocks.pop() {
            for statement in block.iter() {
                match statement {
                    naga::Statement::Call { function, .. } => {
                        if seen.insert(*function) {
                            walk(module, &module.functions[*function], seen, out);
                        }
                    }
                    naga::Statement::Block(inner) => blocks.push(inner),
                    naga::Statement::If { accept, reject, .. } => blocks.extend([accept, reject]),
                    naga::Statement::Switch { cases, .. } => {
                        blocks.extend(cases.iter().map(|case| &case.body));
                    }
                    naga::Statement::Loop {
                        body, continuing, ..
                    } => blocks.extend([body, continuing]),
                    _ => {}
                }
            }
        }
    }
    let mut bindings = BTreeSet::new();
    walk(module, &entry.function, &mut BTreeSet::new(), &mut bindings);
    bindings
}

fn global(module: &naga::Module, group: u32, binding: u32) -> &naga::GlobalVariable {
    module
        .global_variables
        .iter()
        .find(|(_, global)| global.binding == Some(naga::ResourceBinding { group, binding }))
        .map(|(_, global)| global)
        .unwrap_or_else(|| panic!("group {group} binding {binding} is declared"))
}

#[test]
fn resolve_module_validates_with_the_main_compute_entry_point() {
    let wgsl = resolve_module_wgsl();
    assert_eq!(Composer::validate(&wgsl), Ok(()));
    let module = parse(&wgsl);
    assert_eq!(
        module.entry_points.len(),
        1,
        "the resolve has one entry point"
    );
    let entry = &module.entry_points[0];
    assert_eq!(entry.name, "main");
    assert_eq!(entry.stage, naga::ShaderStage::Compute);
    assert_eq!(entry.workgroup_size, [8, 8, 1]);
}

#[test]
fn resolve_binds_group_two_zero_to_twelve() {
    let module = parse(&resolve_module_wgsl());
    let mut bindings: Vec<(u32, u32)> = module
        .global_variables
        .iter()
        .map(|(_, global)| {
            let binding = global.binding.expect("every global is bound");
            (binding.group, binding.binding)
        })
        .collect();
    bindings.sort_unstable();
    assert_eq!(
        bindings,
        (0..=12).map(|binding| (2, binding)).collect::<Vec<_>>()
    );
    for (_, global) in module.global_variables.iter() {
        assert_eq!(
            global.binding.expect("every global is bound").group,
            2,
            "no binding sits outside group 2"
        );
    }
    for binding in 1..=8 {
        let global = global(&module, 2, binding);
        assert!(
            matches!(
                global.space,
                naga::AddressSpace::Storage { access }
                    if access == naga::StorageAccess::LOAD
            ),
            "binding {binding} is a read-only storage buffer"
        );
    }
    for binding in 9..=12 {
        let global = global(&module, 2, binding);
        assert_eq!(global.space, naga::AddressSpace::Handle);
        assert!(
            matches!(
                module.types[global.ty].inner,
                naga::TypeInner::Image {
                    class: naga::ImageClass::Storage {
                        access: naga::StorageAccess::STORE,
                        ..
                    },
                    ..
                }
            ),
            "binding {binding} is a write-only storage texture"
        );
    }
}

#[test]
fn resolve_stage_uses_eight_storage_buffers_and_four_storage_textures() {
    let module = parse(&resolve_module_wgsl());
    let entry = &module.entry_points[0];
    let used = used_bindings(&module, entry);
    assert_eq!(
        used,
        (0..=12)
            .map(|binding| (2, binding))
            .collect::<BTreeSet<_>>(),
        "the stage reaches every declared binding"
    );
    let mut storage_buffers = 0;
    let mut storage_textures = 0;
    for (group, binding) in used {
        let global = global(&module, group, binding);
        match global.space {
            naga::AddressSpace::Storage { access } => {
                assert_eq!(access, naga::StorageAccess::LOAD);
                storage_buffers += 1;
            }
            naga::AddressSpace::Handle => {
                if matches!(
                    module.types[global.ty].inner,
                    naga::TypeInner::Image {
                        class: naga::ImageClass::Storage { .. },
                        ..
                    }
                ) {
                    storage_textures += 1;
                }
            }
            naga::AddressSpace::Uniform => {}
            ref other => panic!("unexpected binding {binding} in {other:?}"),
        }
    }
    assert_eq!(storage_buffers, 8, "S32 allows eight storage buffers");
    assert_eq!(storage_textures, 4, "S32 allows four storage textures");
}

#[test]
fn resolve_calls_only_the_depth_from_key_helper() {
    let wgsl = resolve_module_wgsl();
    assert!(wgsl.starts_with(DEPTH_WGSL));
    assert!(
        wgsl[DEPTH_WGSL.len()..].starts_with(PASS_VIEW_WGSL),
        "the scene twins come right after the depth helpers"
    );
    let body = wgsl
        .strip_prefix(DEPTH_WGSL)
        .expect("the resolve module starts with the depth helper");
    let calls = depth_calls(body);
    let allowed = ["raa_depth_from_key", "raa_depth_is_far"];
    for call in &calls {
        assert!(
            allowed.contains(call),
            "{call} is not one of the depth helpers the resolve may call"
        );
    }
    assert_eq!(calls, ["raa_depth_from_key"]);
}

#[test]
fn resolve_drops_the_wetness_waterline_and_tier_textures() {
    let wgsl = resolve_module_wgsl();
    for removed in ["waterline", "wetness", "texture_2d_array"] {
        assert!(
            !wgsl.contains(removed),
            "the adapted resolve removes {removed}"
        );
    }
}

#[test]
fn resolve_has_one_debug_branch_per_code_one_to_four() {
    let wgsl = resolve_module_wgsl();
    for code in 1..=4 {
        let branch = format!("debug_view == {code}u");
        assert!(
            wgsl.contains(&branch),
            "the resolve has a branch for debug code {code}"
        );
    }
}

#[test]
fn lod_ramp_gives_blue_green_and_red_and_clamps_above_twelve() {
    assert_rgb(lod_ramp(0), [0.1, 0.3, 1.0]);
    assert_rgb(lod_ramp(6), [0.2, 0.9, 0.3]);
    assert_rgb(lod_ramp(12), [1.0, 0.2, 0.1]);
    assert_rgb(lod_ramp(3), [0.15, 0.6, 0.65]);
    assert_eq!(
        lod_ramp(64),
        lod_ramp(12),
        "levels above 12 clamp to the red stop"
    );
}

#[test]
fn vg_frame_is_320_bytes() {
    let module = parse(&resolve_module_wgsl());
    let (handle, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("VgFrame"))
        .expect("VgFrame is declared");
    let naga::TypeInner::Struct { span, .. } = &ty.inner else {
        panic!("VgFrame is a struct");
    };
    assert_eq!(*span, 320);
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).expect("the types lay out");
    assert_eq!(layouter[handle].size, 320);
}

fn float_at(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn word_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn assert_matrix(bytes: &[u8], at: usize, expected: &Mat4) {
    for (k, value) in expected.to_cols_array().iter().enumerate() {
        assert_eq!(float_at(bytes, at + 4 * k), *value);
    }
}

#[test]
fn write_vg_frame_fills_the_320_byte_record() {
    let params = VgFrameParams {
        view_proj: Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
        unjittered_view_proj: Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0)),
        previous_view_proj: Mat4::from_translation(Vec3::new(7.0, 8.0, 9.0)),
        inv_view_proj: Mat4::from_translation(Vec3::new(10.0, 11.0, 12.0)),
        viewport: [64.0, 32.0, 1.0 / 64.0, 1.0 / 32.0],
        camera_jitter: [0.5, -0.5, 0.25, -0.25],
        camera_pos: [13.0, 14.0, 15.0, 16.0],
        debug_view: 1,
        history_valid: true,
    };
    let mut out = [0xAAu8; VG_FRAME_GPU_SIZE];
    write_vg_frame(&mut out, &params);
    assert_matrix(&out, 0, &params.view_proj);
    assert_matrix(&out, 64, &params.unjittered_view_proj);
    assert_matrix(&out, 128, &params.previous_view_proj);
    assert_matrix(&out, 192, &params.inv_view_proj);
    for (k, value) in params.viewport.iter().enumerate() {
        assert_eq!(float_at(&out, 256 + 4 * k), *value);
    }
    for (k, value) in params.camera_jitter.iter().enumerate() {
        assert_eq!(float_at(&out, 272 + 4 * k), *value);
    }
    for (k, value) in params.camera_pos.iter().enumerate() {
        assert_eq!(float_at(&out, 288 + 4 * k), *value);
    }
    assert_eq!(word_at(&out, 304), params.debug_view);
    assert_eq!(word_at(&out, 308), 1, "the history is valid");
    assert_eq!(word_at(&out, 312), 0, "flags.z is reserved");
    assert_eq!(word_at(&out, 316), 0, "flags.w is reserved");
}

#[test]
fn resolve_and_surface_bind_groups_create_without_validation_error() {
    let test = "resolve_and_surface_bind_groups_create_without_validation_error";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let device = gpu.device();
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let resolve = Resolve::new(&gpu);
    let surface = VgSurface::new(&gpu, (64, 64));
    let vis = VisibilityBuffer::new(&gpu, (64, 64));
    let pipelines = CullPipelines::new(&gpu);
    let work = CullWork::new(&gpu, &pipelines, "raa-vg resolve variants", 16, 16);
    let scene = VgScene::new();

    let frame = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa-vg resolve variants frame"),
        size: VG_FRAME_GPU_SIZE as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = VgFrameParams {
        viewport: [64.0, 64.0, 1.0 / 64.0, 1.0 / 64.0],
        ..Default::default()
    };
    let mut bytes = [0u8; VG_FRAME_GPU_SIZE];
    write_vg_frame(&mut bytes, &params);
    gpu.queue().write_buffer(&frame, 0, &bytes);
    let materials = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa-vg resolve variants materials"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let shade = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa-vg resolve variants shade"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let group = resolve.bindings(&gpu, &frame, &vis, &work, &scene, &materials, &surface);
    let layout = surface_layout(&gpu);
    let _surface_group = surface_bind_group(&gpu, &layout, &surface, &shade);

    let mut encoder = device.create_command_encoder(&Default::default());
    vis.clear(&mut encoder);
    resolve.record(&mut encoder, &group, surface.size());
    gpu.queue().submit(Some(encoder.finish()));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    let error = block_on(errors.pop());
    assert!(error.is_none(), "resolve validation: {error:?}");
    assert_eq!(surface.size(), (64, 64));
    assert_eq!(surface.allocated_bytes(), 64 * 64 * (16 + 8 + 16 + 4));
    assert_eq!(resolve.allocated_bytes(), 16);
}
