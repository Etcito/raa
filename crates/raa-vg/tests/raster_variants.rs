//! The S32 raster modules (`raster_common`, `vis_write`, `raster_hw`,
//! `raster_hw_fragment`, `raster_sw`, `raster_sw_body`, `depth_stamp`, `raster_sw_depth` and
//! `shadow_merge`): naga validation of each module, its entry points and stages, the group-2
//! bindings, the per-stage storage-buffer budget, the flat integer varyings, the one depth
//! helper their visibility write calls and the CPU twin of the stamped cascade depth words.

#![cfg(not(target_arch = "wasm32"))]

use raa_shader::{Composer, DEPTH_WGSL};
use raa_vg::culling::PASS_VIEW_WGSL;
use raa_vg::raster::{
    DEPTH_STAMP_RANGE, DEPTH_STAMP_SHIFT, DEPTH_STAMPS, RASTER_COMMON_WGSL, RASTER_SW_BODY_WGSL,
    VIS_WRITE_WGSL, raster_hw_module_wgsl, raster_sw_depth_module_wgsl, raster_sw_module_wgsl,
    shadow_merge_module_wgsl, stamp_of, stamped, stamped_depth,
};
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

/// The storage bindings (1-8) an entry point reaches in group 2.
fn stage_storage(module: &naga::Module, entry: &naga::EntryPoint) -> BTreeSet<u32> {
    used_bindings(module, entry)
        .into_iter()
        .filter(|(group, binding)| *group == 2 && (1..=8).contains(binding))
        .map(|(_, binding)| binding)
        .collect()
}

#[test]
fn raster_hw_module_validates_with_its_entry_points_and_stages() {
    let wgsl = raster_hw_module_wgsl();
    assert_eq!(Composer::validate(&wgsl), Ok(()));
    let module = parse(&wgsl);
    let mut entry_points: Vec<(&str, naga::ShaderStage)> = module
        .entry_points
        .iter()
        .map(|entry| (entry.name.as_str(), entry.stage))
        .collect();
    entry_points.sort_unstable();
    assert_eq!(
        entry_points,
        [
            ("fs_visibility", naga::ShaderStage::Fragment),
            ("vs_depth", naga::ShaderStage::Vertex),
            ("vs_visibility", naga::ShaderStage::Vertex),
        ]
    );
}

#[test]
fn raster_hw_binds_group_two_zero_to_eight() {
    let module = parse(&raster_hw_module_wgsl());
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
        [
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8)
        ]
    );
    for (_, global) in module.global_variables.iter() {
        let binding = global.binding.expect("every global is bound");
        assert_eq!(binding.group, 2, "no binding sits outside group 2");
        match binding.binding {
            0 => assert!(matches!(global.space, naga::AddressSpace::Uniform)),
            1..=6 => assert!(matches!(
                global.space,
                naga::AddressSpace::Storage { access } if access == naga::StorageAccess::LOAD
            )),
            _ => assert!(matches!(
                global.space,
                naga::AddressSpace::Storage { access }
                    if access == naga::StorageAccess::LOAD | naga::StorageAccess::STORE
            )),
        }
    }
}

#[test]
fn raster_stages_stay_within_the_storage_budget() {
    let module = parse(&raster_hw_module_wgsl());
    for entry in &module.entry_points {
        let storage = stage_storage(&module, entry);
        assert!(
            storage.len() <= 8,
            "entry point {} uses more than eight storage buffers",
            entry.name
        );
        match entry.stage {
            naga::ShaderStage::Vertex => assert_eq!(
                storage,
                BTreeSet::from([1, 2, 3, 4, 5, 6]),
                "the vertex stage reads the scene and cull-state buffers"
            ),
            naga::ShaderStage::Fragment => assert_eq!(
                storage,
                BTreeSet::from([7, 8]),
                "the fragment stage writes only the visibility buffers"
            ),
            _ => panic!("the raster module holds vertex and fragment stages only"),
        }
    }
}

#[test]
fn vertex_integer_outputs_are_flat() {
    let module = parse(&raster_hw_module_wgsl());
    let entry = module
        .entry_points
        .iter()
        .find(|entry| entry.name == "vs_visibility")
        .expect("vs_visibility is an entry point");
    let result = entry
        .function
        .result
        .as_ref()
        .expect("vs_visibility returns its varyings");
    let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
        panic!("the varyings of vs_visibility are a struct");
    };
    let mut integer_outputs = 0;
    for member in members {
        let naga::TypeInner::Scalar(scalar) = &module.types[member.ty].inner else {
            continue;
        };
        if !matches!(scalar.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint) {
            continue;
        }
        integer_outputs += 1;
        assert!(
            matches!(
                member.binding,
                Some(naga::Binding::Location {
                    interpolation: Some(naga::Interpolation::Flat),
                    ..
                })
            ),
            "the integer varying {:?} is flat",
            member.name
        );
    }
    assert_eq!(
        integer_outputs, 1,
        "the flat visible id is the only integer varying"
    );
}

/// The module's own WGSL (after the `DEPTH_WGSL` declarations) calls exactly one depth helper:
/// `raa_depth_key`, which orders standard-Z depth the way the reversed-Z bits of the source do.
#[test]
fn raster_calls_only_the_depth_key_helper() {
    let wgsl = raster_hw_module_wgsl();
    assert!(wgsl.starts_with(DEPTH_WGSL));
    assert!(
        wgsl[DEPTH_WGSL.len()..].starts_with(PASS_VIEW_WGSL),
        "the scene twins come right after the depth helpers"
    );
    let order = [
        DEPTH_WGSL,
        PASS_VIEW_WGSL,
        RASTER_COMMON_WGSL,
        VIS_WRITE_WGSL,
    ]
    .concat();
    assert!(
        wgsl.starts_with(&order),
        "the builder concatenates DEPTH_WGSL, PASS_VIEW_WGSL, RASTER_COMMON_WGSL and VIS_WRITE_WGSL in order"
    );
    let body = wgsl
        .strip_prefix(DEPTH_WGSL)
        .expect("the raster module starts with the depth helper");
    assert_eq!(depth_calls(body), ["raa_depth_key"]);
    let vis_write = wgsl
        .find(VIS_WRITE_WGSL)
        .expect("vis_write is part of the module");
    let vertex = wgsl
        .find("fn vs_visibility(")
        .expect("the vertex entry point follows the shared writes");
    assert!(
        vis_write < vertex,
        "write_visibility is declared before its callers"
    );
}

/// The S32 hardware raster on a software adapter: a baked triangle fan drawn through the real
/// cull and `Rasterizer::record` into a 64x64 visibility buffer.
///
/// Inputs, all initialised here because no other pass of the frame exists in this test:
/// the visibility depth is cleared to empty keys (0) before the passes; the fan is one baked
/// mesh of 8 triangles with centre (0, 0, 0), radius 1 m, in the XY plane and counter-clockwise
/// from +Z; the instance model is the identity with the CASTS_SHADOW flag; the camera sits at
/// (0, 0, 4) looking at the origin, fov 60 degrees, near 0.1, far 16. Both the instance (radius
/// 1 m at the origin) and the camera stay inside a 16 m box, so the view's extent encloses
/// them. Only the visibility buffer is read back: no depth or HDR target.
#[test]
fn hardware_raster_writes_a_baked_fan() {
    use raa_cluster::{BuildOptions, ClusterStore, OwnedClusterInput, bake};
    use raa_core::MeshId;
    use raa_core::testing::{block_on, skip, software_gpu};
    use raa_math::{Mat4, Vec3};
    use raa_vg::culling::{CullPhase, CullPipelines, CullWork, MAIN_FIRST};
    use raa_vg::raster::{RasterTarget, Rasterizer, VisibilityBuffer};
    use raa_vg::{
        INSTANCE_CASTS_SHADOW, MeshKeys, PassView, VgInstanceRecord, VgScene, ViewSlots,
        hierarchy_word,
    };

    let test = "hardware_raster_writes_a_baked_fan";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let device = gpu.device();

    let mut input = OwnedClusterInput::single_triangle();
    input.uvs = None;
    let mut positions = vec![[0.0f32, 0.0, 0.0]];
    let mut normals = vec![[0.0f32, 0.0, 1.0]];
    let mut indices = Vec::new();
    for corner in 0..8u32 {
        let angle = corner as f32 / 8.0 * std::f32::consts::TAU;
        positions.push([angle.cos(), angle.sin(), 0.0]);
        normals.push([0.0, 0.0, 1.0]);
    }
    for corner in 0..8u32 {
        indices.extend_from_slice(&[0, 1 + corner, 1 + (corner + 1) % 8]);
    }
    input.mesh.positions = positions;
    input.mesh.normals = normals;
    input.mesh.indices = indices;

    let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the fan bakes");
    let quant = [
        mesh.quant_origin[0],
        mesh.quant_origin[1],
        mesh.quant_origin[2],
    ];
    let quant_step = mesh.quant_step;
    let uv = [mesh.uv_origin[0], mesh.uv_origin[1]];
    let uv_step = mesh.uv_step;
    let bounds = [
        mesh.bounds_center[0],
        mesh.bounds_center[1],
        mesh.bounds_center[2],
        mesh.bounds_radius,
    ];
    assert!(
        mesh.bounds_radius > 0.5 && mesh.bounds_radius <= 1.5,
        "the fan spans its 1 m radius, got {}",
        mesh.bounds_radius
    );

    let mut store = ClusterStore::new();
    store.attach(1, vec![mesh]);
    let mut keys = MeshKeys::default();
    keys.set(1, &[MeshId(0)]);

    let mut scene = VgScene::new();
    assert!(scene.upload_pool(&gpu, &store, &keys), "the pool uploads");
    let range = scene.section_range(1, 0).expect("the section is packed");
    let (_, depth, root) = scene
        .section_header(1, 0)
        .expect("the section header exists");
    scene.upload_instances(
        &gpu,
        &[VgInstanceRecord {
            model: Mat4::IDENTITY,
            previous_rows: [[0.0; 4]; 3],
            quant: [quant[0], quant[1], quant[2], quant_step],
            uv_quant: [uv[0], uv[1], uv_step[0], uv_step[1]],
            ranges: [
                range.node_base,
                range.cluster_base,
                range.vertex_base,
                range.triangle_base,
            ],
            ids: [0, 0, INSTANCE_CASTS_SHADOW, hierarchy_word(depth, root)],
            bounds,
        }],
    );
    assert_eq!(scene.instance_count(), 1, "one instance record is uploaded");

    let eye = Vec3::new(0.0, 0.0, 4.0);
    let proj = Mat4::perspective_rh_zo(60f32.to_radians(), 1.0, 0.1, 16.0);
    let view_proj = proj * Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
    let view = PassView::perspective(
        view_proj,
        view_proj,
        eye,
        [64.0, 64.0],
        [proj.cols[0][0], proj.cols[1][1]],
        0.1,
        1.0,
    );
    let slots = ViewSlots::new(&gpu);
    slots.write(&gpu, MAIN_FIRST, &view);

    let pyramid = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("raa-vg raster variants pyramid"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue().write_texture(
        pyramid.as_image_copy(),
        &1.0f32.to_le_bytes(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let pyramid_view = pyramid.create_view(&wgpu::TextureViewDescriptor::default());

    let vis = VisibilityBuffer::new(&gpu, (64, 64));
    let pipelines = CullPipelines::new(&gpu);
    let work = CullWork::new(&gpu, &pipelines, "raa-vg raster variants", 1024, 1024);
    let scene_group = pipelines.scene_bind_group(&gpu, &slots, &scene, &pyramid_view);
    let rasterizer = Rasterizer::new(&gpu);
    let bindings = rasterizer.bindings(&gpu, &slots, MAIN_FIRST, &scene, &work, &vis);

    let bytes = 64 * 64 * 4;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa-vg raster variants readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut encoder = device.create_command_encoder(&Default::default());
    vis.clear(&mut encoder);
    pipelines.record(
        &mut encoder,
        &scene,
        &scene_group,
        MAIN_FIRST,
        &work,
        CullPhase::First,
    );
    rasterizer.record(
        &mut encoder,
        &bindings,
        &work,
        RasterTarget::Visibility(&vis),
    );
    encoder.copy_buffer_to_buffer(vis.depth(), 0, &readback, 0, bytes);
    gpu.queue().submit(Some(encoder.finish()));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    let error = block_on(errors.pop());
    assert!(error.is_none(), "raster validation: {error:?}");

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    rx.recv().expect("map callback").expect("map");
    let mapped = slice.get_mapped_range().expect("mapped range");
    let word = |index: usize| {
        u32::from_le_bytes(
            mapped[index * 4..index * 4 + 4]
                .try_into()
                .expect("four bytes"),
        )
    };
    let centre_key = word(32 * 64 + 32);
    let corner_key = word(0);
    drop(mapped);
    readback.unmap();

    assert_ne!(centre_key, 0, "the fan covers the centre pixel");
    let centre_depth = 1.0 - f32::from_bits(centre_key);
    assert!(
        centre_depth > 0.0 && centre_depth < 1.0,
        "the centre depth {centre_depth} lies between the near and far planes"
    );
    assert_eq!(corner_key, 0, "the corner pixel stays an empty key");
}

#[test]
fn raster_sw_module_validates_as_a_compute_main() {
    let wgsl = raster_sw_module_wgsl();
    assert_eq!(Composer::validate(&wgsl), Ok(()));
    let module = parse(&wgsl);
    let entry_points: Vec<(&str, naga::ShaderStage)> = module
        .entry_points
        .iter()
        .map(|entry| (entry.name.as_str(), entry.stage))
        .collect();
    assert_eq!(entry_points, [("main", naga::ShaderStage::Compute)]);
}

#[test]
fn raster_sw_binds_group_two_zero_to_eight() {
    let module = parse(&raster_sw_module_wgsl());
    let mut bindings: Vec<(u32, u32)> = module
        .global_variables
        .iter()
        .filter_map(|(_, global)| {
            global
                .binding
                .map(|binding| (binding.group, binding.binding))
        })
        .collect();
    bindings.sort_unstable();
    assert_eq!(
        bindings,
        (0..=8).map(|binding| (2, binding)).collect::<Vec<_>>(),
        "the software pass binds the same nine group-2 entries as the hardware pass"
    );
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = global.binding else {
            continue; // the workgroup memory of the body
        };
        assert_eq!(binding.group, 2, "no binding sits outside group 2");
        match binding.binding {
            0 => assert!(matches!(global.space, naga::AddressSpace::Uniform)),
            1..=6 => assert!(matches!(
                global.space,
                naga::AddressSpace::Storage { access } if access == naga::StorageAccess::LOAD
            )),
            _ => assert!(matches!(
                global.space,
                naga::AddressSpace::Storage { access }
                    if access == naga::StorageAccess::LOAD | naga::StorageAccess::STORE
            )),
        }
    }
}

#[test]
fn raster_sw_stage_reads_at_most_eight_storage_buffers() {
    let module = parse(&raster_sw_module_wgsl());
    let entry = module
        .entry_points
        .iter()
        .find(|entry| entry.name == "main")
        .expect("main is an entry point");
    let storage = stage_storage(&module, entry);
    assert!(
        storage.len() <= 8,
        "the compute stage uses more than eight storage buffers"
    );
    assert_eq!(
        storage,
        BTreeSet::from([1, 2, 3, 4, 5, 6, 7, 8]),
        "main reads the scene and cull-state buffers and writes the visibility buffers"
    );
}

/// The software module's own WGSL calls exactly one depth helper, `raa_depth_key`, reached
/// through `write_visibility`; the body adds no depth comparison of its own. One workgroup
/// per software cluster strides over `sw_count - sw_base` and reads `visible[sw_base + item]`.
#[test]
fn raster_sw_calls_only_the_depth_key_helper() {
    let wgsl = raster_sw_module_wgsl();
    let order = [
        DEPTH_WGSL,
        PASS_VIEW_WGSL,
        RASTER_COMMON_WGSL,
        VIS_WRITE_WGSL,
    ]
    .concat();
    assert!(
        wgsl.starts_with(&order),
        "the builder concatenates DEPTH_WGSL, PASS_VIEW_WGSL, RASTER_COMMON_WGSL and VIS_WRITE_WGSL in order"
    );
    assert!(
        wgsl.ends_with(RASTER_SW_BODY_WGSL),
        "the shared body is the last section of the module"
    );
    let body = wgsl
        .strip_prefix(DEPTH_WGSL)
        .expect("the raster module starts with the depth helper");
    assert_eq!(depth_calls(body), ["raa_depth_key"]);
    assert!(
        RASTER_SW_BODY_WGSL.contains("const GROUP_SIZE: u32 = 128u;"),
        "the body defines the workgroup size once for every module that includes it"
    );
    assert!(
        RASTER_SW_BODY_WGSL.contains("let visible_index = state.sw_base + item;"),
        "the software window starts at sw_base"
    );
}

/// The CPU twin of `depth_stamp.wgsl`: every listed stamp and depth round-trips within 1e-4
/// and never reads back nearer, a nearer depth takes a larger word under one stamp, and a
/// later stamp beats every word an earlier stamp left.
#[test]
fn stamped_words_round_trip_and_order_depths_and_stamps() {
    assert_eq!(
        (DEPTH_STAMP_SHIFT, DEPTH_STAMP_RANGE, DEPTH_STAMPS),
        (24, 16777215, 255)
    );
    let depths = [0.0, 0.3, 0.9, 1.0];
    for stamp in [1, 37, 255] {
        let mut previous = u32::MAX;
        for depth in depths {
            let word = stamped(stamp, depth);
            assert_eq!(stamp_of(word), stamp);
            let back = stamped_depth(word);
            assert!(
                (back - depth).abs() <= 1e-4,
                "stamp {stamp}: depth {depth} reads back as {back}"
            );
            assert!(
                back >= depth - 1e-6,
                "stamp {stamp}: depth {depth} reads back nearer at {back}"
            );
            assert!(
                word < previous,
                "stamp {stamp}: a nearer depth takes a larger word"
            );
            previous = word;
        }
    }
    for earlier in 1..DEPTH_STAMPS {
        assert!(
            stamped(earlier + 1, 1.0) > stamped(earlier, 0.0),
            "stamp {} beats every word of stamp {earlier}",
            earlier + 1
        );
    }
}

/// The cascade depth modules: both validate, the software depth module is the compute `main`
/// over group 2 bindings 0-7, the merge module is the fullscreen `vs_main`/`fs_main` over
/// bindings 0-1 with `fs_main` returning `frag_depth`, each entry point stays within eight
/// storage buffers, and each calls only `raa_depth_key` and `raa_depth_from_key`.
#[test]
fn cascade_depth_modules_validate_with_the_contract() {
    let sw_depth = raster_sw_depth_module_wgsl();
    let merge = shadow_merge_module_wgsl();
    assert_eq!(Composer::validate(&sw_depth), Ok(()));
    assert_eq!(Composer::validate(&merge), Ok(()));
    assert!(
        sw_depth.starts_with(&[DEPTH_WGSL, PASS_VIEW_WGSL, RASTER_COMMON_WGSL].concat()),
        "the software depth builder starts with DEPTH_WGSL, PASS_VIEW_WGSL and raster_common"
    );
    assert!(
        sw_depth.ends_with(RASTER_SW_BODY_WGSL),
        "the shared body is the last section of the software depth module"
    );
    assert!(
        merge.starts_with(&[DEPTH_WGSL, PASS_VIEW_WGSL].concat()),
        "the merge builder starts with DEPTH_WGSL and PASS_VIEW_WGSL"
    );
    assert!(
        !merge.contains(RASTER_COMMON_WGSL),
        "the merge module does not include raster_common"
    );

    let module = parse(&sw_depth);
    let entries: Vec<(&str, naga::ShaderStage)> = module
        .entry_points
        .iter()
        .map(|entry| (entry.name.as_str(), entry.stage))
        .collect();
    assert_eq!(entries, [("main", naga::ShaderStage::Compute)]);
    let main = &module.entry_points[0];
    assert_eq!(
        stage_storage(&module, main),
        BTreeSet::from([1, 2, 3, 4, 5, 6, 7]),
        "main reads the scene, the cull state and the stamped depth buffer"
    );
    let mut bindings: Vec<(u32, u32)> = module
        .global_variables
        .iter()
        .filter_map(|(_, global)| {
            global
                .binding
                .map(|binding| (binding.group, binding.binding))
        })
        .collect();
    bindings.sort_unstable();
    assert_eq!(
        bindings,
        (0..=7).map(|binding| (2, binding)).collect::<Vec<_>>(),
        "the software depth pass binds group 2 entries 0-7"
    );
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = global.binding else {
            continue; // the workgroup memory of the body
        };
        assert_eq!(binding.group, 2, "no binding sits outside group 2");
        match binding.binding {
            0 => assert!(matches!(global.space, naga::AddressSpace::Uniform)),
            1..=6 => assert!(matches!(
                global.space,
                naga::AddressSpace::Storage { access } if access == naga::StorageAccess::LOAD
            )),
            _ => assert!(matches!(
                global.space,
                naga::AddressSpace::Storage { access }
                    if access.contains(naga::StorageAccess::LOAD | naga::StorageAccess::STORE)
            )),
        }
    }

    let module = parse(&merge);
    let mut entries: Vec<(&str, naga::ShaderStage)> = module
        .entry_points
        .iter()
        .map(|entry| (entry.name.as_str(), entry.stage))
        .collect();
    entries.sort_unstable();
    assert_eq!(
        entries,
        [
            ("fs_main", naga::ShaderStage::Fragment),
            ("vs_main", naga::ShaderStage::Vertex),
        ]
    );
    for entry in &module.entry_points {
        assert!(
            stage_storage(&module, entry).len() <= 8,
            "entry point {} uses more than eight storage buffers",
            entry.name
        );
    }
    let fragment = module
        .entry_points
        .iter()
        .find(|entry| entry.name == "fs_main")
        .expect("fs_main is an entry point");
    let result = fragment
        .function
        .result
        .as_ref()
        .expect("fs_main returns frag_depth");
    assert!(
        matches!(
            result.binding,
            Some(naga::Binding::BuiltIn(naga::BuiltIn::FragDepth))
        ),
        "fs_main writes frag_depth"
    );
    let mut bindings: Vec<(u32, u32)> = module
        .global_variables
        .iter()
        .filter_map(|(_, global)| {
            global
                .binding
                .map(|binding| (binding.group, binding.binding))
        })
        .collect();
    bindings.sort_unstable();
    assert_eq!(bindings, [(2, 0), (2, 1)]);
    for (_, global) in module.global_variables.iter() {
        let binding = global.binding.expect("every merge global is bound");
        match binding.binding {
            0 => assert!(matches!(
                global.space,
                naga::AddressSpace::Storage { access } if access == naga::StorageAccess::LOAD
            )),
            1 => assert!(matches!(global.space, naga::AddressSpace::Uniform)),
            other => panic!("unexpected merge binding {other}"),
        }
    }

    for wgsl in [&sw_depth, &merge] {
        let body = wgsl
            .strip_prefix(DEPTH_WGSL)
            .expect("the cascade module starts with the depth helper");
        let calls = depth_calls(body);
        for call in &calls {
            assert!(
                ["raa_depth_key", "raa_depth_from_key"].contains(call),
                "{call} is not one of the stamped depth helpers"
            );
        }
        assert_eq!(calls, ["raa_depth_key", "raa_depth_from_key"]);
    }
}
