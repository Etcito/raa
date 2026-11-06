//! Section 67 WGSL evaluator: complete shader variants generated through the public
//! generation interfaces parse and type-check with naga; no GPU rendering is done here.
use raa_sdf_trace::wgsl::{
    PROGRAM_EVAL_WGSL, VOLUME_ABI_VERSION, VOLUME_COARSE_TILE, VOLUME_MAX_AXIS, VOLUME_MIP_LEVELS,
    VolumeUniform, composite_module, eval_module, trace_module, volume_mip_layout, volume_module,
};
use raa_sdf_trace::{
    COMPOSITE_NAME, COMPOSITE_ORDER, Instance, Source, TRACE_NAME, TRACE_ORDER, TraceProgram,
    VolumeLayout, target_size,
};

fn program(sources: usize, code_len: u32) -> TraceProgram {
    let source = Source {
        code_start: 0,
        code_len,
        data_start: 0,
        data_len: 18,
        min: [-1.0; 3],
        max: [1.0; 3],
        lipschitz: 1.0,
        flags: 0,
    };
    TraceProgram {
        sources: vec![source; sources],
        code: vec![1; code_len as usize],
        data: vec![0.0; 18],
        materials: Vec::new(),
        instances: (0..sources)
            .map(|index| Instance {
                world_from_local: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                local_from_world: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dist_scale: 1.0,
                source: index as u32,
                material: -1,
                key: index as u32,
                deform_kind: u32::from(index % 2 == 1),
                bend_angle: 0.5,
                bend_length: 2.0,
                axes: 56 | (index as u32 & 1), // Bent cases: negative along, toward=7.
                bend: (index % 2 == 1).then_some([0.0, 30.0, 90.0, -1.0, 1.0, 0.0, 1.0, 0.0]),
            })
            .collect(),
    }
}

fn check(program: &TraceProgram) {
    for source in [
        eval_module(program),
        trace_module(program),
        volume_module(program),
        composite_module(),
    ] {
        validate(&source);
    }
}

fn validate(source: &str) -> naga::Module {
    let module =
        naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("parse: {}", e.message()));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("validate: {e:?}"));
    module
}

#[test]
fn generated_variants_parse_and_validate() {
    for (sources, code_len) in [(0, 0), (1, 2), (2, 7), (4, 64)] {
        check(&program(sources, code_len));
    }
}

#[test]
fn shared_evaluator_is_a_composable_snippet() {
    for forbidden in ["@compute", "@vertex", "@fragment"] {
        assert!(!PROGRAM_EVAL_WGSL.contains(forbidden), "{forbidden}");
    }
    for binding in [
        "@binding(0) var<storage, read> trace_code",
        "@binding(1) var<storage, read> trace_data",
    ] {
        assert!(PROGRAM_EVAL_WGSL.contains(binding), "{binding}");
    }
    for name in [
        "trace_map",
        "trace_primitive",
        "trace_combine",
        "trace_eval",
        "trace_rest_point",
        "trace_instance",
        "trace_material",
    ] {
        assert!(PROGRAM_EVAL_WGSL.contains(name), "{name}");
    }
    let module = eval_module(&program(1, 2));
    assert!(
        module.contains("fn sdf_eval"),
        "the generated module has its entry point"
    );
}

#[test]
fn trace_and_composite_entry_points_and_layouts() {
    let compute = validate(&trace_module(&program(1, 2)));
    assert_eq!(compute.entry_points.len(), 1);
    assert_eq!(compute.entry_points[0].workgroup_size, [8, 8, 1]);
    let composite = validate(&composite_module());
    assert_eq!(composite.entry_points.len(), 2);
    assert_eq!(composite.entry_points[0].stage, naga::ShaderStage::Vertex);
    assert_eq!(composite.entry_points[1].stage, naga::ShaderStage::Fragment);
    for (name, size) in [
        ("TraceSource", 48),
        ("TraceInstance", 128),
        ("TraceBend", 32),
        ("TraceNode", 32),
        ("TraceMaterial", 32),
        ("TraceFrame", 512),
    ] {
        let ty = compute
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap()
            .1;
        assert!(
            matches!(ty.inner, naga::TypeInner::Struct { span, .. } if span == size),
            "{name}"
        );
    }
}

fn bindings(module: &naga::Module) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = module
        .global_variables
        .iter()
        .filter_map(|(_, global)| {
            global
                .binding
                .map(|binding| (binding.group, binding.binding))
        })
        .collect();
    out.sort_unstable();
    out
}

fn storage_format(module: &naga::Module, group: u32, binding: u32) -> Option<naga::StorageFormat> {
    let global = module
        .global_variables
        .iter()
        .map(|(_, global)| global)
        .find(|global| {
            global
                .binding
                .is_some_and(|b| b.group == group && b.binding == binding)
        })?;
    match module.types[global.ty].inner {
        naga::TypeInner::Image {
            class: naga::ImageClass::Storage { format, .. },
            ..
        } => Some(format),
        _ => None,
    }
}

#[test]
fn installed_shader_layouts_match_the_frame_passes() {
    assert_eq!((TRACE_NAME, TRACE_ORDER), ("trace.sdf", 55));
    assert_eq!((COMPOSITE_NAME, COMPOSITE_ORDER), ("trace.composite", 560));
    let compute = validate(&trace_module(&program(1, 2)));
    assert_eq!(
        bindings(&compute),
        (0..=10).map(|binding| (0, binding)).collect::<Vec<_>>()
    );
    assert_eq!(
        storage_format(&compute, 0, 9),
        Some(naga::StorageFormat::Rgba16Float)
    );
    assert_eq!(
        storage_format(&compute, 0, 10),
        Some(naga::StorageFormat::R32Float)
    );
    let composite = validate(&composite_module());
    assert_eq!(bindings(&composite), vec![(0, 0), (1, 0), (1, 1)]);
}

#[test]
fn volume_entry_points_and_copy_layout() {
    let module = validate(&volume_module(&program(2, 7)));
    let entries: Vec<_> = module
        .entry_points
        .iter()
        .map(|entry| (entry.name.as_str(), entry.stage, entry.workgroup_size))
        .collect();
    assert_eq!(
        entries,
        ["sdf_trace", "sdf_stamp", "sdf_mip", "sdf_coarse"].map(|name| (
            name,
            naga::ShaderStage::Compute,
            [8, 8, 1]
        ))
    );
    let expected: Vec<_> = (0..=10)
        .map(|binding| (0, binding))
        .chain((0..=4).map(|binding| (1, binding)))
        .collect();
    assert_eq!(bindings(&module), expected);
    assert_eq!(storage_format(&module, 1, 1), None);
    assert_eq!(
        storage_format(&module, 1, 4),
        Some(naga::StorageFormat::Rgba32Float)
    );
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = global.binding.filter(|b| b.group == 1) else {
            continue;
        };
        let ty = &module.types[global.ty].inner;
        match binding.binding {
            0 => assert!(matches!(ty, naga::TypeInner::Struct { span: 64, .. })),
            1 => assert!(matches!(
                ty,
                naga::TypeInner::Image {
                    dim: naga::ImageDimension::D3,
                    class: naga::ImageClass::Sampled { multi: false, .. },
                    ..
                }
            )),
            2 => assert!(matches!(ty, naga::TypeInner::Array { stride: 4, .. })),
            _ => {}
        }
    }
}

#[test]
fn volume_tracing_keeps_exact_refinement_and_depth_outputs() {
    let module = validate(&volume_module(&program(2, 7)));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    for (index, entry) in module.entry_points.iter().enumerate() {
        let used: Vec<_> = module
            .global_variables
            .iter()
            .filter(|(handle, _)| !info.get_entry_point(index)[*handle].is_empty())
            .filter_map(|(_, global)| global.name.as_deref())
            .collect();
        let required: &[&str] = match entry.name.as_str() {
            "sdf_trace" => &[
                "trace_code",
                "trace_data",
                "trace_bends",
                "trace_nodes",
                "trace_order",
                "trace_materials",
                "trace_field",
                "trace_coarse",
                "trace_color",
                "trace_depth",
            ],
            "sdf_stamp" => &[
                "trace_code",
                "trace_data",
                "trace_instances",
                "trace_voxels",
            ],
            "sdf_mip" => &["trace_field", "trace_voxels"],
            "sdf_coarse" => &["trace_frame", "trace_field", "trace_coarse_out"],
            _ => panic!("unexpected entry {}", entry.name),
        };
        for name in required {
            assert!(used.contains(name), "{} must use {name}", entry.name);
        }
        if entry.name == "sdf_trace" {
            assert!(!used.contains(&"trace_voxels"));
            assert!(!used.contains(&"trace_coarse_out"));
        }
    }
}

#[test]
fn frozen_volume_uniform_and_all_mip_copy_strides() {
    assert_eq!(
        (VOLUME_ABI_VERSION, VOLUME_MIP_LEVELS, VOLUME_MAX_AXIS),
        (1, 5, 1024)
    );
    assert_eq!(std::mem::size_of::<VolumeUniform>(), 64);
    assert_eq!(std::mem::align_of::<VolumeUniform>(), 16);
    assert_eq!(std::mem::offset_of!(VolumeUniform, origin), 0);
    assert_eq!(std::mem::offset_of!(VolumeUniform, shape), 16);
    assert_eq!(std::mem::offset_of!(VolumeUniform, encoding), 32);
    assert_eq!(std::mem::offset_of!(VolumeUniform, build), 48);
    let module = validate(&volume_module(&program(1, 1)));
    let ty = module
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some("TraceVolume"))
        .unwrap()
        .1;
    let naga::TypeInner::Struct { members, span } = &ty.inner else {
        panic!("uniform");
    };
    assert_eq!(*span, 64);
    assert_eq!(
        members.iter().map(|m| m.offset).collect::<Vec<_>>(),
        [0, 16, 32, 48]
    );
    for dimensions in [[16, 16, 16], [1008, 48, 16], [1024, 1024, 512]] {
        for mip in 0..VOLUME_MIP_LEVELS {
            let layout = volume_mip_layout(dimensions, mip).unwrap();
            let [w, h, d] = dimensions.map(|n| n >> mip);
            assert_eq!(layout.dimensions, [w, h, d]);
            assert_eq!(layout.bytes_per_row % 256, 0);
            assert!(layout.bytes_per_row >= w && layout.bytes_per_row - w < 256);
            assert_eq!(layout.rows_per_image, h);
            assert_eq!(
                layout.slice_stride,
                u64::from(layout.bytes_per_row) * u64::from(h)
            );
            assert_eq!(layout.buffer_size, layout.slice_stride * u64::from(d));
            assert_eq!(layout.workgroups, [w.div_ceil(32), h.div_ceil(8), d]);
            // Last valid byte and final packed word fit; a narrow mip never copies padding.
            let last_byte = u64::from(d - 1) * layout.slice_stride
                + u64::from(h - 1) * u64::from(layout.bytes_per_row)
                + u64::from(w - 1);
            assert!(last_byte < layout.buffer_size);
            assert!(w.div_ceil(4) * 4 <= layout.bytes_per_row);
        }
    }
    for dims in [[0, 16, 16], [1025, 16, 16], [31, 16, 16]] {
        assert!(volume_mip_layout(dims, 0).is_none());
    }
    assert!(volume_mip_layout([16; 3], 5).is_none());
}

fn encode(distance: f64, range: f64) -> i8 {
    ((distance / range).clamp(-1.0, 1.0) * 127.0).floor() as i8
}

fn decode(byte: i8, range: f64) -> f64 {
    (f64::from(byte) / 127.0).max(-1.0) * range
}

#[test]
fn snorm_encoding_rounding_saturation_and_gpu_byte_order() {
    for range in [0.01, 1.0, 1e6] {
        for step in -2048..=2048 {
            let distance = f64::from(step) * range / 1024.0;
            let byte = encode(distance, range);
            assert_ne!(byte, -128);
            let decoded = decode(byte, range);
            if distance >= -range {
                assert!(decoded <= distance + range * 1e-12);
                if distance <= range {
                    assert!(distance - decoded <= range / 127.0 + range * 1e-12);
                }
            }
            // Traversal never takes abs(): negative clipping cannot turn an interior into a skip.
            assert!(decoded.max(0.0) <= distance.max(0.0) + range * 1e-12);
        }
        assert_eq!(decode(-128, range), -range);
    }
    let bytes = [-127i8, -1, 0, 127];
    let packed = bytes
        .iter()
        .enumerate()
        .fold(0u32, |p, (lane, b)| p | (u32::from(*b as u8) << (8 * lane)));
    assert_eq!(packed.to_le_bytes(), [129, 255, 0, 127]);
    for lane in 0..4 {
        assert_eq!(((packed >> (8 * lane)) & 255) as u8 as i8, bytes[lane]);
    }
}

fn length(p: [f64; 3]) -> f64 {
    p.iter().map(|v| v * v).sum::<f64>().sqrt()
}

// Independent mathematical field: a sphere and an off-grid, sub-voxel thin plate.
fn reference_field(p: [f64; 3]) -> f64 {
    let sphere = length([p[0] + 0.3, p[1] - 0.1, p[2]]) - 0.23;
    let q = [
        (p[0] - 0.04).abs() - 0.003,
        p[1].abs() - 0.4,
        p[2].abs() - 0.8,
    ];
    let plate =
        length(q.map(|v| v.max(0.0))) + q.into_iter().fold(f64::NEG_INFINITY, f64::max).min(0.0);
    sphere.min(plate)
}

#[test]
fn five_mips_bound_entire_cells_including_thin_surfaces_and_clipping() {
    let n = 16usize;
    let h = 2.0 / n as f64;
    for range in [0.01, 0.2, 4.0] {
        let mut cells = Vec::new();
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let p = [x, y, z].map(|i| -1.0 + (i as f64 + 0.5) * h);
                    cells.push(encode(
                        reference_field(p) / 3.0 - 3f64.sqrt() * h / 2.0 - 1e-6,
                        range,
                    ));
                }
            }
        }
        let mut width = n;
        for mip in 0..5 {
            let spacing = h * f64::from(1 << mip);
            for z in 0..width {
                for y in 0..width {
                    for x in 0..width {
                        let clearance = (decode(cells[(z * width + y) * width + x], range)
                            - range / 127.0
                            - 1e-6)
                            .max(0.0);
                        // Corners, center and off-center points exercise the full footprint, not only samples.
                        for fraction in [[0.5; 3], [0.13, 0.89, 0.37]]
                            .into_iter()
                            .chain((0..8).map(|c| [0, 1, 2].map(|a| f64::from((c >> a) & 1))))
                        {
                            let p = [x, y, z].map(|i| i as f64);
                            let p = [0, 1, 2].map(|a| -1.0 + (p[a] + fraction[a]) * spacing);
                            assert!(
                                clearance <= reference_field(p).max(0.0) + 1e-12,
                                "mip {mip}, {p:?}"
                            );
                        }
                    }
                }
            }
            if mip == 4 {
                break;
            }
            let next = width / 2;
            let mut reduced = vec![127; next * next * next];
            for z in 0..next {
                for y in 0..next {
                    for x in 0..next {
                        reduced[(z * next + y) * next + x] = (0..8)
                            .map(|c| {
                                let [cx, cy, cz] =
                                    [x * 2 + (c & 1), y * 2 + ((c >> 1) & 1), z * 2 + (c >> 2)];
                                cells[(cz * width + cy) * width + cx]
                            })
                            .min()
                            .unwrap();
                    }
                }
            }
            cells = reduced;
            width = next;
        }
        assert!(
            cells[0] <= 0,
            "the root must retain both sub-voxel surfaces"
        );
    }
    let children = [-1i8, 127, 127, 127, 127, 127, 127, 127];
    assert!(children.into_iter().map(f64::from).sum::<f64>() / 8.0 > 0.0);
    assert!(
        children.into_iter().min().unwrap() < 0,
        "averaging would skip this possible surface"
    );
}

#[test]
fn coarse_cones_cover_full_resolution_edge_tiles_and_off_axis_rays() {
    let target = [2057u32, 1235];
    let tile = VOLUME_COARSE_TILE;
    let coarse = target.map(|d| d.div_ceil(tile));
    assert_eq!(coarse, [258, 155]);
    let subtract = |a: [f64; 3], b: [f64; 3]| [0, 1, 2].map(|i| a[i] - b[i]);
    let normalize = |v: [f64; 3]| v.map(|x| x / length(v));
    for orthographic in [false, true] {
        let ray = |uv: [f64; 2]| {
            let xy = [uv[0] * 5.0 - 1.7, 1.3 - uv[1] * 3.0];
            if orthographic {
                ([xy[0] * 10.0, xy[1] * 10.0, 0.1], [0.0, 0.0, 99.9])
            } else {
                (
                    [xy[0] * 0.1, xy[1] * 0.1, 0.1],
                    [xy[0] * 99.9, xy[1] * 99.9, 99.9],
                )
            }
        };
        for cell in [[0, 0], [128, 77], [257, 154]] {
            let lo = [0, 1].map(|a| f64::from(cell[a] * tile) / f64::from(target[a]));
            let hi = [0, 1]
                .map(|a| f64::from(((cell[a] + 1) * tile).min(target[a])) / f64::from(target[a]));
            let uv = [0, 1].map(|a| (lo[a] + hi[a]) * 0.5);
            let (o, delta) = ray(uv);
            let dir = normalize(delta);
            let mut origin_radius = 0f64;
            let mut delta_radius = 0f64;
            let mut projection = length(delta);
            for corner in 0..4 {
                let corner_uv = [0, 1].map(|a| if corner & (1 << a) == 0 { lo[a] } else { hi[a] });
                let (co, cd) = ray(corner_uv);
                origin_radius = origin_radius.max(length(subtract(co, o)));
                delta_radius = delta_radius.max(length(subtract(cd, delta)));
                projection = projection.min((0..3).map(|a| cd[a] * dir[a]).sum::<f64>());
            }
            let direction_radius = (2.0 * delta_radius / projection).min(2.0);
            for y in 0..=16 {
                for x in 0..=16 {
                    let uv = [
                        lo[0] + (hi[0] - lo[0]) * f64::from(x) / 16.0,
                        lo[1] + (hi[1] - lo[1]) * f64::from(y) / 16.0,
                    ];
                    let (po, pd) = ray(uv);
                    let pd = normalize(pd);
                    for t in [0.0, 0.01, 1.0, 100.0, 10000.0] {
                        let offset = [0, 1, 2].map(|a| po[a] - o[a] + t * (pd[a] - dir[a]));
                        assert!(length(offset) <= origin_radius + direction_radius * t + 1e-10);
                    }
                }
            }
        }
    }
    // Every full pixel, including a partial tile on both axes, maps to a valid coarse texel.
    for y in 0..target[1] {
        for x in 0..target[0] {
            assert!(x / tile < coarse[0] && y / tile < coarse[1]);
        }
    }
}

#[test]
fn volume_guards_stale_results_and_preserves_full_target_coordinates() {
    let shader = volume_module(&program(1, 1));
    // Guard the ABI's safety-critical wiring as well as validating every entry's data flow above.
    for required in [
        "trace_volume.build.z == trace_volume.shape.w",
        "if (!trace_volume_ready()) { return 0.0; }",
        "coarse.y == 1.0 && coarse.z == f32(trace_volume.shape.w)",
        "if (cone.x < 0.0) { return 0.0; }",
        "var start = 0.0;",
        "let size = textureDimensions(trace_color);",
        "let pixel = vec2<i32>(id.xy);",
        "trace_candidate(index, origin, direction, start, nearest)",
        "trace_value(hit, p + x).d - trace_value(hit, p - x).d",
        "let clip = trace_frame.view_proj * vec4<f32>(p, 1.0);",
        "if (cell.x >= size.x) { continue; }",
    ] {
        assert!(shader.contains(required), "missing invariant: {required}");
    }
    assert!(!shader.contains("MAX_TARGET"));
    assert!(composite_module().contains("if (color.a == 0.0) { discard; }"));
}

#[test]
fn scheduled_mips_and_uniforms_follow_the_frozen_abi() {
    let layout = VolumeLayout::enclose([-4.0, -2.0, 0.5], [12.0, 6.0, 4.5]).expect("bounds");
    assert!(
        layout
            .base
            .iter()
            .all(|&axis| axis > 0 && axis <= VOLUME_MAX_AXIS && axis % 16 == 0)
    );
    assert!(layout.spacing > 0.0 && layout.range > 0.0 && layout.roundoff() > 0.0);
    for level in 0..VOLUME_MIP_LEVELS {
        let mip = layout.mip(level);
        let abi = volume_mip_layout(layout.base, level).expect("frozen ABI layout");
        assert_eq!(abi.dimensions, mip.size, "mip {level} dimensions");
        assert_eq!(
            abi.bytes_per_row, mip.bytes_per_row,
            "mip {level} row pitch"
        );
        assert_eq!(abi.rows_per_image, mip.rows_per_image);
        assert_eq!(abi.workgroups, mip.dispatch);
        assert_eq!(
            abi.slice_stride,
            u64::from(mip.bytes_per_row) * u64::from(mip.size[1])
        );
        let uniform = layout.uniform_revision(level, 7);
        assert_eq!(uniform.len(), 64);
        assert_eq!(
            f32::from_le_bytes(uniform[0..4].try_into().unwrap()),
            layout.origin[0]
        );
        assert_eq!(
            f32::from_le_bytes(uniform[12..16].try_into().unwrap()),
            layout.spacing
        );
        assert_eq!(
            u32::from_le_bytes(uniform[16..20].try_into().unwrap()),
            layout.base[0]
        );
        assert_eq!(u32::from_le_bytes(uniform[28..32].try_into().unwrap()), 7);
        assert_eq!(
            f32::from_le_bytes(uniform[32..36].try_into().unwrap()),
            layout.range
        );
        assert_eq!(
            f32::from_le_bytes(uniform[36..40].try_into().unwrap()),
            layout.roundoff()
        );
        assert_eq!(
            u32::from_le_bytes(uniform[48..52].try_into().unwrap()),
            level
        );
        assert_eq!(
            u32::from_le_bytes(uniform[52..56].try_into().unwrap()),
            mip.pitch_words
        );
        assert_eq!(u32::from_le_bytes(uniform[56..60].try_into().unwrap()), 7);
    }
    let base = volume_mip_layout(layout.base, 0).expect("base copy");
    assert!(layout.buffer_bytes() >= base.buffer_size);
}

#[test]
fn empty_scene_dummy_volume_and_full_target_are_independent() {
    let layout = VolumeLayout::dummy();
    assert!(
        layout
            .base
            .iter()
            .all(|&axis| (16..=VOLUME_MAX_AXIS).contains(&axis) && axis % 16 == 0)
    );
    for level in 0..VOLUME_MIP_LEVELS {
        assert!(volume_mip_layout(layout.base, level).is_some());
    }
    // The volume-axis limit never caps the image: targets keep every render pixel.
    assert_eq!(target_size(4096, 2160), (4096, 2160));
    assert_eq!(target_size(1, 1), (1, 1));
    let coarse = |(w, h): (u32, u32)| {
        (
            w.div_ceil(VOLUME_COARSE_TILE),
            h.div_ceil(VOLUME_COARSE_TILE),
        )
    };
    assert_eq!(coarse(target_size(1920, 1080)), (240, 135));
    assert_eq!(coarse(target_size(2057, 1235)), (258, 155));
}
