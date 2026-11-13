//! The S32 cull-control module and the `pass_view.wgsl` scene twins: naga validation of the
//! composed module, its entry points, group-2 bindings and per-stage storage-buffer count, the
//! layouter sizes of every shared record and the `depth_calls` text helper the later modules
//! pin their depth helpers with.

#![cfg(not(target_arch = "wasm32"))]

use raa_cluster::weld::{decode_octahedral, encode_octahedral};
use raa_core::testing::{skip, software_gpu};
use raa_core::wgpu;
use raa_math::Vec3;
use raa_shader::{Composer, DEPTH_WGSL};
use raa_vg::culling::{PASS_VIEW_WGSL, cull_control_module_wgsl, cull_module_wgsl};
use raa_vg::hzb::{Hzb, hzb_down_module_wgsl, hzb_seed_module_wgsl, hzb_size};
use wgpu::naga;

/// Every shared record gets one storage buffer so the naga front-end emits its type and the
/// layouter can size it.
const PROBE: &str = r#"
@group(0) @binding(0) var<storage, read_write> probe_view: PassView;
@group(0) @binding(1) var<storage, read_write> probe_instance: VgInstance;
@group(0) @binding(2) var<storage, read_write> probe_node: VgNode;
@group(0) @binding(3) var<storage, read_write> probe_cluster: VgCluster;
@group(0) @binding(4) var<storage, read_write> probe_vertex: PackedVertex;
@group(0) @binding(5) var<storage, read_write> probe_visible: VisibleCluster;
@group(0) @binding(6) var<storage, read_write> probe_queue: QueueEntry;
@group(0) @binding(7) var<storage, read_write> probe_state: CullState;
@group(0) @binding(8) var<storage, read_write> probe_read: CullStateRead;

@compute @workgroup_size(1)
fn probe_main() {
    probe_view.stamp = probe_instance.ids.x;
    probe_instance.ranges.x = probe_node.first;
    probe_node.count = probe_cluster.counts;
    probe_cluster.vertex_offset = probe_vertex.a;
    probe_vertex.a = probe_visible.instance;
    probe_visible.cluster = probe_queue.item;
    probe_queue.instance = probe_read.dropped;
    probe_read.write_head = atomicLoad(&probe_state.write_head);
}
"#;

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

fn type_handle(module: &naga::Module, name: &str) -> naga::Handle<naga::Type> {
    module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some(name))
        .map(|(handle, _)| handle)
        .unwrap_or_else(|| panic!("{name} is declared"))
}

fn type_size(module: &naga::Module, name: &str) -> u32 {
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).expect("the types lay out");
    layouter[type_handle(module, name)].size
}

fn storage_buffers_used(module: &naga::Module, entry: &naga::EntryPoint) -> usize {
    let mut used = std::collections::BTreeSet::new();
    for (_, expression) in entry.function.expressions.iter() {
        if let naga::Expression::GlobalVariable(handle) = *expression
            && matches!(
                module.global_variables[handle].space,
                naga::AddressSpace::Storage { .. }
            )
        {
            used.insert(handle);
        }
    }
    used.len()
}

#[test]
fn cull_control_module_validates_with_three_entry_points() {
    let wgsl = cull_control_module_wgsl();
    assert_eq!(Composer::validate(&wgsl), Ok(()));
    let module = parse(&wgsl);
    let mut entry_points: Vec<&str> = module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    entry_points.sort_unstable();
    assert_eq!(entry_points, ["advance", "begin_second_phase", "finish"]);
}

#[test]
fn cull_control_binds_state_and_args_at_group_two() {
    let module = parse(&cull_control_module_wgsl());
    let mut bindings: Vec<(u32, u32)> = module
        .global_variables
        .iter()
        .map(|(_, global)| {
            let binding = global.binding.expect("every global is bound");
            (binding.group, binding.binding)
        })
        .collect();
    bindings.sort_unstable();
    assert_eq!(bindings, [(2, 0), (2, 1)]);
    for (_, global) in module.global_variables.iter() {
        assert!(matches!(
            global.space,
            naga::AddressSpace::Storage { access }
                if access.contains(naga::StorageAccess::LOAD | naga::StorageAccess::STORE)
        ));
    }
    for entry in &module.entry_points {
        assert!(
            storage_buffers_used(&module, entry) <= 8,
            "entry point {} uses more than eight storage buffers",
            entry.name
        );
    }
}

#[test]
fn scene_twins_match_the_s32_and_s29_sizes() {
    let module = parse(&format!("{PASS_VIEW_WGSL}\n{PROBE}"));
    for (name, bytes) in [
        ("PassView", 432),
        ("VgInstance", 192),
        ("VgNode", 48),
        ("VgCluster", 48),
        ("CullState", 48),
        ("CullStateRead", 48),
        ("PackedVertex", 16),
        ("VisibleCluster", 8),
        ("QueueEntry", 8),
    ] {
        assert_eq!(type_size(&module, name), bytes, "{name}");
    }
}

#[test]
fn cull_control_args_have_the_s32_offsets() {
    let module = parse(&cull_control_module_wgsl());
    let naga::TypeInner::Struct { members, span } =
        &module.types[type_handle(&module, "Args")].inner
    else {
        panic!("Args is a struct");
    };
    assert_eq!(*span, 56);
    let offsets: Vec<(&str, u32)> = members
        .iter()
        .map(|member| (member.name.as_deref().unwrap_or(""), member.offset))
        .collect();
    assert_eq!(
        offsets,
        [
            ("queue_x", 0),
            ("queue_y", 4),
            ("queue_z", 8),
            ("software_x", 12),
            ("software_y", 16),
            ("software_z", 20),
            ("draw_vertex_count", 24),
            ("draw_instance_count", 28),
            ("draw_first_vertex", 32),
            ("draw_first_instance", 36),
            ("merge_vertex_count", 40),
            ("merge_instance_count", 44),
            ("merge_first_vertex", 48),
            ("merge_first_instance", 52),
        ]
    );
}

#[test]
fn depth_calls_lists_each_helper_once() {
    assert_eq!(
        depth_calls(&cull_control_module_wgsl()),
        [
            "raa_depth_nearer",
            "raa_depth_nearest",
            "raa_depth_farthest",
            "raa_depth_is_far",
            "raa_depth_key",
            "raa_depth_from_key",
        ]
    );
    assert_eq!(
        depth_calls("fn probe() { return raa_depth_key(0.5) + raa_depth_key(0.25); }"),
        ["raa_depth_key"]
    );
}

#[test]
fn cull_module_validates_with_seed_and_process() {
    let wgsl = cull_module_wgsl();
    assert_eq!(Composer::validate(&wgsl), Ok(()));
    let module = parse(&wgsl);
    let mut entry_points: Vec<&str> = module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    entry_points.sort_unstable();
    assert_eq!(entry_points, ["process", "seed"]);
    for entry in &module.entry_points {
        assert!(
            storage_buffers_used(&module, entry) <= 8,
            "entry point {} uses more than eight storage buffers",
            entry.name
        );
    }
}

#[test]
fn cull_scene_and_work_bindings_match_the_s32_groups() {
    let module = parse(&cull_module_wgsl());
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
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8)
        ]
    );
    for (_, global) in module.global_variables.iter() {
        let binding = global.binding.expect("every global is bound");
        match (binding.group, binding.binding) {
            (2, 0) => assert!(matches!(global.space, naga::AddressSpace::Uniform)),
            (2, 1..=3) => assert!(matches!(
                global.space,
                naga::AddressSpace::Storage { access }
                    if access == naga::StorageAccess::LOAD
            )),
            (2, 4) => assert!(matches!(global.space, naga::AddressSpace::Handle)),
            _ => assert!(matches!(
                global.space,
                naga::AddressSpace::Storage { access }
                    if access.contains(naga::StorageAccess::LOAD | naga::StorageAccess::STORE)
            )),
        }
    }
}

/// The helper declarations of `DEPTH_WGSL` name every depth helper; only the calls the cull
/// body makes are pinned here. RAA's cull never uses the reverse-order key helpers.
#[test]
fn cull_calls_only_the_allowed_depth_helpers() {
    let wgsl = cull_module_wgsl();
    let body = wgsl
        .strip_prefix(DEPTH_WGSL)
        .expect("the cull module starts with the depth helper");
    let calls = depth_calls(body);
    let allowed = [
        "raa_depth_nearer",
        "raa_depth_nearest",
        "raa_depth_farthest",
        "raa_depth_is_far",
    ];
    for call in &calls {
        assert!(
            allowed.contains(call),
            "{call} is not one of the standard-Z helpers the cull may call"
        );
    }
    assert!(
        calls.contains(&"raa_depth_nearer"),
        "the occlusion test compares through raa_depth_nearer"
    );
    for forbidden in ["page_requests", "texture_info"] {
        assert!(
            !wgsl.contains(forbidden),
            "the RAA cull never pages cluster data: {forbidden} is removed"
        );
    }
}

/// The body of `fn name(..)` in `src`, up to its closing brace at column 0.
fn function_body<'a>(src: &'a str, name: &str) -> &'a str {
    let marker = format!("fn {name}(");
    let start = src
        .find(&marker)
        .unwrap_or_else(|| panic!("{name} is declared"));
    let rest = &src[start..];
    let end = rest.find("\n}").expect("the function closes") + 2;
    &rest[..end]
}

/// `pack_vertex` (crates/raa-cluster/src/dag.rs:801-828) writes the eight u16 halves
/// `[grid x, grid y, grid z, oct x, oct y, u, v, 0]`, so its little-endian u32 words are
/// `a = grid x | grid y << 16`, `b = grid z | oct x << 16`, `c = oct y | u << 16`, `d = v`.
/// Every decode helper must read exactly those words and halves.
#[test]
fn decode_helpers_read_the_pack_vertex_halves() {
    let position = function_body(PASS_VIEW_WGSL, "decode_position");
    assert!(
        position.contains("v.a & 0xffffu"),
        "grid x is the low half of a"
    );
    assert!(
        position.contains("v.a >> 16u"),
        "grid y is the high half of a"
    );
    assert!(
        position.contains("v.b & 0xffffu"),
        "grid z is the low half of b"
    );

    let normal = function_body(PASS_VIEW_WGSL, "decode_normal");
    assert!(
        normal.contains("i32(v.b) >> 16u"),
        "octahedral x is the sign-extended high half of b"
    );
    assert!(
        !normal.contains("v.b << 16u"),
        "octahedral x must not be taken from the low half of b, which is grid z"
    );
    assert!(
        normal.contains("i32(v.c << 16u) >> 16u"),
        "octahedral y is the sign-extended low half of c"
    );

    let uv = function_body(PASS_VIEW_WGSL, "decode_uv");
    assert!(uv.contains("v.c >> 16u"), "u is the high half of c");
    assert!(uv.contains("v.d & 0xffffu"), "v is the low half of d");
}

/// `pack_vertex`'s halves: `grid x, grid y, grid z, oct x, oct y, u, v, 0` as u16 words.
fn packed_vertex(grid: [u16; 3], normal: Vec3, uv: [u16; 2]) -> [u8; 16] {
    let oct = encode_octahedral(normal);
    let data = [
        grid[0],
        grid[1],
        grid[2],
        oct[0] as u16,
        oct[1] as u16,
        uv[0],
        uv[1],
        0,
    ];
    let mut bytes = [0u8; 16];
    for (index, word) in data.iter().enumerate() {
        bytes[index * 2..index * 2 + 2].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

const DECODE_PROBE: &str = r#"
@group(0) @binding(0) var<storage, read> probe_instance: VgInstance;
@group(0) @binding(1) var<storage, read> probe_vertices: array<PackedVertex>;
@group(0) @binding(2) var<storage, read_write> probe_out: array<vec4<f32>>;

@compute @workgroup_size(1)
fn probe_decode(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = probe_vertices[id.x];
    let base = id.x * 3u;
    probe_out[base] = vec4<f32>(decode_position(probe_instance, v), 1.0);
    probe_out[base + 1u] = vec4<f32>(decode_normal(v), 0.0);
    let uv = decode_uv(probe_instance, v);
    probe_out[base + 2u] = vec4<f32>(uv, 0.0, 0.0);
}
"#;

/// The three decode helpers on an adapter: vertices packed with `pack_vertex`'s field order
/// and normals encoded by the crate's `encode_octahedral` round trip through
/// `PASS_VIEW_WGSL`. SKIP without a software adapter.
#[test]
fn decode_helpers_round_trip_the_packed_vertex() {
    let test = "decode_helpers_round_trip_the_packed_vertex";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };

    let quant = [1.5f32, -2.0, 3.25, 0.5];
    let uv_quant = [10.0f32, 20.0, 0.25, 0.5];
    let grid = [
        [100u16, 200, 300],
        [4, 5, 6],
        [65535, 0, 1234],
        [77, 88, 99],
    ];
    let uv = [[1234u16, 5678], [0, 65535], [7, 9], [4321, 8765]];
    // The first normal keeps grid z (300) far from its octahedral x, so reading the low half
    // of b instead of the high half cannot pass.
    let normals = [
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-0.3, 0.5, -0.8).normalize_or_zero(),
        Vec3::new(0.2, -0.6, 0.7).normalize_or_zero(),
    ];

    let mut vertices = Vec::new();
    for index in 0..normals.len() {
        vertices.extend_from_slice(&packed_vertex(grid[index], normals[index], uv[index]));
    }
    let mut instance = [0u8; 192];
    for (slot, value) in quant.iter().enumerate() {
        instance[112 + slot * 4..116 + slot * 4].copy_from_slice(&value.to_le_bytes());
    }
    for (slot, value) in uv_quant.iter().enumerate() {
        instance[128 + slot * 4..132 + slot * 4].copy_from_slice(&value.to_le_bytes());
    }

    let shader = format!("{PASS_VIEW_WGSL}\n{DECODE_PROBE}");
    assert_eq!(Composer::validate(&shader), Ok(()));
    let device = gpu.device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa-vg.cull_variants.decode"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("raa-vg.cull_variants.decode"),
        layout: None,
        module: &module,
        entry_point: Some("probe_decode"),
        compilation_options: Default::default(),
        cache: None,
    });

    let storage = |label: &str, bytes: &[u8], read_back: bool| {
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if read_back {
            usage |= wgpu::BufferUsages::COPY_SRC;
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage,
            mapped_at_creation: false,
        });
        gpu.queue().write_buffer(&buffer, 0, bytes);
        buffer
    };
    let instance_buffer = storage("raa-vg.cull_variants.instance", &instance, false);
    let vertex_buffer = storage("raa-vg.cull_variants.vertices", &vertices, false);
    let out_bytes = normals.len() * 3 * 16;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa-vg.cull_variants.out"),
        size: out_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa-vg.cull_variants.readback"),
        size: out_bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-vg.cull_variants.decode"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vertex_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("raa-vg.cull_variants.decode"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa-vg.cull_variants.decode"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(normals.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, out_bytes as u64);
    gpu.queue().submit([encoder.finish()]);

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
    let decoded: Vec<[f32; 4]> = mapped
        .chunks_exact(16)
        .map(|word| {
            std::array::from_fn(|slot| {
                f32::from_le_bytes(word[slot * 4..slot * 4 + 4].try_into().expect("four bytes"))
            })
        })
        .collect();
    drop(mapped);
    readback.unmap();

    for (index, normal) in normals.iter().enumerate() {
        let expected_position = Vec3::new(
            quant[0] + f32::from(grid[index][0]) * quant[3],
            quant[1] + f32::from(grid[index][1]) * quant[3],
            quant[2] + f32::from(grid[index][2]) * quant[3],
        );
        let expected_uv = [
            uv_quant[0] + f32::from(uv[index][0]) * uv_quant[2],
            uv_quant[1] + f32::from(uv[index][1]) * uv_quant[3],
        ];
        let expected_normal = decode_octahedral(encode_octahedral(*normal));
        let position = decoded[index * 3];
        let normal_out = decoded[index * 3 + 1];
        let uv_out = decoded[index * 3 + 2];
        for (axis, expected) in [
            expected_position.x,
            expected_position.y,
            expected_position.z,
        ]
        .iter()
        .enumerate()
        {
            assert!(
                (position[axis] - expected).abs() <= 1.0e-4,
                "position {index} axis {axis}: {} != {expected}",
                position[axis]
            );
        }
        for (axis, expected) in [expected_normal.x, expected_normal.y, expected_normal.z]
            .iter()
            .enumerate()
        {
            assert!(
                (normal_out[axis] - expected).abs() <= 1.0e-3,
                "normal {index} axis {axis}: {} != {expected}",
                normal_out[axis]
            );
        }
        assert!(
            (uv_out[0] - expected_uv[0]).abs() <= 1.0e-4
                && (uv_out[1] - expected_uv[1]).abs() <= 1.0e-4,
            "uv {index}: ({}, {}) != {expected_uv:?}",
            uv_out[0],
            uv_out[1]
        );
    }
}

/// Every bound global of `module` as `(group, binding)`, sorted.
fn bindings(module: &naga::Module) -> Vec<(u32, u32)> {
    let mut bindings: Vec<(u32, u32)> = module
        .global_variables
        .iter()
        .map(|(_, global)| {
            let binding = global.binding.expect("every global is bound");
            (binding.group, binding.binding)
        })
        .collect();
    bindings.sort_unstable();
    bindings
}

/// The hzb seed module validates with both seed entry points and its four declarations at
/// group 0; the downsample module validates with `main` and its two group-0 bindings.
#[test]
fn hzb_modules_validate_with_their_group_zero_bindings() {
    let seed = hzb_seed_module_wgsl();
    assert_eq!(Composer::validate(&seed), Ok(()));
    let module = parse(&seed);
    let mut entry_points: Vec<&str> = module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    entry_points.sort_unstable();
    assert_eq!(entry_points, ["seed_from_depth", "seed_from_keys"]);
    assert_eq!(bindings(&module), [(0, 0), (0, 1), (0, 2), (0, 3)]);
    for (_, global) in module.global_variables.iter() {
        let binding = global.binding.expect("every global is bound");
        match binding.binding {
            0 => assert!(matches!(global.space, naga::AddressSpace::Uniform)),
            1 => assert!(matches!(
                global.space,
                naga::AddressSpace::Storage { access } if access == naga::StorageAccess::LOAD
            )),
            _ => assert!(matches!(global.space, naga::AddressSpace::Handle)),
        }
    }

    let down = hzb_down_module_wgsl();
    assert_eq!(Composer::validate(&down), Ok(()));
    let module = parse(&down);
    let mut entry_points: Vec<&str> = module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    entry_points.sort_unstable();
    assert_eq!(entry_points, ["main"]);
    assert_eq!(bindings(&module), [(0, 0), (0, 1)]);
    for (_, global) in module.global_variables.iter() {
        assert!(matches!(global.space, naga::AddressSpace::Handle));
    }
}

/// Both hzb bodies reduce depth only through `raa_depth_farthest`, optionally converting a
/// visibility key with `raa_depth_from_key`; the nearest start makes empty keys far.
#[test]
fn hzb_reduces_the_farthest_depth_through_the_allowed_helpers() {
    let allowed = [
        "raa_depth_farthest",
        "raa_depth_from_key",
        "raa_depth_is_far",
    ];
    for wgsl in [hzb_seed_module_wgsl(), hzb_down_module_wgsl()] {
        let body = wgsl
            .strip_prefix(DEPTH_WGSL)
            .expect("every hzb module starts with the depth helper");
        let calls = depth_calls(body);
        for call in &calls {
            assert!(
                allowed.contains(call),
                "{call} is not one of the depth helpers the hzb may call"
            );
        }
        assert!(
            calls.contains(&"raa_depth_farthest"),
            "the hzb reduces toward the farthest depth"
        );
    }
}

/// The pyramid rounds each viewport axis down to its previous power of two.
#[test]
fn hzb_size_is_the_previous_power_of_two_of_each_axis() {
    assert_eq!(hzb_size((1, 1)), (1, 1));
    assert_eq!(hzb_size((1023, 1023)), (512, 512));
    assert_eq!(hzb_size((1280, 720)), (1024, 512));
}

/// The pyramid allocates its texture and both pipelines without a validation error.
#[test]
fn hzb_creates_without_validation_error() {
    let test = "hzb_creates_without_validation_error";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let hzb = Hzb::new(&gpu, (1280, 720));
    let _ = gpu.device().poll(wgpu::PollType::wait_indefinitely());
    let error = raa_core::testing::block_on(errors.pop());
    assert!(error.is_none(), "hzb validation: {error:?}");
    assert_eq!(hzb.size(), (1024, 512));
    let expected: u64 = 16
        + (0..11)
            .map(|mip| u64::from((1024u32 >> mip).max(1)) * u64::from((512u32 >> mip).max(1)) * 4)
            .sum::<u64>();
    assert_eq!(hzb.allocated_bytes(), expected);
}

/// The control module's queue dispatch size is the `process` workgroup size, so the queue
/// level never launches twice the workgroups the traversal needs.
#[test]
fn process_group_size_matches_the_process_workgroup() {
    let cull_wgsl = cull_module_wgsl();
    assert_eq!(Composer::validate(&cull_wgsl), Ok(()));
    let cull = parse(&cull_wgsl);
    let process = cull
        .entry_points
        .iter()
        .find(|entry| entry.name == "process")
        .expect("the cull module has a process entry point");
    assert_eq!(process.stage, naga::ShaderStage::Compute);

    let control = parse(&cull_control_module_wgsl());
    let (_, constant) = control
        .constants
        .iter()
        .find(|(_, constant)| constant.name.as_deref() == Some("PROCESS_GROUP_SIZE"))
        .expect("the control module declares PROCESS_GROUP_SIZE");
    let value = match &control.global_expressions[constant.init] {
        naga::Expression::Literal(naga::Literal::U32(value)) => *value,
        other => panic!("PROCESS_GROUP_SIZE is not a u32 literal: {other:?}"),
    };

    assert_eq!(
        process.workgroup_size,
        [value, 1, 1],
        "PROCESS_GROUP_SIZE must equal the process workgroup size"
    );
}

/// A 1x1 R32Float occlusion pyramid holding `depth` (standard-Z: 0.0 nearest, 1.0 far).
fn flat_pyramid(gpu: &raa_core::GpuContext, depth: f32) -> wgpu::TextureView {
    let extent = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("raa-vg cull variants pyramid"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue().write_texture(
        texture.as_image_copy(),
        &depth.to_le_bytes(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        extent,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Copies `size` bytes of `source` into a mapped staging buffer and returns them.
fn read_buffer(gpu: &raa_core::GpuContext, source: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let device = gpu.device();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa-vg cull variants readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    gpu.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    rx.recv().expect("map callback").expect("map");
    let bytes = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    bytes
}

/// Two-phase occlusion on the software adapter (SKIP line without one). Five instances of one
/// baked fan (1 m radius) stand at x = -2..2 m, z = 0, seen from (0, 0, 6) looking down -Z
/// (near 0.1, far 16): the camera and the instances lie inside a 16 m box. The stale pyramid
/// of the first phase holds depth 0.0 (the nearest possible surface, as if last frame's depth
/// hid everything), so phase 0 rejects every instance into the retry list; this frame's pyramid
/// holds 1.0 (far: nothing drawn yet hides them), so phase 1 recovers all five. No raster or
/// depth target is involved: the cull state and the visible list are read back directly.
#[test]
fn second_phase_recovers_instances_the_stale_pyramid_rejected() {
    use raa_cluster::{BuildOptions, ClusterStore, OwnedClusterInput, bake};
    use raa_core::MeshId;
    use raa_core::testing::block_on;
    use raa_math::Mat4;
    use raa_vg::culling::{CullPhase, CullPipelines, CullWork, MAIN_FIRST, MAIN_SECOND};
    use raa_vg::{
        INSTANCE_CASTS_SHADOW, MeshKeys, PassView, VgInstanceRecord, VgScene, ViewSlots,
        hierarchy_word,
    };

    let test = "second_phase_recovers_instances_the_stale_pyramid_rejected";
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
        mesh.quant_step,
    ];
    let uv_quant = [
        mesh.uv_origin[0],
        mesh.uv_origin[1],
        mesh.uv_step[0],
        mesh.uv_step[1],
    ];
    let bounds = [
        mesh.bounds_center[0],
        mesh.bounds_center[1],
        mesh.bounds_center[2],
        mesh.bounds_radius,
    ];

    let mut store = ClusterStore::new();
    store.attach(1, vec![mesh]);
    let mut keys = MeshKeys::default();
    keys.set(1, &[MeshId(0)]);
    let mut scene = VgScene::new();
    assert!(scene.upload_pool(&gpu, &store, &keys), "the pool uploads");
    let range = scene.section_range(1, 0).expect("the section is packed");
    let (_, depth, root) = scene.section_header(1, 0).expect("the section header");
    const COUNT: u32 = 5;
    let records: Vec<VgInstanceRecord> = (0..COUNT)
        .map(|i| VgInstanceRecord {
            model: Mat4::from_translation(Vec3::new(i as f32 - 2.0, 0.0, 0.0)),
            previous_rows: [[0.0; 4]; 3],
            quant,
            uv_quant,
            ranges: [
                range.node_base,
                range.cluster_base,
                range.vertex_base,
                range.triangle_base,
            ],
            ids: [i, 0, INSTANCE_CASTS_SHADOW, hierarchy_word(depth, root)],
            bounds,
        })
        .collect();
    scene.upload_instances(&gpu, &records);
    assert_eq!(scene.instance_count(), COUNT);

    let eye = Vec3::new(0.0, 0.0, 6.0);
    let proj = Mat4::perspective_rh_zo(60f32.to_radians(), 1.0, 0.1, 16.0);
    let view_proj = proj * Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
    let mut first = PassView::perspective(
        view_proj,
        view_proj,
        eye,
        [64.0, 64.0],
        [proj.cols[0][0], proj.cols[1][1]],
        0.1,
        1.0,
    );
    first.occlusion_enabled = 1;
    first.hzb_size = [1.0, 1.0];
    let mut second = first;
    second.phase = 1;
    let slots = ViewSlots::new(&gpu);
    slots.write(&gpu, MAIN_FIRST, &first);
    slots.write(&gpu, MAIN_SECOND, &second);

    let stale = flat_pyramid(&gpu, 0.0);
    let current = flat_pyramid(&gpu, 1.0);
    let pipelines = CullPipelines::new(&gpu);
    let capacity = 1024u64;
    let work = CullWork::new(&gpu, &pipelines, "raa-vg cull variants", capacity, capacity);
    let stale_group = pipelines.scene_bind_group(&gpu, &slots, &scene, &stale);
    let current_group = pipelines.scene_bind_group(&gpu, &slots, &scene, &current);

    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut encoder = device.create_command_encoder(&Default::default());
    pipelines.record(
        &mut encoder,
        &scene,
        &stale_group,
        MAIN_FIRST,
        &work,
        CullPhase::First,
    );
    pipelines.record(
        &mut encoder,
        &scene,
        &current_group,
        MAIN_SECOND,
        &work,
        CullPhase::Second,
    );
    gpu.queue().submit(Some(encoder.finish()));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    let error = block_on(errors.pop());
    assert!(error.is_none(), "cull validation: {error:?}");

    // CullStateRead words: 4 sw_count, 5 hw_count, 6 sw_base, 7 hw_base (the counts when the
    // second phase began, so the first phase's selection), 9 dropped.
    let state = read_buffer(&gpu, work.state(), 48);
    let word = |bytes: &[u8], index: usize| {
        u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().expect("4 bytes"))
    };
    let (sw_count, hw_count) = (word(&state, 4), word(&state, 5));
    let (sw_base, hw_base) = (word(&state, 6), word(&state, 7));
    let first_drawn = sw_base + hw_base;
    let second_drawn = (sw_count - sw_base) + (hw_count - hw_base);
    assert_eq!(word(&state, 9), 0, "no cluster is dropped");
    assert_eq!(
        first_drawn, 0,
        "the stale pyramid rejects every cluster in phase 0"
    );
    assert!(
        second_drawn > 0,
        "phase 1 recovers clusters, drew {second_drawn}"
    );

    // Visible entries: software ones from the front, hardware ones from the back.
    let visible = read_buffer(&gpu, work.visible(), capacity * 8);
    let entry = |index: u64| {
        (
            word(&visible, index as usize * 2),
            word(&visible, index as usize * 2 + 1),
        )
    };
    let entries: Vec<(u32, u32)> = (0..u64::from(sw_count))
        .map(entry)
        .chain((0..u64::from(hw_count)).map(|i| entry(capacity - 1 - i)))
        .collect();
    let second_bit = 0x8000_0000u32;
    assert!(
        entries
            .iter()
            .all(|&(_, cluster)| cluster & second_bit != 0),
        "every recovered cluster carries the second-phase bit: {entries:?}"
    );
    let drawn: std::collections::BTreeSet<u32> = entries.iter().map(|&(i, _)| i).collect();
    assert_eq!(
        drawn,
        (0..COUNT).collect(),
        "phase 1 draws all {COUNT} instances: {entries:?}"
    );
}
