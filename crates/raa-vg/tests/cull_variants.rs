//! The S32 cull-control module and the `pass_view.wgsl` scene twins: naga validation of the
//! composed module, its entry points, group-2 bindings and per-stage storage-buffer count, the
//! layouter sizes of every shared record and the `depth_calls` text helper the later modules
//! pin their depth helpers with.

#![cfg(not(target_arch = "wasm32"))]

use raa_cluster::weld::{decode_octahedral, encode_octahedral};
use raa_core::testing::{skip, software_gpu};
use raa_core::wgpu;
use raa_math::Vec3;
use raa_shader::Composer;
use raa_vg::culling::{PASS_VIEW_WGSL, cull_control_module_wgsl};
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
