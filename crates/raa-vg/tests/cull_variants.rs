//! The S32 cull-control module and the `pass_view.wgsl` scene twins: naga validation of the
//! composed module, its entry points, group-2 bindings and per-stage storage-buffer count, the
//! layouter sizes of every shared record and the `depth_calls` text helper the later modules
//! pin their depth helpers with.

#![cfg(not(target_arch = "wasm32"))]

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
