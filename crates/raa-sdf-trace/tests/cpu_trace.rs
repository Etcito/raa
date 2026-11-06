//! Section 67 CPU twin: analytic distances, smooth combinations, transforms, materials and placeholders.
use raa_sdf_trace::{Instance, Material, Source, TraceProgram, bvh, cpu, opcode};
const IDENTITY: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];
fn moved(x: f32) -> [[f32; 4]; 3] { let mut matrix = IDENTITY; matrix[0][3] = -x; matrix }
fn primitive(operation: u32, matrix: [[f32; 4]; 3], params: [f32; 4], material: u32) -> (u32, Vec<f32>) {
    let mut data: Vec<f32> = matrix.iter().flatten().copied().collect();
    data.extend([1.0, material as f32]); data.extend(params); (operation, data)
}
fn program(parts: &[(u32, Vec<f32>)], instances: Vec<Instance>) -> TraceProgram {
    let (mut code, mut data) = (Vec::new(), Vec::new());
    for (operation, words) in parts {
        code.push(operation | ((data.len() as u32) << 8));
        data.extend_from_slice(words);
    }
    let material = Material { color: [1.0, 0.0, 0.0], roughness: 0.5, metallic: 0.0 };
    let source = Source { code_start: 0, code_len: code.len() as u32, data_start: 0, data_len: data.len() as u32, min: [-4.0; 3], max: [4.0; 3], lipschitz: 1.0, flags: 0 };
    TraceProgram { sources: vec![source], code, data, materials: vec![material, material], instances }
}
fn instance(x: f32, scale: f32) -> Instance {
    Instance {
        world_from_local: [[scale, 0.0, 0.0, x], [0.0, scale, 0.0, 0.0], [0.0, 0.0, scale, 0.0]],
        local_from_world: [[1.0 / scale, 0.0, 0.0, -x / scale], [0.0, 1.0 / scale, 0.0, 0.0], [0.0, 0.0, 1.0 / scale, 0.0]],
        dist_scale: scale, source: 0, material: -1, key: 0, deform_kind: 0, bend_angle: 0.0, bend_length: 0.0, axes: 0, bend: None,
    }
}
fn two_spheres(operation: u32, smooth: f32) -> TraceProgram {
    program(&[
        primitive(opcode::SPHERE, moved(-0.5), [0.5, 0.0, 0.0, 0.0], 0),
        primitive(opcode::SPHERE, moved(0.5), [0.5, 0.0, 0.0, 0.0], 1),
        (operation, vec![smooth]),
    ], vec![instance(0.0, 1.0)])
}
#[test]
fn all_eight_primitives_match_analytic_distances() {
    let cases = [
        (opcode::BOX, [0.5, 0.5, 0.5, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5),
        (opcode::SPHERE, [0.5, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5),
        (opcode::ROUND_BOX, [0.5, 0.5, 0.5, 0.1], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0], 0.1),
        (opcode::CAPSULE, [0.5, 0.25, 0.0, 0.0], [0.0, 0.75, 0.0], [0.0, 1.0, 0.0], 0.25),
        (opcode::CYLINDER, [0.5, 0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0], 0.5),
        (opcode::CONE, [0.5, 0.5, 0.0, 0.0], [0.5, -0.5, 0.0], [0.0, 1.0, 0.0], 0.5),
        (opcode::TORUS, [0.5, 0.15, 0.0, 0.0], [0.65, 0.0, 0.0], [0.0, 0.0, 0.0], 0.35),
        (opcode::ELLIPSOID, [0.5, 0.3, 0.4, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0], -0.3),
    ];
    for (operation, params, boundary, outside, expected) in cases {
        let program = program(&[primitive(operation, IDENTITY, params, 0)], vec![instance(0.0, 1.0)]);
        assert!(cpu::distance(&program, 0, boundary).abs() < 1e-5, "op {operation} boundary");
        assert!((cpu::distance(&program, 0, outside) - expected).abs() < 1e-5, "op {operation} outside");
    }
}
#[test]
fn smooth_combinations_and_material_selection_follow_the_stack() {
    for (operation, expected) in [(opcode::UNION, -0.125), (opcode::SUBTRACT, 0.125), (opcode::INTERSECT, 0.125)] {
        let program = two_spheres(operation, 0.5);
        assert!((cpu::distance(&program, 0, [0.0, 0.0, 0.0]) - expected).abs() < 1e-6, "op {operation}");
    }
    let union = two_spheres(opcode::UNION, 0.0);
    assert_eq!((cpu::material(&union, 0, [0.9, 0.0, 0.0]), cpu::material(&union, 0, [-0.9, 0.0, 0.0])), (1, 0));
    assert_eq!(cpu::material(&two_spheres(opcode::SUBTRACT, 0.0), 0, [0.9, 0.0, 0.0]), 0);
    let intersect = two_spheres(opcode::INTERSECT, 0.0);
    assert_eq!((cpu::material(&intersect, 0, [0.9, 0.0, 0.0]), cpu::material(&intersect, 0, [-0.9, 0.0, 0.0])), (0, 1));
    let mut overridden = two_spheres(opcode::UNION, 0.0);
    overridden.instances[0].material = 1;
    assert_eq!(cpu::material(&overridden, 0, [-0.9, 0.0, 0.0]), 1);
}
#[test]
fn local_and_instance_transforms_place_the_primitive() {
    let source_moved = program(&[primitive(opcode::SPHERE, moved(1.0), [0.5, 0.0, 0.0, 0.0], 0)], vec![instance(2.0, 1.0)]);
    assert!(cpu::distance(&source_moved, 0, [3.5, 0.0, 0.0]).abs() < 1e-5);
    assert!((cpu::distance(&source_moved, 0, [3.0, 0.0, 0.0]) + 0.5).abs() < 1e-5);
    let scaled = program(&[primitive(opcode::SPHERE, IDENTITY, [0.5, 0.0, 0.0, 0.0], 0)], vec![instance(2.0, 2.0)]);
    assert!(cpu::distance(&scaled, 0, [3.0, 0.0, 0.0]).abs() < 1e-5);
    assert!((cpu::distance(&scaled, 0, [4.0, 0.0, 0.0]) - 1.0).abs() < 1e-5);
    assert!((cpu::distance(&scaled, 0, [2.0, 0.0, 0.0]) + 1.0).abs() < 1e-5);
}
#[test]
fn trace_ray_hits_the_nearest_instance_and_a_placeholder_box() {
    let spheres = program(&[primitive(opcode::SPHERE, IDENTITY, [0.5, 0.0, 0.0, 0.0], 1)], vec![instance(1.0, 1.0), instance(3.0, 1.0)]);
    let hit = cpu::trace_ray(&spheres, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]).expect("a hit");
    assert_eq!((hit.instance, hit.material), (0, 1));
    assert!((hit.t - 0.5).abs() < 1e-4 && (hit.normal[0] + 1.0).abs() < 1e-3, "{hit:?}");
    let hit = cpu::trace_ray(&spheres, [5.0, 0.0, 0.0], [-1.0, 0.0, 0.0]).expect("a hit");
    assert_eq!(hit.instance, 1);
    assert!((hit.t - 1.5).abs() < 1e-4 && (hit.normal[0] - 1.0).abs() < 1e-3, "{hit:?}");
    assert!(cpu::trace_ray(&spheres, [0.0, 5.0, 0.0], [0.0, 1.0, 0.0]).is_none());
    let mut boxed = program(&[primitive(opcode::BOX, IDENTITY, [0.5, 0.5, 0.5, 0.0], 0)], vec![instance(0.0, 1.0)]);
    boxed.sources[0].flags = 1;
    boxed.sources[0].min = [-0.5; 3];
    boxed.sources[0].max = [0.5; 3];
    assert!(boxed.sources[0].placeholder());
    assert!((cpu::distance(&boxed, 0, [1.0, 0.0, 0.0]) - 0.5).abs() < 1e-6);
    let hit = cpu::trace_ray(&boxed, [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]).expect("a hit");
    assert!((hit.t - 1.5).abs() < 1e-4, "{hit:?}");
}
#[test]
fn negative_and_non_uniform_scales_keep_the_hit_and_the_tree_conservative() {
    let mut scaled = program(&[primitive(opcode::SPHERE, IDENTITY, [0.5, 0.0, 0.0, 0.0], 0)], vec![instance(0.0, 1.0)]);
    let item = &mut scaled.instances[0];
    item.world_from_local = [[-2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]];
    item.local_from_world = [[-0.5, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0 / 3.0, 0.0]];
    item.dist_scale = 0.5;
    let hit = cpu::trace_ray(&scaled, [-5.0, 0.0, 0.0], [1.0, 0.0, 0.0]).expect("a mirrored sphere is hit");
    assert!((hit.t - 4.0).abs() < 1e-3, "{hit:?}");
    assert!(cpu::trace_ray(&scaled, [0.0, 5.0, 0.0], [0.0, 1.0, 0.0]).is_none());
    let tree = bvh::build(&scaled);
    assert_eq!(core::mem::size_of::<bvh::Node>(), 32);
    assert!(tree.nodes.iter().all(|node| node.count <= bvh::LEAF as u32));
}
#[test]
fn bent_reference_points_follow_mama_rail_semantics_and_are_never_stepped_over() {
    let radius = 4.0 / std::f32::consts::PI;
    let mut bent = program(&[primitive(opcode::SPHERE, moved(2.1), [0.1, 0.0, 0.0, 0.0], 0)], vec![instance(0.0, 1.0)]);
    bent.instances[0].deform_kind = 1;
    bent.instances[0].bend_angle = -std::f32::consts::FRAC_PI_2;
    bent.instances[0].bend_length = 2.0;
    bent.instances[0].bend = Some([0.0, -90.0, 90.0, 0.0, 2.0, 0.0, 0.0, 0.0]);
    assert!(cpu::distance(&bent, 0, [radius, 0.0, -(radius + 0.2)]).abs() < 1e-4);
    let hit = cpu::trace_ray(&bent, [radius, 0.0, -8.0], [0.0, 0.0, 1.0]).expect("the bent surface is hit");
    assert!(hit.t < 8.0 - radius + 1e-3, "{hit:?}");
    assert!(cpu::distance(&bent, 0, [radius, 0.0, -8.0 + hit.t]).abs() < 1e-3, "{hit:?}");
    bent.instances[0].bend = Some([0.0, 90.0, 90.0, 0.0, 2.0, 0.0, 0.0, 0.0]);
    bent.instances[0].bend_angle = std::f32::consts::FRAC_PI_2;
    bent.instances[0].axes = 1;
    assert!(cpu::distance(&bent, 0, [radius, 0.0, radius + 0.2]).abs() < 1e-4);
    let mut matrix = IDENTITY;
    matrix[0][3] = -0.5;
    matrix[1][3] = -1.0;
    let mut non_default = program(&[primitive(opcode::SPHERE, matrix, [0.1, 0.0, 0.0, 0.0], 0)], vec![instance(0.0, 1.0)]);
    non_default.instances[0].deform_kind = 1;
    non_default.instances[0].bend_angle = std::f32::consts::FRAC_PI_2;
    non_default.instances[0].bend_length = 2.0;
    non_default.instances[0].bend = Some([1.0, 90.0, 0.0, -1.0, 1.0, 0.5, 0.0, 0.0]);
    assert!(cpu::distance(&non_default, 0, [0.5, radius - 1.0, radius + 0.1]).abs() < 1e-4);
}
