//! Conservative median-split instance BVH (section 67).

use crate::program::TraceProgram;

pub const LEAF: usize = 4;
pub const NODE_WORDS: usize = 8;
pub const TRAVERSAL_STACK: usize = 32;

/// Leaf: instances at `order[left..left + count]`. Internal: children at `left`, `left + 1`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Node { pub min: [f32; 3], pub max: [f32; 3], pub left: u32, pub count: u32 }

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bvh { pub nodes: Vec<Node>, pub order: Vec<u32> }

/// Conservative world bounds of one instance.
pub fn instance_bounds(program: &TraceProgram, index: usize) -> ([f32; 3], [f32; 3]) {
    let Some((instance, source)) = program.instances.get(index)
        .and_then(|i| program.sources.get(i.source as usize).map(|s| (i, s)))
    else { return ([-1e30; 3], [1e30; 3]); };
    let (mut lo, mut hi) = (source.min, source.max);
    if let (1, Some(bend)) = (instance.deform_kind, instance.bend) {
        let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
        let margin = 16.0 * (bend[1] as f32).to_radians().abs().max(1.0) * (diag + bend[3].abs() as f32 + bend[4].abs() as f32 + (bend[5].powi(2) + bend[6].powi(2) + bend[7].powi(2)).sqrt() as f32 + 1.0);
        (0..3).for_each(|a| { lo[a] -= margin; hi[a] += margin; });
    }
    let matrix = &instance.world_from_local;
    let (mut min, mut max) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for corner in 0..8 {
        let p = [0, 1, 2].map(|a| lo[a] + (hi[a] - lo[a]) * ((corner >> a) & 1) as f32);
        for axis in 0..3 {
            let value = matrix[axis][0] * p[0] + matrix[axis][1] * p[1] + matrix[axis][2] * p[2] + matrix[axis][3];
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }
    (min, max)
}

/// Build the median-split tree over every instance of `program`.
pub fn build(program: &TraceProgram) -> Bvh {
    let count = program.instances.len();
    let bounds: Vec<_> = (0..count).map(|i| instance_bounds(program, i)).collect();
    let mut bvh = Bvh { nodes: vec![Node::default()], order: (0..count as u32).collect() };
    split(&mut bvh, &bounds, 0, count, 0, 0);
    bvh
}

fn split(bvh: &mut Bvh, bounds: &[([f32; 3], [f32; 3])], start: usize, end: usize, index: usize, depth: usize) {
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for &i in &bvh.order[start..end] {
        let (a, b) = bounds[i as usize];
        (0..3).for_each(|k| { lo[k] = lo[k].min(a[k]); hi[k] = hi[k].max(b[k]); });
    }
    if end - start <= LEAF {
        bvh.nodes[index] = Node { min: lo, max: hi, left: start as u32, count: (end - start) as u32 };
        return;
    }
    let axis = depth % 3;
    let key = |i: u32| bounds[i as usize].0[axis] + bounds[i as usize].1[axis];
    bvh.order[start..end].sort_by(|a, b| key(*a).total_cmp(&key(*b)).then(a.cmp(b)));
    let mid = start + (end - start) / 2;
    let children = bvh.nodes.len();
    bvh.nodes.extend([Node::default(), Node::default()]);
    bvh.nodes[index] = Node { min: lo, max: hi, left: children as u32, count: 0 };
    split(bvh, bounds, start, mid, children, depth + 1);
    split(bvh, bounds, mid, end, children + 1, depth + 1);
}
