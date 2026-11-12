//! Hierarchy of the cluster DAG: leaves from sibling sets, the batched node tree with its
//! fan-out, bounds and parent-error propagation, and the resumable hierarchy state.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/hierarchy.rs`, revision 7e465b5.

use std::collections::VecDeque;
use std::ops::Range;

use raa_math::Vec3;

use crate::bounds::Sphere;
use crate::packed::PackedNode;
use crate::partition::morton_order;

/// Maximum children of an interior node.
pub const FAN_OUT: usize = 8;

/// Most clusters one hierarchy unit builds: a level above it is cut into runs.
pub const HIERARCHY_BATCH: u32 = 1024;

/// One sibling set, ready to become a leaf node.
#[derive(Clone, Copy, Debug)]
pub struct Leaf {
    pub level: u32,
    pub first_cluster: u32,
    pub cluster_count: u32,
    pub bounds: Sphere,
    pub parent_lod_bounds: Sphere,
    pub parent_error: f32,
}

/// The finished tree; `depth` is the longest root-to-leaf path counted in nodes.
#[derive(Clone, Debug, PartialEq)]
pub struct Hierarchy {
    pub nodes: Vec<PackedNode>,
    pub root: u32,
    pub depth: u32,
}

/// A subtree root built but not stored yet: its children are stored with it.
#[derive(Clone, Copy, Debug)]
struct Pending {
    node: PackedNode,
    depth: u32,
}

/// A run of consecutive leaves (indices into the state's leaves), built by one unit.
#[derive(Clone, Copy, Debug)]
struct Run {
    start: usize,
    end: usize,
    last_of_level: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Stage {
    Runs,
    Combine,
    Finish,
    Done,
}

/// Resumable hierarchy build. Each unit builds one run, combines one chunk of run roots of
/// the current level, or stores the level roots and the root.
pub struct HierarchyState {
    leaves: Vec<Leaf>,
    runs: Vec<Run>,
    run_cursor: usize,
    stage: Stage,
    pending: VecDeque<Pending>,
    next: Vec<Pending>,
    level_roots: Vec<Pending>,
    nodes: Vec<PackedNode>,
    root: u32,
    depth: u32,
    units_done: usize,
    units_total: usize,
}

impl HierarchyState {
    /// Plans the build: a stable sort by level, Morton order for each level above one batch,
    /// and the cut of every level into runs. The sorts are the only whole-array work here.
    pub fn new(mut leaves: Vec<Leaf>) -> HierarchyState {
        leaves.sort_by_key(|leaf| leaf.level);
        let mut runs = Vec::new();
        let mut combine_units = 0;
        for range in level_ranges(&leaves) {
            let first_run = runs.len();
            if let Some(level) = leaves.get_mut(range.clone()) {
                cut_level(level, range.start, &mut runs);
            }
            combine_units += combine_steps(runs.len() - first_run);
        }
        let stage = if leaves.is_empty() {
            Stage::Done
        } else {
            Stage::Runs
        };
        HierarchyState {
            leaves,
            units_total: runs.len() + combine_units + 1,
            runs,
            run_cursor: 0,
            stage,
            pending: VecDeque::new(),
            next: Vec::new(),
            level_roots: Vec::new(),
            nodes: Vec::new(),
            root: 0,
            depth: 0,
            units_done: 0,
        }
    }

    /// Runs one unit; true once the root is stored.
    pub fn step(&mut self) -> bool {
        match self.stage {
            Stage::Runs => self.run_unit(),
            Stage::Combine => self.combine_unit(),
            Stage::Finish => self.finish_unit(),
            Stage::Done => return true,
        }
        self.units_done += 1;
        self.settle();
        self.stage == Stage::Done
    }

    /// Fraction of the units done, 0..=1, never decreasing; 1 only once the root is stored.
    pub fn progress(&self) -> f32 {
        if self.stage == Stage::Done {
            return 1.0;
        }
        (self.units_done as f32 / self.units_total as f32).min(1.0)
    }

    /// Completes any remaining units and returns the hierarchy.
    pub fn finish(mut self) -> Hierarchy {
        while !self.step() {}
        Hierarchy {
            nodes: self.nodes,
            root: self.root,
            depth: self.depth,
        }
    }

    fn run_unit(&mut self) {
        let Some(run) = self.runs.get(self.run_cursor).copied() else {
            return;
        };
        self.run_cursor += 1;
        if let Some(leaves) = self.leaves.get_mut(run.start..run.end) {
            let (node, depth) = build_subtree(leaves, &mut self.nodes);
            self.pending.push_back(Pending { node, depth });
        }
        if run.last_of_level {
            self.stage = Stage::Combine;
        }
    }

    /// Combines the next chunk of at most FAN_OUT run roots; a lone root passes through.
    fn combine_unit(&mut self) {
        let take = self.pending.len().min(FAN_OUT);
        let chunk: Vec<Pending> = self.pending.drain(..take).collect();
        match chunk.as_slice() {
            [only] => self.next.push(*only),
            _ => {
                let nodes: Vec<PackedNode> = chunk.iter().map(|p| p.node).collect();
                let depth = chunk.iter().map(|p| p.depth).fold(0, u32::max) + 1;
                let parent = summarize(&nodes, node_index(&self.nodes));
                self.nodes.extend(nodes);
                self.next.push(Pending {
                    node: parent,
                    depth,
                });
            }
        }
        if self.pending.is_empty() {
            self.pending.extend(self.next.drain(..));
        }
    }

    /// Stores the level roots contiguously, then the root summarising them.
    fn finish_unit(&mut self) {
        let roots: Vec<PackedNode> = self.level_roots.iter().map(|p| p.node).collect();
        let depth = self.level_roots.iter().map(|p| p.depth).fold(0, u32::max) + 1;
        let root = summarize(&roots, node_index(&self.nodes));
        self.nodes.extend(roots);
        self.root = node_index(&self.nodes);
        self.nodes.push(root);
        self.depth = depth;
        self.stage = Stage::Done;
    }

    /// Closes a level once its run roots have combined into one, then moves on.
    fn settle(&mut self) {
        if self.stage == Stage::Combine && self.pending.len() == 1 && self.next.is_empty() {
            self.level_roots.extend(self.pending.drain(..));
            self.stage = if self.run_cursor < self.runs.len() {
                Stage::Runs
            } else {
                Stage::Finish
            };
        }
    }
}

/// Builds the hierarchy of `leaves`; no leaf gives the empty hierarchy.
pub fn build_hierarchy(leaves: Vec<Leaf>) -> Hierarchy {
    HierarchyState::new(leaves).finish()
}

fn level_ranges(leaves: &[Leaf]) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    for (i, leaf) in leaves.iter().enumerate() {
        if leaves.get(i + 1).map(|next| next.level) != Some(leaf.level) {
            ranges.push(start..i + 1);
            start = i + 1;
        }
    }
    ranges
}

/// Orders one level (Morton order when it covers more than one batch) and cuts it into runs
/// of at most HIERARCHY_BATCH clusters, at least one leaf each.
fn cut_level(level: &mut [Leaf], offset: usize, runs: &mut Vec<Run>) {
    if level.is_empty() {
        return;
    }
    let total: u64 = level.iter().map(|leaf| u64::from(leaf.cluster_count)).sum();
    if total > u64::from(HIERARCHY_BATCH) {
        let centers: Vec<Vec3> = level.iter().map(|leaf| leaf.bounds.center).collect();
        let order = morton_order(&centers);
        let sorted: Vec<Leaf> = order
            .iter()
            .filter_map(|&i| level.get(i as usize).copied())
            .collect();
        if sorted.len() == level.len() {
            level.copy_from_slice(&sorted);
        }
    }
    let mut start = 0;
    let mut clusters = 0u64;
    for (i, leaf) in level.iter().enumerate() {
        let count = u64::from(leaf.cluster_count);
        if i > start && clusters + count > u64::from(HIERARCHY_BATCH) {
            runs.push(Run {
                start: offset + start,
                end: offset + i,
                last_of_level: false,
            });
            start = i;
            clusters = 0;
        }
        clusters += count;
    }
    runs.push(Run {
        start: offset + start,
        end: offset + level.len(),
        last_of_level: true,
    });
}

/// Combine units that turn `runs` run roots into one level root.
fn combine_steps(mut runs: usize) -> usize {
    let mut steps = 0;
    while runs > 1 {
        runs = runs.div_ceil(FAN_OUT);
        steps += runs;
    }
    steps
}

/// Returns the subtree's root, not yet stored, and its depth. Children are stored together
/// because the GPU format addresses them as a run; grandchildren are stored before them.
fn build_subtree(leaves: &mut [Leaf], nodes: &mut Vec<PackedNode>) -> (PackedNode, u32) {
    if let [leaf] = leaves {
        return (leaf_node(leaf), 1);
    }
    let mut children = Vec::with_capacity(FAN_OUT);
    let mut depth = 0;
    for chunk in split_spatially(leaves) {
        let (child, child_depth) = build_subtree(chunk, nodes);
        children.push(child);
        depth = depth.max(child_depth);
    }
    let parent = summarize(&children, node_index(nodes));
    nodes.extend(children);
    (parent, depth + 1)
}

/// Splits into at most FAN_OUT contiguous, spatially coherent runs by repeatedly halving the
/// largest run along its longest axis.
fn split_spatially(leaves: &mut [Leaf]) -> Vec<&mut [Leaf]> {
    let mut runs: Vec<&mut [Leaf]> = vec![leaves];
    while runs.len() < FAN_OUT {
        let largest = runs
            .iter()
            .enumerate()
            .max_by_key(|(_, run)| run.len())
            .map(|(i, _)| i);
        let Some(largest) = largest.filter(|&i| runs.get(i).is_some_and(|run| run.len() >= 2))
        else {
            break;
        };
        let run = runs.swap_remove(largest);
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for leaf in run.iter() {
            lo = lo.min(leaf.bounds.center);
            hi = hi.max(leaf.bounds.center);
        }
        let axis = longest_axis(hi - lo);
        let middle = run.len() / 2;
        run.select_nth_unstable_by(middle, |a, b| {
            component(a.bounds.center, axis).total_cmp(&component(b.bounds.center, axis))
        });
        let (left, right) = run.split_at_mut(middle);
        runs.push(left);
        runs.push(right);
    }
    runs
}

fn longest_axis(extent: Vec3) -> usize {
    let mut axis = 0;
    for candidate in 1..3 {
        if component(extent, candidate) > component(extent, axis) {
            axis = candidate;
        }
    }
    axis
}

fn component(v: Vec3, axis: usize) -> f32 {
    let [x, y, z] = v.to_array();
    match axis {
        0 => x,
        1 => y,
        _ => z,
    }
}

fn leaf_node(leaf: &Leaf) -> PackedNode {
    PackedNode {
        center: leaf.bounds.center.to_array(),
        radius: leaf.bounds.radius,
        lod_center: leaf.parent_lod_bounds.center.to_array(),
        lod_radius: leaf.parent_lod_bounds.radius,
        max_parent_error: leaf.parent_error,
        first: leaf.first_cluster,
        count: leaf.cluster_count,
        is_leaf: 1,
    }
}

/// The inner node over `children`, which are stored from index `first`.
fn summarize(children: &[PackedNode], first: u32) -> PackedNode {
    let bounds = Sphere::merged_all(children.iter().map(|c| sphere_of(c.center, c.radius)));
    let lod = Sphere::merged_all(
        children
            .iter()
            .map(|c| sphere_of(c.lod_center, c.lod_radius)),
    );
    PackedNode {
        center: bounds.center.to_array(),
        radius: bounds.radius,
        lod_center: lod.center.to_array(),
        lod_radius: lod.radius,
        max_parent_error: children
            .iter()
            .map(|c| c.max_parent_error)
            .fold(0.0, f32::max),
        first,
        count: u32::try_from(children.len()).unwrap_or(u32::MAX),
        is_leaf: 0,
    }
}

fn sphere_of(center: [f32; 3], radius: f32) -> Sphere {
    Sphere {
        center: Vec3::from_array(center),
        radius,
    }
}

fn node_index(nodes: &[PackedNode]) -> u32 {
    u32::try_from(nodes.len()).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Leaf `i` of a level, with `clusters` clusters at first cluster `i * clusters`.
    fn leaf(i: u32, level: u32, clusters: u32) -> Leaf {
        let center = Vec3::new(i as f32 * 2.0, (i % 7) as f32, (i * 3 % 11) as f32);
        Leaf {
            level,
            first_cluster: i * clusters,
            cluster_count: clusters,
            bounds: Sphere {
                center,
                radius: 0.75,
            },
            parent_lod_bounds: Sphere {
                center: center + Vec3::ONE,
                radius: 1.5,
            },
            parent_error: (i % 13) as f32 + 0.5,
        }
    }

    fn sphere(node: &PackedNode) -> Sphere {
        sphere_of(node.center, node.radius)
    }

    /// Checks the tree rules from `node`, collecting the first cluster of each leaf reached
    /// and returning the longest path in nodes.
    fn walk(h: &Hierarchy, node: u32, found: &mut Vec<u32>) -> u32 {
        let packed = h.nodes[node as usize];
        if packed.is_leaf == 1 {
            found.push(packed.first);
            return 1;
        }
        assert!(packed.count as usize <= FAN_OUT);
        let mut deepest = 0;
        let mut max_error = 0.0f32;
        for child in packed.first..packed.first + packed.count {
            assert!(child < node, "children are stored before their parent");
            let c = h.nodes[child as usize];
            assert!(sphere(&packed).contains(sphere(&c)));
            max_error = max_error.max(c.max_parent_error);
            deepest = deepest.max(walk(h, child, found));
        }
        assert_eq!(packed.max_parent_error, max_error);
        1 + deepest
    }

    /// Asserts that every leaf in `expected` is reachable exactly once; returns the depth.
    fn check_tree(h: &Hierarchy, expected: Vec<u32>) -> u32 {
        assert_eq!(h.root as usize, h.nodes.len() - 1);
        let mut found = Vec::new();
        let depth = walk(h, h.root, &mut found);
        found.sort_unstable();
        assert_eq!(found, expected);
        assert_eq!(depth, h.depth);
        depth
    }

    fn many_leaves() -> Vec<Leaf> {
        (0..400).map(|i| leaf(i, 0, 8)).collect()
    }

    #[test]
    fn one_leaf_gives_leaf_and_root() {
        let h = build_hierarchy(vec![leaf(0, 0, 3)]);
        assert_eq!(h.nodes.len(), 2);
        assert_eq!((h.root, h.depth), (1, 2));
        assert_eq!(h.nodes[0].is_leaf, 1);
        assert_eq!(h.nodes[1].is_leaf, 0);
    }

    #[test]
    fn one_level_of_100_leaves_is_one_tree() {
        let h = build_hierarchy((0..100).map(|i| leaf(i, 0, 1)).collect());
        check_tree(&h, (0..100).collect());
    }

    #[test]
    fn three_levels_hang_under_three_level_roots() {
        let mut leaves: Vec<Leaf> = (0..12)
            .map(|i| {
                leaf(
                    i,
                    if i < 4 {
                        0
                    } else if i < 9 {
                        1
                    } else {
                        2
                    },
                    1,
                )
            })
            .collect();
        leaves.reverse();
        let h = build_hierarchy(leaves);
        let depth = check_tree(&h, (0..12).collect());
        assert!(depth >= 3);
        let root = h.nodes[h.root as usize];
        assert_eq!(root.count, 3);
        for (k, (lo, hi)) in [(0, 4), (4, 9), (9, 12)].into_iter().enumerate() {
            let mut found = Vec::new();
            walk(&h, root.first + k as u32, &mut found);
            found.sort_unstable();
            assert_eq!(found, (lo..hi).collect::<Vec<u32>>());
        }
    }

    #[test]
    fn many_clusters_are_cut_into_runs() {
        let h = build_hierarchy(many_leaves());
        check_tree(&h, (0..400).map(|i| i * 8).collect());
    }

    #[test]
    fn stepping_equals_build_hierarchy() {
        let mut state = HierarchyState::new(many_leaves());
        let mut last = 0.0;
        while !state.step() {
            let p = state.progress();
            assert!((last..1.0).contains(&p));
            last = p;
        }
        assert_eq!(state.progress(), 1.0);
        assert_eq!(state.finish(), build_hierarchy(many_leaves()));
    }

    #[test]
    fn no_leaf_gives_empty_hierarchy() {
        let h = build_hierarchy(Vec::new());
        assert_eq!(
            h,
            Hierarchy {
                nodes: Vec::new(),
                root: 0,
                depth: 0
            }
        );
        assert!(HierarchyState::new(Vec::new()).step());
    }
}
