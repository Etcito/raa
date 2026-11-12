//! Grouping of adjacent clusters of one level: the cluster adjacency, the group seeding and
//! reach, and the merge of groups that came out too small.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/group.rs`, revision 7e465b5.

use crate::bounds::Sphere;
use raa_math::Vec3;
use std::collections::BTreeMap;

/// Number of clusters a group aims for.
pub const TARGET_GROUP_SIZE: usize = 16;
/// Groups never grow beyond this, even when merging leftovers.
pub const MAX_GROUP_SIZE: usize = 24;
/// Groups smaller than this pull in spatial neighbours or merge into another group.
pub const MIN_GROUP_SIZE: usize = 6;

/// Triangles one collect unit reads.
const TRIANGLES_PER_UNIT: usize = 4096;
/// Clusters one collect unit visits.
const CLUSTERS_PER_UNIT: usize = 1024;
/// Edge pairs one unit turns into links.
const PAIRS_PER_UNIT: usize = 4096;
/// Links one unit counts into the per-cluster lists.
const LINKS_PER_UNIT: usize = 4096;
/// Clusters one seeding or merge unit visits.
const GROUP_CLUSTERS_PER_UNIT: usize = 1024;

/// The phase of the sliced adjacency build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Collect,
    SortEdges,
    BuildLinks,
    SortLinks,
    Count,
    Done,
}

/// The sliced cluster adjacency: `triangles[c]` holds the welded position-id triangles of
/// cluster `c`; after the steps, `finish` gives for each cluster its neighbours and the number
/// of edges shared with each, ascending by neighbour.
pub struct AdjacencyState {
    cluster_count: usize,
    phase: Phase,
    cluster: usize,
    triangle: usize,
    edges: Vec<(u64, u32)>,
    edge_cursor: usize,
    links: Vec<(u32, u32)>,
    link_cursor: usize,
    current_link: Option<(u32, u32)>,
    current_count: u32,
    adjacency: Vec<Vec<(u32, u32)>>,
}

impl AdjacencyState {
    /// A state that builds the adjacency of `cluster_count` clusters.
    pub fn new(cluster_count: usize) -> AdjacencyState {
        AdjacencyState {
            cluster_count,
            phase: Phase::Collect,
            cluster: 0,
            triangle: 0,
            edges: Vec::new(),
            edge_cursor: 0,
            links: Vec::new(),
            link_cursor: 0,
            current_link: None,
            current_count: 0,
            adjacency: vec![Vec::new(); cluster_count],
        }
    }

    /// Runs one unit of the build and returns `true` when it is done. The caller keeps passing
    /// the same `triangles` slice; one collect unit reads at most 4096 triangles.
    pub fn step(&mut self, triangles: &[Vec<[u32; 3]>]) -> bool {
        match self.phase {
            Phase::Collect => {
                if self.collect(triangles) {
                    self.phase = Phase::SortEdges;
                }
            }
            Phase::SortEdges => {
                self.edges.sort_unstable();
                self.edges.dedup();
                self.phase = Phase::BuildLinks;
            }
            Phase::BuildLinks => {
                if self.build_links() {
                    self.phase = Phase::SortLinks;
                }
            }
            Phase::SortLinks => {
                self.links.sort_unstable();
                self.phase = Phase::Count;
            }
            Phase::Count => {
                if self.count_links() {
                    self.phase = Phase::Done;
                }
            }
            Phase::Done => return true,
        }
        self.phase == Phase::Done
    }

    /// The per-cluster neighbour lists, ascending by neighbour. Call it after `step` returned
    /// `true`; a state whose collection is unfinished has no triangles to continue it and
    /// returns the lists of the edges read so far.
    pub fn finish(mut self) -> Vec<Vec<(u32, u32)>> {
        while !matches!(self.phase, Phase::Done | Phase::Collect) {
            self.step(&[]);
        }
        self.adjacency
    }

    /// Reads the next triangles into `(edge key, cluster)` pairs; true when every cluster was
    /// read.
    fn collect(&mut self, triangles: &[Vec<[u32; 3]>]) -> bool {
        let mut read = 0;
        let mut visited = 0;
        while self.cluster < self.cluster_count
            && read < TRIANGLES_PER_UNIT
            && visited < CLUSTERS_PER_UNIT
        {
            let Some(cluster_triangles) = triangles.get(self.cluster) else {
                self.cluster = self.cluster_count;
                self.triangle = 0;
                break;
            };
            visited += 1;
            let remaining = cluster_triangles.len().saturating_sub(self.triangle);
            let take = (TRIANGLES_PER_UNIT - read).min(remaining);
            if take == 0 {
                self.cluster += 1;
                self.triangle = 0;
                continue;
            }
            let end = self.triangle + take;
            for triangle in &cluster_triangles[self.triangle..end] {
                for side in 0..3 {
                    let a = triangle[side];
                    let b = triangle[(side + 1) % 3];
                    let key = u64::from(a.min(b)) << 32 | u64::from(a.max(b));
                    self.edges.push((key, self.cluster as u32));
                }
            }
            read += take;
            self.triangle = end;
            if self.triangle >= cluster_triangles.len() {
                self.cluster += 1;
                self.triangle = 0;
            }
        }
        self.cluster >= self.cluster_count
    }

    /// Turns the runs of equal edges into `(a, b)` links in both directions; true when every
    /// edge was visited.
    fn build_links(&mut self) -> bool {
        let mut visited = 0;
        while self.edge_cursor < self.edges.len() {
            let key = self.edges[self.edge_cursor].0;
            let mut end = self.edge_cursor + 1;
            while end < self.edges.len() && self.edges[end].0 == key {
                end += 1;
            }
            for i in self.edge_cursor..end {
                for j in (i + 1)..end {
                    let a = self.edges[i].1;
                    let b = self.edges[j].1;
                    self.links.push((a, b));
                    self.links.push((b, a));
                }
            }
            visited += end - self.edge_cursor;
            self.edge_cursor = end;
            if visited >= PAIRS_PER_UNIT {
                break;
            }
        }
        self.edge_cursor >= self.edges.len()
    }

    /// Counts the runs of equal links into the per-cluster lists; true when every link was
    /// counted.
    fn count_links(&mut self) -> bool {
        let mut visited = 0;
        while self.link_cursor < self.links.len() && visited < LINKS_PER_UNIT {
            let link = self.links[self.link_cursor];
            self.link_cursor += 1;
            visited += 1;
            if self.current_link == Some(link) {
                self.current_count += 1;
            } else {
                self.flush_link();
                self.current_link = Some(link);
                self.current_count = 1;
            }
        }
        if self.link_cursor >= self.links.len() {
            self.flush_link();
            true
        } else {
            false
        }
    }

    /// Appends the pending run of links to its cluster's list.
    fn flush_link(&mut self) {
        let Some((cluster, neighbour)) = self.current_link.take() else {
            self.current_count = 0;
            return;
        };
        if let Some(list) = self.adjacency.get_mut(cluster as usize) {
            list.push((neighbour, self.current_count));
        }
        self.current_count = 0;
    }
}

/// The adjacency of `triangles` in one call: for each cluster, its neighbours in ascending
/// order with the number of shared edges.
pub fn cluster_adjacency(triangles: &[Vec<[u32; 3]>]) -> Vec<Vec<(u32, u32)>> {
    let mut state = AdjacencyState::new(triangles.len());
    while !state.step(triangles) {}
    state.finish()
}

/// The phase of the sliced grouping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupPhase {
    Adjacency,
    Order,
    Seed,
    Merge,
    Done,
}

/// The sliced grouping of adjacent clusters: the adjacency of [`AdjacencyState`] first, then the
/// Morton order of the bounds centres, the seed and reach loop that grows groups towards
/// [`TARGET_GROUP_SIZE`], and the merge of the groups that came out below [`MIN_GROUP_SIZE`].
/// Each cluster ends in exactly one group and no group is above [`MAX_GROUP_SIZE`].
pub struct GroupState {
    cluster_count: usize,
    phase: GroupPhase,
    adjacency: Option<AdjacencyState>,
    adjacency_lists: Vec<Vec<(u32, u32)>>,
    order: Vec<u32>,
    order_position: usize,
    assigned_total: usize,
    groups: Vec<Vec<u32>>,
    group_of: Vec<u32>,
    merge_cursor: usize,
    progress: f32,
}

impl GroupState {
    /// A state that groups `cluster_count` clusters.
    pub fn new(cluster_count: usize) -> GroupState {
        GroupState {
            cluster_count,
            phase: GroupPhase::Adjacency,
            adjacency: Some(AdjacencyState::new(cluster_count)),
            adjacency_lists: Vec::new(),
            order: Vec::new(),
            order_position: 0,
            assigned_total: 0,
            groups: Vec::new(),
            group_of: vec![u32::MAX; cluster_count],
            merge_cursor: 0,
            progress: 0.0,
        }
    }

    /// Runs one unit of the grouping and returns `true` when it is done. The caller keeps passing
    /// the same slices: the first units build the adjacency, then the Morton order, then each
    /// unit seeds groups until 1024 clusters were assigned, and each merge unit visits at most
    /// 1024 clusters.
    pub fn step(&mut self, triangles: &[Vec<[u32; 3]>], bounds: &[Sphere]) -> bool {
        match self.phase {
            GroupPhase::Adjacency => self.step_adjacency(triangles),
            GroupPhase::Order => self.step_order(bounds),
            GroupPhase::Seed => self.step_seed(bounds),
            GroupPhase::Merge => self.step_merge(),
            GroupPhase::Done => self.bump_progress(1.0),
        }
        self.phase == GroupPhase::Done
    }

    /// The fraction of the grouping done so far, in `0..=1`; it never decreases and is 1 once
    /// the state is done.
    pub fn progress(&self) -> f32 {
        self.progress.clamp(0.0, 1.0)
    }

    /// The cluster indices of each group. Call it after `step` returned `true`; a state that is
    /// still collecting its adjacency returns the groups built so far, and an unfinished merge
    /// is completed here because it needs no input.
    pub fn finish(mut self) -> Vec<Vec<u32>> {
        while self.phase == GroupPhase::Merge {
            self.step_merge();
        }
        self.groups.retain(|group| !group.is_empty());
        self.groups
    }

    fn bump_progress(&mut self, value: f32) {
        if value > self.progress {
            self.progress = value;
        }
    }

    fn step_adjacency(&mut self, triangles: &[Vec<[u32; 3]>]) {
        let Some(mut state) = self.adjacency.take() else {
            self.phase = GroupPhase::Order;
            return;
        };
        if state.step(triangles) {
            self.adjacency_lists = state.finish();
            self.phase = GroupPhase::Order;
            self.bump_progress(0.2);
        } else {
            self.adjacency = Some(state);
        }
    }

    fn step_order(&mut self, bounds: &[Sphere]) {
        let centers: Vec<Vec3> = (0..self.cluster_count)
            .map(|index| bounds.get(index).map_or(Vec3::ZERO, |sphere| sphere.center))
            .collect();
        self.order = crate::partition::morton_order(&centers);
        self.phase = GroupPhase::Seed;
        self.bump_progress(0.3);
    }

    fn step_seed(&mut self, bounds: &[Sphere]) {
        let mut assigned = 0;
        let mut scanned = 0;
        while self.order_position < self.order.len()
            && assigned < GROUP_CLUSTERS_PER_UNIT
            && scanned < GROUP_CLUSTERS_PER_UNIT
        {
            let position = self.order_position;
            self.order_position += 1;
            scanned += 1;
            let Some(&seed) = self.order.get(position) else {
                continue;
            };
            if self
                .group_of
                .get(seed as usize)
                .copied()
                .unwrap_or(u32::MAX)
                != u32::MAX
            {
                continue;
            }
            let id = self.groups.len() as u32;
            let mut group = vec![seed];
            if let Some(slot) = self.group_of.get_mut(seed as usize) {
                *slot = id;
            }
            assigned += 1;
            let mut connection: BTreeMap<u32, u32> = BTreeMap::new();
            self.add_connections(seed, &mut connection);
            while group.len() < TARGET_GROUP_SIZE && assigned < GROUP_CLUSTERS_PER_UNIT {
                let Some(next) = best_connection(&connection) else {
                    break;
                };
                connection.remove(&next);
                group.push(next);
                if let Some(slot) = self.group_of.get_mut(next as usize) {
                    *slot = id;
                }
                assigned += 1;
                self.add_connections(next, &mut connection);
            }
            if group.len() < MIN_GROUP_SIZE
                && let Some(reach) = group_reach(bounds, &group)
            {
                for &candidate in self.order.iter().skip(position + 1).take(32) {
                    if group.len() >= MIN_GROUP_SIZE || assigned >= GROUP_CLUSTERS_PER_UNIT {
                        break;
                    }
                    if self
                        .group_of
                        .get(candidate as usize)
                        .copied()
                        .unwrap_or(u32::MAX)
                        != u32::MAX
                    {
                        continue;
                    }
                    let Some(candidate_bounds) = bounds.get(candidate as usize) else {
                        continue;
                    };
                    if (candidate_bounds.center - reach.center).length()
                        <= reach.radius + candidate_bounds.radius
                    {
                        group.push(candidate);
                        if let Some(slot) = self.group_of.get_mut(candidate as usize) {
                            *slot = id;
                        }
                        assigned += 1;
                    }
                }
            }
            self.groups.push(group);
        }
        self.assigned_total += assigned;
        if self.cluster_count == 0 {
            self.bump_progress(0.8);
        } else {
            let fraction = self.assigned_total as f32 / self.cluster_count as f32;
            self.bump_progress(0.3 + 0.5 * fraction.min(1.0));
        }
        if self.order_position >= self.order.len() {
            self.phase = GroupPhase::Merge;
            self.bump_progress(0.8);
        }
    }

    fn step_merge(&mut self) {
        let mut visited = 0;
        while self.merge_cursor < self.groups.len() && visited < GROUP_CLUSTERS_PER_UNIT {
            let index = self.merge_cursor;
            self.merge_cursor += 1;
            let size = self.groups.get(index).map_or(0, Vec::len);
            visited += size.max(1);
            if size == 0 || size >= MIN_GROUP_SIZE {
                continue;
            }
            if let Some(target) = self.merge_target(index) {
                self.merge_into(index, target);
            }
        }
        if self.merge_cursor >= self.groups.len() {
            self.phase = GroupPhase::Done;
            self.bump_progress(1.0);
        } else if !self.groups.is_empty() {
            let fraction = self.merge_cursor as f32 / self.groups.len() as f32;
            self.bump_progress(0.8 + 0.15 * fraction.min(1.0));
        }
    }

    fn add_connections(&self, cluster: u32, connection: &mut BTreeMap<u32, u32>) {
        let Some(list) = self.adjacency_lists.get(cluster as usize) else {
            return;
        };
        for &(neighbor, shared) in list {
            if self
                .group_of
                .get(neighbor as usize)
                .copied()
                .unwrap_or(u32::MAX)
                == u32::MAX
            {
                *connection.entry(neighbor).or_insert(0) += shared;
            }
        }
    }

    fn merge_target(&self, index: usize) -> Option<u32> {
        let group = self.groups.get(index)?;
        let size = group.len();
        let mut weights: BTreeMap<u32, u32> = BTreeMap::new();
        for &cluster in group {
            let Some(list) = self.adjacency_lists.get(cluster as usize) else {
                continue;
            };
            for &(neighbor, shared) in list {
                let other = self
                    .group_of
                    .get(neighbor as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                if other == u32::MAX || other as usize == index {
                    continue;
                }
                *weights.entry(other).or_insert(0) += shared;
            }
        }
        let mut best: Option<u32> = None;
        let mut best_weight = 0;
        for (&candidate, &weight) in &weights {
            let Some(other) = self.groups.get(candidate as usize) else {
                continue;
            };
            if other.is_empty() || other.len() + size > MAX_GROUP_SIZE {
                continue;
            }
            if best.is_none() || weight > best_weight {
                best = Some(candidate);
                best_weight = weight;
            }
        }
        best
    }

    fn merge_into(&mut self, index: usize, target: u32) {
        let moved = std::mem::take(&mut self.groups[index]);
        for &cluster in &moved {
            if let Some(slot) = self.group_of.get_mut(cluster as usize) {
                *slot = target;
            }
        }
        if let Some(destination) = self.groups.get_mut(target as usize) {
            destination.extend(moved);
        }
    }
}

/// The unassigned neighbour with the strongest connection, ties by the lowest cluster index.
fn best_connection(connection: &BTreeMap<u32, u32>) -> Option<u32> {
    let mut best: Option<(u32, u32)> = None;
    for (&cluster, &weight) in connection {
        let replace = match best {
            None => true,
            Some((best_cluster, best_weight)) => {
                weight > best_weight || (weight == best_weight && cluster < best_cluster)
            }
        };
        if replace {
            best = Some((cluster, weight));
        }
    }
    best.map(|(cluster, _)| cluster)
}

/// A sphere around the group, inflated to twice the merged radius to define "close enough to
/// join"; a group without bounds has no reach.
fn group_reach(bounds: &[Sphere], group: &[u32]) -> Option<Sphere> {
    let first = group
        .first()
        .and_then(|&cluster| bounds.get(cluster as usize))?;
    let mut merged = *first;
    for &cluster in &group[1..] {
        if let Some(sphere) = bounds.get(cluster as usize) {
            merged = merged.merged(*sphere);
        }
    }
    Some(Sphere {
        center: merged.center,
        radius: merged.radius * 2.0,
    })
}

/// The groups of `triangles` in one call: cluster indices per group, every cluster exactly once.
pub fn group_clusters(triangles: &[Vec<[u32; 3]>], bounds: &[Sphere]) -> Vec<Vec<u32>> {
    let mut state = GroupState::new(triangles.len());
    while !state.step(triangles, bounds) {}
    state.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_clusters_sharing_three_edges_report_weight_three() {
        let triangles = vec![vec![[0, 1, 2]], vec![[0, 2, 1]]];
        let adjacency = cluster_adjacency(&triangles);
        assert_eq!(adjacency, vec![vec![(1, 3)], vec![(0, 3)]]);
    }

    #[test]
    fn a_cluster_with_no_shared_edge_has_an_empty_list() {
        let triangles = vec![vec![[0, 1, 2]], vec![[0, 2, 1]], vec![[3, 4, 5]]];
        let adjacency = cluster_adjacency(&triangles);
        assert_eq!(adjacency[0], vec![(1, 3)]);
        assert_eq!(adjacency[1], vec![(0, 3)]);
        assert_eq!(adjacency[2], Vec::new());
    }

    #[test]
    fn lists_are_ascending_by_neighbour() {
        let triangles = vec![vec![[0, 1, 2]], vec![[0, 2, 1]], vec![[0, 1, 3]]];
        let adjacency = cluster_adjacency(&triangles);
        assert_eq!(adjacency[0], vec![(1, 3), (2, 1)]);
        assert_eq!(adjacency[1], vec![(0, 3), (2, 1)]);
    }

    #[test]
    fn an_edge_used_by_three_clusters_links_every_pair() {
        let triangles = vec![vec![[0, 1, 2]], vec![[0, 1, 3]], vec![[1, 0, 4]]];
        let adjacency = cluster_adjacency(&triangles);
        assert_eq!(
            adjacency,
            vec![
                vec![(1, 1), (2, 1)],
                vec![(0, 1), (2, 1)],
                vec![(0, 1), (1, 1)],
            ]
        );
    }

    #[test]
    fn stepping_one_unit_at_a_time_matches_the_one_call_form() {
        let mut triangles = Vec::new();
        for c in 0..300u32 {
            let mut cluster = vec![[c, c + 1, c + 2], [c, c + 2, c + 3]];
            for k in 0..14u32 {
                let base = 100_000 + c * 100 + k * 3;
                cluster.push([base, base + 1, base + 2]);
            }
            triangles.push(cluster);
        }
        let expected = cluster_adjacency(&triangles);
        assert_eq!(expected[0], vec![(1, 2), (2, 1)]);
        let mut state = AdjacencyState::new(triangles.len());
        let mut steps = 0;
        while !state.step(&triangles) {
            steps += 1;
            assert!(steps < 1000);
        }
        assert_eq!(state.finish(), expected);
    }

    const TILE_QUADS: u32 = 8;

    /// Builds one cluster of 128 triangles per tile of an 8 x 8 quad lattice, with the tile's
    /// quads on the shared lattice so neighbouring tiles share position ids. Bounds come from
    /// the tile's vertices.
    fn quad_tiles(tiles: &[(u32, u32)]) -> (Vec<Vec<[u32; 3]>>, Vec<Sphere>) {
        let max_x = tiles.iter().map(|&(x, _)| x).max().unwrap_or(0);
        let stride = (max_x + 1) * TILE_QUADS + 1;
        let vertex = |x: u32, y: u32| y * stride + x;
        let mut triangles = Vec::with_capacity(tiles.len());
        let mut bounds = Vec::with_capacity(tiles.len());
        for &(tile_x, tile_y) in tiles {
            let mut cluster = Vec::with_capacity(128);
            let mut points = Vec::with_capacity(4 * 8 * 8);
            for quad_y in 0..TILE_QUADS {
                for quad_x in 0..TILE_QUADS {
                    let x = tile_x * TILE_QUADS + quad_x;
                    let y = tile_y * TILE_QUADS + quad_y;
                    let a = vertex(x, y);
                    let b = vertex(x + 1, y);
                    let c = vertex(x, y + 1);
                    let d = vertex(x + 1, y + 1);
                    cluster.push([a, c, d]);
                    cluster.push([a, d, b]);
                    points.push(Vec3::new(x as f32, y as f32, 0.0));
                    points.push(Vec3::new((x + 1) as f32, y as f32, 0.0));
                    points.push(Vec3::new(x as f32, (y + 1) as f32, 0.0));
                    points.push(Vec3::new((x + 1) as f32, (y + 1) as f32, 0.0));
                }
            }
            triangles.push(cluster);
            bounds.push(Sphere::from_points(points.into_iter()));
        }
        (triangles, bounds)
    }

    /// A `x` by `y` tile grid of 128-triangle clusters.
    fn quad_grid(x: u32, y: u32) -> (Vec<Vec<[u32; 3]>>, Vec<Sphere>) {
        let mut tiles = Vec::with_capacity((x * y) as usize);
        for tile_y in 0..y {
            for tile_x in 0..x {
                tiles.push((tile_x, tile_y));
            }
        }
        quad_tiles(&tiles)
    }

    #[test]
    fn every_cluster_is_in_exactly_one_group() {
        let (triangles, bounds) = quad_grid(4, 4);
        let groups = group_clusters(&triangles, &bounds);
        let mut seen = vec![0u32; triangles.len()];
        for group in &groups {
            for &cluster in group {
                seen[cluster as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&count| count == 1));
    }

    #[test]
    fn no_group_exceeds_the_maximum_size() {
        let (triangles, bounds) = quad_grid(10, 10);
        let groups = group_clusters(&triangles, &bounds);
        assert!(groups.iter().all(|group| !group.is_empty()));
        assert!(groups.iter().all(|group| group.len() <= MAX_GROUP_SIZE));
    }

    #[test]
    fn a_connected_grid_has_no_group_below_the_minimum_size() {
        let (triangles, bounds) = quad_grid(10, 10);
        let groups = group_clusters(&triangles, &bounds);
        assert!(
            groups.iter().all(|group| group.len() >= MIN_GROUP_SIZE),
            "sizes: {:?}",
            groups.iter().map(Vec::len).collect::<Vec<_>>()
        );
    }

    #[test]
    fn two_far_apart_islands_stay_separate_groups() {
        let (triangles, bounds) = quad_tiles(&[(0, 0), (1, 0), (40, 0), (41, 0)]);
        let groups = group_clusters(&triangles, &bounds);
        assert_eq!(groups.len(), 2);
        let mut sorted: Vec<Vec<u32>> = groups
            .into_iter()
            .map(|mut group| {
                group.sort();
                group
            })
            .collect();
        sorted.sort();
        assert_eq!(sorted, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn progress_never_decreases_and_ends_at_one() {
        let (triangles, bounds) = quad_grid(4, 4);
        let mut state = GroupState::new(triangles.len());
        let mut last = state.progress();
        assert_eq!(last, 0.0);
        let mut steps = 0;
        while !state.step(&triangles, &bounds) {
            let now = state.progress();
            assert!(now >= last);
            assert!((0.0..=1.0).contains(&now));
            last = now;
            steps += 1;
            assert!(steps < 10_000);
        }
        assert_eq!(state.progress(), 1.0);
    }

    #[test]
    fn grouping_unit_steps_match_the_one_call_form() {
        let (triangles, bounds) = quad_grid(6, 6);
        let expected = group_clusters(&triangles, &bounds);
        let mut state = GroupState::new(triangles.len());
        let mut steps = 0;
        while !state.step(&triangles, &bounds) {
            steps += 1;
            assert!(steps < 10_000);
        }
        assert_eq!(state.finish(), expected);
    }
}
