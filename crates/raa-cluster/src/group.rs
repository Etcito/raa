//! Grouping of adjacent clusters of one level: the cluster adjacency, the group seeding and
//! reach, and the merge of groups that came out too small.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/group.rs`, revision 7e465b5.

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
}
