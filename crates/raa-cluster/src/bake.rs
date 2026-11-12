//! Sliced cluster bake: `ClusterBake` advances the DAG build in bounded work units, `bake()`
//! runs it to the end, and the progress and status types report it.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/dag.rs`, lines 92-123, revision 7e465b5.

use std::collections::VecDeque;

use raa_math::Vec3;

use crate::cut::ClusterMesh;
use crate::dag::{BuildCluster, BuiltGroup, LevelState, PackState, push_initial_clusters};
use crate::error::{ClusterError, EMPTY, NOT_FINISHED};
use crate::hierarchy::{Hierarchy, HierarchyState};
use crate::input::{BuildOptions, ClusterInput, InputBounds, OwnedClusterInput};
use crate::partition::{ClusterLimits, PartitionState};
use crate::thin::find_pieces;
use crate::weld::{FOLIAGE_GRID_MARGIN, WELD_BATCH, WeldState, WeldedMesh};

/// Partition clusters pushed into the build list by one unit, at most this many clusters...
const PUSH_CLUSTERS: usize = 1024;
/// ...and this many triangles.
const PUSH_TRIANGLES: usize = 4096;

/// The work one `ClusterBake::step` call may run, in units.
#[derive(Clone, Copy, Debug)]
pub struct WorkBudget {
    pub units: u32,
}

/// Whether a bake has more units to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BakeStatus {
    Working,
    Done,
}

/// The stage of a bake, in the order the stages run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BakeStage {
    Weld,
    Partition,
    Level,
    Hierarchy,
    Pack,
    Done,
}

/// A snapshot of a bake. `fraction` never decreases, is below 1 while working and is exactly 1
/// when done; `total_units` is an estimate and never below `done_units`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BakeProgress {
    pub stage: BakeStage,
    /// The level being built, counted from 1; 0 before the first level.
    pub level: u32,
    pub done_units: u64,
    pub total_units: u64,
    pub fraction: f32,
}

/// The phase of a bake; each phase runs one kind of unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Weld,
    Search,
    Reweld,
    Partition,
    Level,
    Sets,
    Hierarchy,
    Clusters,
    Done,
}

/// A resumable cluster bake. The units run the module states in a fixed order, so the result
/// depends only on the input and the options, never on the budgets.
pub struct ClusterBake {
    input: OwnedClusterInput,
    options: BuildOptions,
    bounds: InputBounds,
    limits: ClusterLimits,
    phase: Phase,
    weld: WeldState,
    weld_steps: usize,
    welded: WeldedMesh,
    partition: PartitionState,
    partition_done: bool,
    pending: VecDeque<Vec<u32>>,
    clusters: Vec<BuildCluster>,
    groups: Vec<BuiltGroup>,
    active: Vec<u32>,
    first_active: usize,
    round_start: usize,
    level_number: u32,
    level: LevelState,
    pack: PackState,
    hierarchy: HierarchyState,
    tree: Hierarchy,
    outcome: Option<Result<ClusterMesh, ClusterError>>,
    plan: u64,
    done_units: u64,
    fraction: f32,
}

impl ClusterBake {
    /// Copies the input and prepares the bake. Refuses invalid options, an invalid mesh and a
    /// mesh without triangle before any work.
    pub fn new(
        input: ClusterInput<'_>,
        options: &BuildOptions,
    ) -> Result<ClusterBake, ClusterError> {
        options.validate()?;
        input.validate()?;
        let input = OwnedClusterInput::from_input(&input);
        let bounds = input.bounds();
        let limits = ClusterLimits {
            max_triangles: options.max_cluster_triangles,
            max_vertices: options.max_cluster_vertices,
        };
        // A rough estimate: a few units per weld batch over the stages.
        let plan = 9 * (input.mesh.triangle_count() as u64).div_ceil(WELD_BATCH as u64) + 2;
        Ok(ClusterBake {
            weld: WeldState::new(&bounds, 0.0),
            weld_steps: 0,
            welded: WeldedMesh {
                positions: Vec::new(),
                attributes: Vec::new(),
                triangles: Vec::new(),
                origin: Vec3::ZERO,
                step: 1.0,
                uv_origin: [0.0; 2],
                uv_step: [1.0; 2],
            },
            partition: PartitionState::new(0, limits, Vec3::ZERO, 1.0),
            partition_done: false,
            pending: VecDeque::new(),
            clusters: Vec::new(),
            groups: Vec::new(),
            active: Vec::new(),
            first_active: 0,
            round_start: 0,
            level_number: 0,
            level: LevelState::new(Vec::new(), 0),
            pack: PackState::new(0),
            hierarchy: HierarchyState::new(Vec::new()),
            tree: Hierarchy {
                nodes: Vec::new(),
                root: 0,
                depth: 0,
            },
            outcome: None,
            plan,
            done_units: 0,
            fraction: 0.0,
            input,
            options: *options,
            bounds,
            limits,
            phase: Phase::Weld,
        })
    }

    /// Runs at most `budget.units` units and reports the status. A zero budget runs nothing, and
    /// a bake that is done stays done.
    pub fn step(&mut self, budget: WorkBudget) -> BakeStatus {
        for _ in 0..budget.units {
            if self.phase == Phase::Done {
                break;
            }
            self.unit();
            self.done_units += 1;
        }
        self.observe();
        if self.phase == Phase::Done {
            BakeStatus::Done
        } else {
            BakeStatus::Working
        }
    }

    /// The stage, level, unit counts and fraction of the bake.
    pub fn progress(&self) -> BakeProgress {
        let total_units = if self.phase == Phase::Done {
            self.done_units
        } else {
            self.plan.max(self.done_units)
        };
        BakeProgress {
            stage: self.stage(),
            level: self.level_number,
            done_units: self.done_units,
            total_units,
            fraction: self.fraction,
        }
    }

    /// The finished mesh, or the error that ended the bake. Refused until the bake is done.
    pub fn finish(mut self) -> Result<ClusterMesh, ClusterError> {
        match self.outcome.take() {
            Some(outcome) => outcome,
            None => Err(ClusterError::new(
                NOT_FINISHED,
                "the bake is not done: keep stepping it until its status is Done",
            )),
        }
    }

    fn stage(&self) -> BakeStage {
        match self.phase {
            Phase::Weld | Phase::Search | Phase::Reweld => BakeStage::Weld,
            Phase::Partition => BakeStage::Partition,
            Phase::Level => BakeStage::Level,
            Phase::Sets | Phase::Hierarchy => BakeStage::Hierarchy,
            Phase::Clusters => BakeStage::Pack,
            Phase::Done => BakeStage::Done,
        }
    }

    /// Records the fraction of the current stage, weighted by stage; kept monotone and below 1
    /// until done.
    fn observe(&mut self) {
        let raw = match self.phase {
            Phase::Weld | Phase::Search | Phase::Reweld => 0.1 * self.weld_fraction(),
            Phase::Partition => 0.1 + 0.1 * self.partition.progress(),
            Phase::Level => {
                let kept = self.first_active.saturating_sub(self.round_start) as f32;
                let level = self.level.progress() * self.round_start as f32;
                let share = (kept + level) / self.first_active.max(1) as f32;
                0.2 + 0.5 * share.min(1.0)
            }
            Phase::Sets => 0.7 + 0.1 * (2.0 * self.pack.progress()).min(1.0),
            Phase::Hierarchy => 0.8 + 0.05 * self.hierarchy.progress(),
            Phase::Clusters => 0.85 + 0.15 * (2.0 * self.pack.progress() - 1.0).clamp(0.0, 1.0),
            Phase::Done => 1.0,
        };
        let value = if self.phase == Phase::Done {
            1.0
        } else {
            raw.clamp(0.0, 0.999)
        };
        self.fraction = self.fraction.max(value);
    }

    fn weld_fraction(&self) -> f32 {
        let triangles = self.input.mesh.triangle_count().max(1) as f32;
        ((self.weld_steps * WELD_BATCH) as f32 / triangles).min(1.0)
    }

    /// Runs one unit of the current phase.
    fn unit(&mut self) {
        match self.phase {
            Phase::Weld | Phase::Reweld => self.weld_unit(),
            Phase::Search => self.search_unit(),
            Phase::Partition => self.partition_unit(),
            Phase::Level => self.level_unit(),
            Phase::Sets => self.sets_unit(),
            Phase::Hierarchy => self.hierarchy_unit(),
            Phase::Clusters => self.clusters_unit(),
            Phase::Done => {}
        }
    }

    /// One batch of the grid weld. The weld that ends is checked: no triangle is `cluster.empty`.
    fn weld_unit(&mut self) {
        self.weld_steps += 1;
        if !self.weld.step(&self.input) {
            return;
        }
        let weld = std::mem::replace(&mut self.weld, WeldState::new(&self.bounds, 0.0));
        self.welded = weld.finish();
        if self.welded.triangles.is_empty() {
            self.fail(ClusterError::new(
                EMPTY,
                "no triangle survives the grid weld",
            ));
        } else if self.phase == Phase::Reweld {
            self.start_partition();
        } else {
            self.phase = Phase::Search;
        }
    }

    /// The piece search over the welded mesh, one unit. Foliage made of separate pieces is welded
    /// again with a grid margin.
    fn search_unit(&mut self) {
        if find_pieces(&self.welded.triangles, &|_| false)
            .pieces
            .is_empty()
        {
            self.start_partition();
        } else {
            self.weld = WeldState::new(&self.bounds, FOLIAGE_GRID_MARGIN);
            self.weld_steps = 0;
            self.phase = Phase::Reweld;
        }
    }

    fn start_partition(&mut self) {
        self.partition = PartitionState::new(
            self.welded.triangles.len(),
            self.limits,
            self.welded.origin,
            self.welded.step,
        );
        self.partition_done = false;
        self.phase = Phase::Partition;
    }

    /// One partition unit, then the initial clusters of what it finished, pushed in bounded parts.
    fn partition_unit(&mut self) {
        if !self.partition_done && self.pending.is_empty() {
            self.partition_done = self.partition.step(
                &|index: u32| corners(&self.welded, index),
                &self.welded.positions,
            );
            let finished = self.partition.drain();
            self.pending.extend(finished);
        }
        self.push_pending();
        if self.partition_done && self.pending.is_empty() {
            self.start_level();
        }
    }

    /// Pushes pending partition clusters into the build list, at most `PUSH_CLUSTERS` clusters and
    /// `PUSH_TRIANGLES` triangles per unit (one cluster at least).
    fn push_pending(&mut self) {
        let (mut visited, mut triangles) = (0, 0);
        while let Some(next) = self.pending.front() {
            let count = next.len();
            if visited > 0 && (visited == PUSH_CLUSTERS || triangles + count > PUSH_TRIANGLES) {
                break;
            }
            let Some(cluster) = self.pending.pop_front() else {
                break;
            };
            push_initial_clusters(
                &self.welded,
                &cluster,
                self.options.max_cluster_vertices,
                &mut self.clusters,
            );
            (visited, triangles) = (visited + 1, triangles + count);
        }
    }

    /// Every pushed cluster is active at the first level.
    fn start_level(&mut self) {
        self.active = (0..self.clusters.len()).map(|index| index as u32).collect();
        self.first_active = self.active.len();
        self.start_round();
    }

    /// Starts the next level over `active`; with at most one active cluster the levels end.
    fn start_round(&mut self) {
        if self.active.len() <= 1 {
            self.start_sets();
            return;
        }
        self.level_number += 1;
        self.round_start = self.active.len();
        self.level = LevelState::new(std::mem::take(&mut self.active), self.level_number);
        self.phase = Phase::Level;
    }

    /// One unit of the current level. A finished level hands its surviving clusters to the next.
    fn level_unit(&mut self) {
        let done = self.level.step(
            &mut self.welded,
            &mut self.clusters,
            &mut self.groups,
            self.limits,
            &self.options,
        );
        if done {
            let level = std::mem::replace(&mut self.level, LevelState::new(Vec::new(), 0));
            self.active = level.finish();
            self.start_round();
        }
    }

    /// The welded triangle count is final here and is the packed source count.
    fn start_sets(&mut self) {
        self.pack = PackState::new(self.welded.triangles.len() as u64);
        self.phase = Phase::Sets;
    }

    fn sets_unit(&mut self) {
        if self.pack.step_sets(&self.clusters, &self.groups) {
            self.hierarchy = HierarchyState::new(self.pack.leaves().to_vec());
            self.phase = Phase::Hierarchy;
        }
    }

    fn hierarchy_unit(&mut self) {
        if self.hierarchy.step() {
            let hierarchy = std::mem::replace(&mut self.hierarchy, HierarchyState::new(Vec::new()));
            self.tree = hierarchy.finish();
            self.phase = Phase::Clusters;
        }
    }

    fn clusters_unit(&mut self) {
        match self.pack.step_clusters(&self.welded, &self.clusters) {
            Ok(false) => {}
            Ok(true) => {
                let pack = std::mem::replace(&mut self.pack, PackState::new(0));
                let tree = std::mem::replace(
                    &mut self.tree,
                    Hierarchy {
                        nodes: Vec::new(),
                        root: 0,
                        depth: 0,
                    },
                );
                self.outcome = Some(Ok(pack.finish(&self.welded, tree)));
                self.phase = Phase::Done;
            }
            Err(error) => self.fail(error),
        }
    }

    /// Ends the bake with an error; the finish returns it.
    fn fail(&mut self, error: ClusterError) {
        self.outcome = Some(Err(error));
        self.phase = Phase::Done;
    }
}

/// The position ids of welded triangle `index`; a missing triangle gives a degenerate one.
fn corners(welded: &WeldedMesh, index: u32) -> [u32; 3] {
    welded
        .triangles
        .get(index as usize)
        .map_or([0; 3], |triangle| triangle.map(|corner| corner.position))
}

/// Bakes `input` in one call: steps the bake to the end, then finishes it.
pub fn bake(input: ClusterInput<'_>, options: &BuildOptions) -> Result<ClusterMesh, ClusterError> {
    let mut bake = ClusterBake::new(input, options)?;
    while bake.step(WorkBudget { units: u32::MAX }) == BakeStatus::Working {}
    bake.finish()
}
