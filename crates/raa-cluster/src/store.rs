//! CPU registry of cluster sections per mesh key, shared by the bake API and raa-vg: attached
//! sections, the queue of sliced runtime bakes, readiness states and the generation counter.
//! The crate's own code: this module has no solstice source.

use std::collections::BTreeMap;

use crate::bake::{BakeStage, ClusterBake, WorkBudget};
use crate::cut::ClusterMesh;
use crate::error::ClusterError;
use crate::input::{BuildOptions, OwnedClusterInput};

/// Readiness of one key's cluster data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterState {
    None,
    Baking,
    Ready,
    Refused,
}

/// Where a ready set came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterSource {
    Attached,
    Runtime,
}

/// The state of one key as the caller sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshClusterStatus {
    pub state: ClusterState,
    pub progress: f32,
    pub source: Option<ClusterSource>,
    pub code: Option<&'static str>,
    pub message: Option<String>,
}

enum Entry {
    Ready {
        sections: Vec<ClusterMesh>,
        source: ClusterSource,
    },
    Baking {
        bakes: Vec<ClusterBake>,
    },
    Refused {
        code: &'static str,
        message: String,
    },
}

/// The registry. Keys are the caller's; a key holds at most one entry, in one of the states above.
#[derive(Default)]
pub struct ClusterStore {
    entries: BTreeMap<u32, Entry>,
    generation: u64,
}

impl ClusterStore {
    /// An empty registry at generation 0.
    pub fn new() -> ClusterStore {
        ClusterStore::default()
    }

    /// Replaces whatever the key held and cancels its bake; the sections are Ready and attached.
    /// The caller has validated them.
    pub fn attach(&mut self, key: u32, sections: Vec<ClusterMesh>) {
        self.put(
            key,
            Entry::Ready {
                sections,
                source: ClusterSource::Attached,
            },
        );
    }

    /// Starts one bake per input. Any error is returned and nothing changes; otherwise the key's
    /// previous data is dropped and it is Baking from the runtime.
    pub fn start_bake(
        &mut self,
        key: u32,
        inputs: Vec<OwnedClusterInput>,
        options: &BuildOptions,
    ) -> Result<(), ClusterError> {
        let bakes = inputs
            .iter()
            .map(|input| ClusterBake::new(input.as_input(), options))
            .collect::<Result<Vec<_>, _>>()?;
        self.put(key, Entry::Baking { bakes });
        Ok(())
    }

    /// Drops the key's data and cancels its bake: state None.
    pub fn clear(&mut self, key: u32) {
        if matches!(self.entries.remove(&key), Some(Entry::Ready { .. })) {
            self.generation += 1;
        }
    }

    /// Records a refusal with its code and message.
    pub fn refuse(&mut self, key: u32, code: &'static str, message: String) {
        self.put(key, Entry::Refused { code, message });
    }

    /// The key's state; an unknown key is None with progress 0.
    pub fn state(&self, key: u32) -> MeshClusterStatus {
        let (state, progress, source, code, message) = match self.entries.get(&key) {
            None => (ClusterState::None, 0.0, None, None, None),
            Some(Entry::Baking { bakes }) => (
                ClusterState::Baking,
                mean_fraction(bakes),
                Some(ClusterSource::Runtime),
                None,
                None,
            ),
            Some(Entry::Ready { source, .. }) => {
                (ClusterState::Ready, 1.0, Some(*source), None, None)
            }
            Some(Entry::Refused { code, message }) => (
                ClusterState::Refused,
                0.0,
                None,
                Some(*code),
                Some(message.clone()),
            ),
        };
        MeshClusterStatus {
            state,
            progress,
            source,
            code,
            message,
        }
    }

    /// The sections of a Ready key, in the caller's order.
    pub fn sections(&self, key: u32) -> Option<&[ClusterMesh]> {
        match self.entries.get(&key) {
            Some(Entry::Ready { sections, .. }) => Some(sections.as_slice()),
            _ => None,
        }
    }

    /// The number of keys in state Baking.
    pub fn baking(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| matches!(entry, Entry::Baking { .. }))
            .count()
    }

    /// Runs units while the clock allows: before each unit the clock is asked, and a false answer
    /// stops the call. Units go to the lowest baking key, then to its first unfinished bake; a key
    /// whose bakes are all done is finished without a unit.
    pub fn step_all(&mut self, unit_clock: &mut dyn FnMut() -> bool) {
        while let Some(key) = self.lowest_baking() {
            if !self.has_unfinished(key) {
                self.finish(key);
                continue;
            }
            if !unit_clock() {
                return;
            }
            if let Some(Entry::Baking { bakes }) = self.entries.get_mut(&key)
                && let Some(bake) = bakes.iter_mut().find(|bake| !is_done(bake))
            {
                bake.step(WorkBudget { units: 1 });
            }
        }
    }

    /// Increases by one whenever a Ready set appears, is replaced or is removed.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn put(&mut self, key: u32, entry: Entry) {
        let was_ready = matches!(self.entries.get(&key), Some(Entry::Ready { .. }));
        let is_ready = matches!(entry, Entry::Ready { .. });
        if was_ready || is_ready {
            self.generation += 1;
        }
        self.entries.insert(key, entry);
    }

    fn lowest_baking(&self) -> Option<u32> {
        self.entries
            .iter()
            .find_map(|(key, entry)| matches!(entry, Entry::Baking { .. }).then_some(*key))
    }

    fn has_unfinished(&self, key: u32) -> bool {
        match self.entries.get(&key) {
            Some(Entry::Baking { bakes }) => bakes.iter().any(|bake| !is_done(bake)),
            _ => false,
        }
    }

    /// Finishes every bake of a key in order. The first failure refuses the key; otherwise the
    /// sections become Ready from the runtime.
    fn finish(&mut self, key: u32) {
        let Some(Entry::Baking { bakes }) = self.entries.remove(&key) else {
            return;
        };
        let mut sections = Vec::with_capacity(bakes.len());
        let mut failure: Option<ClusterError> = None;
        for bake in bakes {
            match bake.finish() {
                Ok(mesh) => sections.push(mesh),
                Err(err) => {
                    if failure.is_none() {
                        failure = Some(err);
                    }
                }
            }
        }
        let entry = match failure {
            Some(err) => Entry::Refused {
                code: err.code,
                message: err.message,
            },
            None => Entry::Ready {
                sections,
                source: ClusterSource::Runtime,
            },
        };
        self.put(key, entry);
    }
}

fn is_done(bake: &ClusterBake) -> bool {
    bake.progress().stage == BakeStage::Done
}

/// The mean fraction of a key's bakes; below 1 while any of them works, 0 without bakes.
fn mean_fraction(bakes: &[ClusterBake]) -> f32 {
    if bakes.is_empty() {
        return 0.0;
    }
    let sum: f64 = bakes
        .iter()
        .map(|bake| f64::from(bake.progress().fraction))
        .sum();
    (sum / bakes.len() as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{EMPTY, INVALID_MESH};
    use raa_geom::MeshData;

    fn options() -> BuildOptions {
        BuildOptions::default()
    }

    /// Calls `step_all` with a clock that allows `units` units.
    fn step(store: &mut ClusterStore, units: u32) {
        let mut left = units;
        store.step_all(&mut || {
            let more = left > 0;
            left = left.saturating_sub(1);
            more
        });
    }

    #[test]
    fn an_unknown_key_is_none_with_no_progress() {
        let store = ClusterStore::new();
        assert_eq!(
            store.state(9),
            MeshClusterStatus {
                state: ClusterState::None,
                progress: 0.0,
                source: None,
                code: None,
                message: None,
            }
        );
        assert!(store.sections(9).is_none());
        assert_eq!(store.generation(), 0);
    }

    #[test]
    fn attach_is_ready_attached_and_bumps_the_generation() {
        let mut store = ClusterStore::new();
        store.attach(4, vec![ClusterMesh::default()]);
        let status = store.state(4);
        assert_eq!(status.state, ClusterState::Ready);
        assert_eq!(status.progress, 1.0);
        assert_eq!(status.source, Some(ClusterSource::Attached));
        assert_eq!(store.sections(4).map(<[ClusterMesh]>::len), Some(1));
        assert_eq!(store.generation(), 1);
        store.attach(4, Vec::new());
        assert_eq!(store.sections(4).map(<[ClusterMesh]>::len), Some(0));
        assert_eq!(store.generation(), 2);
    }

    #[test]
    fn start_bake_is_baking_below_one_with_no_sections() {
        let mut store = ClusterStore::new();
        let input = OwnedClusterInput::displaced_sphere(2);
        store.start_bake(1, vec![input], &options()).expect("bake");
        let status = store.state(1);
        assert_eq!(status.state, ClusterState::Baking);
        assert!(status.progress < 1.0);
        assert_eq!(status.source, Some(ClusterSource::Runtime));
        assert!(store.sections(1).is_none());
        assert_eq!(store.baking(), 1);
        assert_eq!(store.generation(), 0);
    }

    #[test]
    fn step_all_advances_monotonically_and_ends_ready_once() {
        let mut store = ClusterStore::new();
        let input = OwnedClusterInput::displaced_sphere(3);
        store.start_bake(1, vec![input], &options()).expect("bake");
        let mut last = store.state(1).progress;
        let mut calls = 0;
        while store.state(1).state == ClusterState::Baking {
            step(&mut store, 5);
            let now = store.state(1).progress;
            assert!(now >= last, "progress fell from {last} to {now}");
            last = now;
            calls += 1;
            assert!(calls < 10_000, "the bake did not finish");
        }
        assert!(
            calls > 1,
            "one call of five units should not finish the bake"
        );
        let status = store.state(1);
        assert_eq!(status.source, Some(ClusterSource::Runtime));
        assert_eq!(store.sections(1).map(<[ClusterMesh]>::len), Some(1));
        assert_eq!(store.generation(), 1);
        assert_eq!(store.baking(), 0);
    }

    #[test]
    fn the_lower_key_is_ready_first() {
        let mut store = ClusterStore::new();
        let big = OwnedClusterInput::displaced_sphere(2);
        let small = OwnedClusterInput::single_triangle();
        store.start_bake(2, vec![big], &options()).expect("bake");
        store.start_bake(9, vec![small], &options()).expect("bake");
        while store.state(2).state == ClusterState::Baking
            || store.state(9).state == ClusterState::Baking
        {
            step(&mut store, 1);
            if store.state(9).state == ClusterState::Ready {
                assert_eq!(
                    store.state(2).state,
                    ClusterState::Ready,
                    "key 9 before key 2"
                );
            }
        }
        assert_eq!(store.state(2).state, ClusterState::Ready);
        assert_eq!(store.state(9).state, ClusterState::Ready);
    }

    #[test]
    fn a_clock_that_refuses_at_once_runs_no_unit() {
        let mut store = ClusterStore::new();
        let input = OwnedClusterInput::displaced_sphere(2);
        store.start_bake(1, vec![input], &options()).expect("bake");
        let before = store.state(1).progress;
        store.step_all(&mut || false);
        let status = store.state(1);
        assert_eq!(status.state, ClusterState::Baking);
        assert_eq!(status.progress, before);
    }

    #[test]
    fn an_invalid_input_leaves_a_ready_entry_untouched() {
        let mut store = ClusterStore::new();
        store.attach(7, vec![ClusterMesh::default()]);
        let mesh = MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![0, 1, 9],
        };
        let input = OwnedClusterInput { mesh, uvs: None };
        let err = store
            .start_bake(7, vec![input], &options())
            .expect_err("an out-of-range index is refused");
        assert_eq!(err.code, INVALID_MESH);
        let status = store.state(7);
        assert_eq!(status.state, ClusterState::Ready);
        assert_eq!(status.source, Some(ClusterSource::Attached));
        assert_eq!(store.generation(), 1);
    }

    #[test]
    fn a_bake_ending_empty_is_refused_with_its_code() {
        // Three corners within 1e-10 of each other and a far fourth point: the triangle collapses.
        let mesh = MeshData {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0e-10, 0.0, 0.0],
                [0.0, 1.0e-10, 0.0],
                [1.0, 1.0, 1.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            indices: vec![0, 1, 2],
        };
        let input = OwnedClusterInput { mesh, uvs: None };
        let mut store = ClusterStore::new();
        store.start_bake(4, vec![input], &options()).expect("bake");
        store.step_all(&mut || true);
        let status = store.state(4);
        assert_eq!(status.state, ClusterState::Refused);
        assert_eq!(status.code, Some(EMPTY));
        assert!(status.message.is_some_and(|m| !m.is_empty()));
        assert!(store.sections(4).is_none());
        assert_eq!(store.generation(), 0);
    }

    #[test]
    fn clear_cancels_a_bake_and_returns_to_none() {
        let mut store = ClusterStore::new();
        let input = OwnedClusterInput::displaced_sphere(2);
        store.start_bake(3, vec![input], &options()).expect("bake");
        step(&mut store, 2);
        store.clear(3);
        assert_eq!(store.state(3).state, ClusterState::None);
        assert_eq!(store.state(3).progress, 0.0);
        assert_eq!(store.baking(), 0);
        store.step_all(&mut || true);
        assert_eq!(store.state(3).state, ClusterState::None);
        assert_eq!(store.generation(), 0);
    }

    #[test]
    fn refuse_records_the_code_and_the_message() {
        let mut store = ClusterStore::new();
        store.refuse(5, EMPTY, "nothing survives".to_string());
        let status = store.state(5);
        assert_eq!(status.state, ClusterState::Refused);
        assert_eq!(status.code, Some(EMPTY));
        assert_eq!(status.message.as_deref(), Some("nothing survives"));
        assert_eq!(status.progress, 0.0);
        assert!(store.sections(5).is_none());
    }
}
