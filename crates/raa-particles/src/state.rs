//! The emitter table shared through `Rc<RefCell<..>>` by the `particles.sim` pass, the
//! opaque and blend passes and the `ParticleHandle`.
//!
//! # Ids and slot ranges
//!
//! [`EmitterTable::add`] hands out the lowest free id, so a removed id is reused before the
//! table grows. Every emitter owns one contiguous range of particle slots, found first fit
//! (lowest base that leaves `max_particles` free) and recorded in [`EmitterTable::slot_emitters`]
//! as a `u32` per slot ([`SLOT_FREE`] marks a hole). Compaction is never needed: a freed
//! range is simply available to the next allocation.
//!
//! The GPU tables (the [`EmitterGpu`] records and the slot -> emitter index table) are
//! marked dirty on every add, set and remove; `particles.sim` uploads them at its next
//! prepare and clears the flag.
//!
//! # Simulation clock
//!
//! The first [`EmitterTable::advance`] starts at `max(0, time_s - PREWARM_S)`, so a
//! fixed-time lookdev frame shows a populated state even though its `dt` is 0. Each later
//! advance runs `floor((time_s - sim_time) / STEP_S)` fixed steps (0 when the time repeats),
//! at most [`MAX_STEPS_PER_FRAME`]: a frame further behind jumps its clock forward and only
//! simulates the last `MAX_STEPS_PER_FRAME` steps.
//!
//! For every step and emitter, the spawn command is
//! `(next serial, floor(accumulated rate) + pending bursts)`; bursts are consumed by the
//! first step of the frame only. Commands live at `[step * MAX_EMITTERS + emitter id]`, so
//! an emitter hole holds `(0, 0)`. [`EmitterTable::count`] sums `min(max_particles, total
//! spawned)` without any GPU readback.

#![allow(dead_code)] // consumed by ParticleFeature (particles.sim, particles.opaque, particles.blend)

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::RaaError;

use crate::desc::EmitterDesc;
use crate::gpu::{
    EmitterGpu, MAX_EMITTERS, MAX_STEPS_PER_FRAME, PREWARM_S, STEP_S, TOTAL_CAPACITY_MAX,
};

/// Emitter index stored for a free particle slot.
pub(crate) const SLOT_FREE: u32 = u32::MAX;

/// `STEP_S` as the f64 clock step (the GPU receives the f32 `STEP_S` as `dt`).
const CLOCK_STEP_S: f64 = 1.0 / 60.0;

/// Tolerance against floating-point noise in a repeated or exactly stepped time (well
/// below a step, far above an f64 rounding error).
const CLOCK_EPSILON: f64 = 1e-6;

/// The shared emitter table; clones see the same emitters and clock.
pub(crate) type SharedEmitters = Rc<RefCell<EmitterTable>>;

/// One contiguous run of particle slots owned by emitter `id`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct SlotRange {
    id: u32,
    base: u32,
    len: u32,
}

/// One emitter and its simulation state.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EmitterEntry {
    /// The current description.
    pub(crate) desc: EmitterDesc,
    /// First particle slot of the emitter's contiguous range.
    pub(crate) slot_base: u32,
    /// Fractional spawn rate carried between steps, in `[0, 1)`.
    pub(crate) accumulator: f32,
    /// Burst particles requested but not spawned yet (consumed by the next frame's first
    /// step).
    pub(crate) pending_bursts: u32,
    /// Serial of the next sample to spawn.
    pub(crate) next_serial: u32,
    /// Samples spawned so far (saturating).
    pub(crate) spawned: u32,
}

/// The emitter table: ids, slot ranges, the GPU table bytes, the spawn commands and the
/// simulation clock.
pub(crate) struct EmitterTable {
    /// Index = emitter id; `None` is a free (reusable) id.
    entries: Vec<Option<EmitterEntry>>,
    /// GPU records, index = emitter id (a free id holds a zeroed record).
    records: Vec<EmitterGpu>,
    /// Live slot ranges, sorted by `base` and never overlapping.
    ranges: Vec<SlotRange>,
    /// Sum of the `max_particles` of the live emitters.
    capacity_used: u32,
    /// Slot -> emitter id, `SLOT_FREE` for a hole.
    slots: Vec<u32>,
    /// `(first serial, count)` at `[step * MAX_EMITTERS + emitter id]`.
    spawns: Vec<[u32; 2]>,
    /// Steps of the last [`EmitterTable::advance`].
    steps: usize,
    /// Simulation time before step 0 of the last advance.
    step_base_s: f64,
    /// Simulated time; `None` until the first advance.
    sim_time_s: Option<f64>,
    /// The GPU tables changed since the last upload.
    gpu_dirty: bool,
}

impl Default for EmitterTable {
    fn default() -> Self {
        EmitterTable {
            entries: Vec::new(),
            records: Vec::new(),
            ranges: Vec::new(),
            capacity_used: 0,
            slots: Vec::new(),
            spawns: Vec::new(),
            steps: 0,
            step_base_s: 0.0,
            sim_time_s: None,
            // The first frame uploads a zeroed table.
            gpu_dirty: true,
        }
    }
}

impl EmitterTable {
    /// An empty table (no emitters, GPU tables dirty).
    pub(crate) fn new() -> EmitterTable {
        EmitterTable::default()
    }

    /// An empty table behind the shared cell.
    pub(crate) fn shared() -> SharedEmitters {
        Rc::new(RefCell::new(EmitterTable::new()))
    }

    /// Adds `desc` under the lowest free id and allocates its first-fit slot range.
    /// Refusals (nothing changed, `desc` validated first): `particles.invalid` from
    /// [`EmitterDesc::validate`], `particles.capacity` when 64 emitters are live, when the
    /// sum of `max_particles` would pass [`TOTAL_CAPACITY_MAX`], or when no contiguous
    /// range is free.
    pub(crate) fn add(&mut self, desc: EmitterDesc) -> Result<u32, RaaError> {
        desc.validate()?;
        if self.live_count() >= MAX_EMITTERS {
            return Err(capacity(format!(
                "cannot add the emitter: {MAX_EMITTERS} emitters are already in the table"
            )));
        }
        let index = self
            .entries
            .iter()
            .position(Option::is_none)
            .unwrap_or(self.entries.len());
        if index >= MAX_EMITTERS {
            return Err(capacity(format!(
                "cannot add the emitter: no free id below {MAX_EMITTERS}"
            )));
        }
        let len = desc.max_particles;
        let Some(total) = self.capacity_used.checked_add(len) else {
            return Err(capacity_total(self.capacity_used, len));
        };
        if total > TOTAL_CAPACITY_MAX {
            return Err(capacity_total(self.capacity_used, len));
        }
        let Some(base) = first_fit(&self.ranges, len, None) else {
            return Err(capacity(format!(
                "cannot add an emitter of {len} particles: no contiguous free range"
            )));
        };
        let id = index as u32;
        let record = EmitterGpu::from_desc(&desc, base);
        let entry = EmitterEntry {
            desc,
            slot_base: base,
            accumulator: 0.0,
            pending_bursts: 0,
            next_serial: 0,
            spawned: 0,
        };
        if index == self.entries.len() {
            self.entries.push(Some(entry));
            self.records.push(record);
        } else {
            self.entries[index] = Some(entry);
            self.records[index] = record;
        }
        self.insert_range(SlotRange { id, base, len });
        self.capacity_used = total;
        self.rebuild_slots();
        self.gpu_dirty = true;
        Ok(id)
    }

    /// Replaces the description of `id`. The slot range is kept while `max_particles` does
    /// not change; otherwise it is reallocated first fit (the old range may be reused).
    /// Refusals (nothing changed): `particles.unknown_emitter`, `particles.invalid` and
    /// `particles.capacity` (new total past [`TOTAL_CAPACITY_MAX`], or no contiguous range).
    pub(crate) fn set(&mut self, id: u32, desc: EmitterDesc) -> Result<(), RaaError> {
        let Some(entry) = self.entries.get(id as usize).and_then(Option::as_ref) else {
            return Err(unknown_emitter(id));
        };
        desc.validate()?;
        let old_len = entry.desc.max_particles;
        let new_len = desc.max_particles;
        let old_base = entry.slot_base;
        let base = if new_len == old_len {
            old_base
        } else {
            let others = self.capacity_used - old_len;
            let Some(total) = others.checked_add(new_len) else {
                return Err(capacity_total(others, new_len));
            };
            if total > TOTAL_CAPACITY_MAX {
                return Err(capacity_total(others, new_len));
            }
            let Some(base) = first_fit(&self.ranges, new_len, Some(id)) else {
                return Err(capacity(format!(
                    "cannot resize emitter {id} to {new_len} particles: no contiguous free range"
                )));
            };
            self.ranges.retain(|r| r.id != id);
            self.insert_range(SlotRange {
                id,
                base,
                len: new_len,
            });
            self.capacity_used = total;
            self.rebuild_slots();
            base
        };
        let record = EmitterGpu::from_desc(&desc, base);
        if let Some(entry) = self.entries.get_mut(id as usize).and_then(Option::as_mut) {
            entry.desc = desc;
            entry.slot_base = base;
        }
        if let Some(slot) = self.records.get_mut(id as usize) {
            *slot = record;
        }
        self.gpu_dirty = true;
        Ok(())
    }

    /// Removes `id` and frees its slot range. Refuses `particles.unknown_emitter`.
    pub(crate) fn remove(&mut self, id: u32) -> Result<(), RaaError> {
        let index = id as usize;
        let Some(entry) = self.entries.get_mut(index).and_then(Option::take) else {
            return Err(unknown_emitter(id));
        };
        self.ranges.retain(|r| r.id != id);
        self.capacity_used -= entry.desc.max_particles;
        if let Some(record) = self.records.get_mut(index) {
            *record = EmitterGpu::default();
        }
        while self.entries.last().is_some_and(Option::is_none) {
            self.entries.pop();
            self.records.pop();
        }
        self.rebuild_slots();
        self.gpu_dirty = true;
        Ok(())
    }

    /// Adds `count` particles to the next frame's first step. Refuses
    /// `particles.unknown_emitter`.
    pub(crate) fn burst(&mut self, id: u32, count: u32) -> Result<(), RaaError> {
        let Some(entry) = self.entries.get_mut(id as usize).and_then(Option::as_mut) else {
            return Err(unknown_emitter(id));
        };
        entry.pending_bursts = entry.pending_bursts.saturating_add(count);
        Ok(())
    }

    /// Advances the fixed-step clock towards `time_s` and writes the spawn commands of the
    /// performed steps. Returns the number of steps (`0` for a repeated time).
    pub(crate) fn advance(&mut self, time_s: f32) -> usize {
        let time = f64::from(time_s);
        let start = match self.sim_time_s {
            Some(t) => t,
            None => (time - f64::from(PREWARM_S)).max(0.0),
        };
        let needed = if time > start {
            (((time - start) / CLOCK_STEP_S) + CLOCK_EPSILON).floor() as usize
        } else {
            0
        };
        let (base, steps) = if needed > MAX_STEPS_PER_FRAME {
            let skipped = (needed - MAX_STEPS_PER_FRAME) as f64;
            (start + skipped * CLOCK_STEP_S, MAX_STEPS_PER_FRAME)
        } else {
            (start, needed)
        };
        self.sim_time_s = Some(base + steps as f64 * CLOCK_STEP_S);
        self.step_base_s = base;
        self.steps = steps;

        self.spawns.clear();
        self.spawns.resize(steps * MAX_EMITTERS, [0, 0]);
        let entries = &mut self.entries;
        let spawns = &mut self.spawns;
        for step in 0..steps {
            for (id, slot) in entries.iter_mut().enumerate() {
                let Some(entry) = slot else { continue };
                entry.accumulator += entry.desc.rate_per_s * STEP_S;
                let whole = if entry.accumulator.is_finite() {
                    entry.accumulator.floor()
                } else {
                    entry.accumulator = 0.0;
                    0.0
                };
                let mut count = if whole > 0.0 { whole as u32 } else { 0 };
                entry.accumulator -= whole;
                if step == 0 {
                    count = count.saturating_add(entry.pending_bursts);
                    entry.pending_bursts = 0;
                }
                spawns[step * MAX_EMITTERS + id] = [entry.next_serial, count];
                entry.next_serial = entry.next_serial.wrapping_add(count);
                entry.spawned = entry.spawned.saturating_add(count);
            }
        }
        steps
    }

    /// Particle slots in use: the sum over emitters of `min(max_particles, total spawned)`.
    pub(crate) fn count(&self) -> u32 {
        self.entries.iter().flatten().fold(0u32, |sum, e| {
            sum.saturating_add(e.spawned.min(e.desc.max_particles))
        })
    }

    /// Live emitters.
    pub(crate) fn len(&self) -> usize {
        self.live_count()
    }

    /// True when no emitter is installed.
    pub(crate) fn is_empty(&self) -> bool {
        self.live_count() == 0
    }

    /// The id table; index = id, `None` = free id.
    pub(crate) fn entries(&self) -> &[Option<EmitterEntry>] {
        &self.entries
    }

    /// The live emitter `id`, if any.
    pub(crate) fn emitter(&self, id: u32) -> Option<&EmitterEntry> {
        self.entries.get(id as usize).and_then(Option::as_ref)
    }

    /// The GPU records; index = emitter id (holes are zeroed).
    pub(crate) fn emitter_records(&self) -> &[EmitterGpu] {
        &self.records
    }

    /// Slot -> emitter id, `SLOT_FREE` for a hole; length is the highest used slot.
    pub(crate) fn slot_emitters(&self) -> &[u32] {
        &self.slots
    }

    /// The last frame's spawn commands, `[step * MAX_EMITTERS + emitter id]`.
    pub(crate) fn spawn_commands(&self) -> &[[u32; 2]] {
        &self.spawns
    }

    /// Steps of the last [`EmitterTable::advance`].
    pub(crate) fn steps(&self) -> usize {
        self.steps
    }

    /// Simulation time before step 0 of the last advance (seconds).
    pub(crate) fn step_base_s(&self) -> f64 {
        self.step_base_s
    }

    /// Simulated time in seconds (0 before the first advance).
    pub(crate) fn sim_time_s(&self) -> f64 {
        self.sim_time_s.unwrap_or(0.0)
    }

    /// The GPU tables changed since the last upload?
    pub(crate) fn gpu_dirty(&self) -> bool {
        self.gpu_dirty
    }

    /// Reads and clears the GPU-table dirty flag.
    pub(crate) fn take_gpu_dirty(&mut self) -> bool {
        std::mem::take(&mut self.gpu_dirty)
    }

    fn live_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }

    fn insert_range(&mut self, range: SlotRange) {
        let at = self.ranges.partition_point(|r| r.base < range.base);
        self.ranges.insert(at, range);
    }

    /// Rebuilds the slot table from the live ranges (at most `TOTAL_CAPACITY_MAX` slots).
    fn rebuild_slots(&mut self) {
        let end = self
            .ranges
            .iter()
            .map(|r| r.base + r.len)
            .max()
            .unwrap_or(0);
        self.slots.clear();
        self.slots.resize(end as usize, SLOT_FREE);
        for range in &self.ranges {
            for slot in range.base..range.base + range.len {
                self.slots[slot as usize] = range.id;
            }
        }
    }
}

/// Lowest base of `len` contiguous slots, first fit, ignoring `skip_id`'s range (a set()
/// frees it conceptually). `None` when the free runs are all too small.
fn first_fit(ranges: &[SlotRange], len: u32, skip_id: Option<u32>) -> Option<u32> {
    let mut cursor = 0u32;
    for range in ranges {
        if Some(range.id) == skip_id {
            continue;
        }
        if range.base.saturating_sub(cursor) >= len {
            return Some(cursor);
        }
        cursor = cursor.max(range.base + range.len);
    }
    if TOTAL_CAPACITY_MAX.saturating_sub(cursor) >= len {
        Some(cursor)
    } else {
        None
    }
}

fn refusal(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

fn unknown_emitter(id: u32) -> RaaError {
    refusal(
        "particles.unknown_emitter",
        format!("emitter {id} is not in the table"),
    )
}

fn capacity(message: String) -> RaaError {
    refusal("particles.capacity", message)
}

fn capacity_total(used: u32, add: u32) -> RaaError {
    capacity(format!(
        "cannot allocate {add} particles on top of {used}: the table holds {TOTAL_CAPACITY_MAX}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::ParticleKind;

    fn desc(max_particles: u32) -> EmitterDesc {
        let mut d = EmitterDesc::for_kind(ParticleKind::Sparkles);
        d.rate_per_s = 0.0;
        d.max_particles = max_particles;
        d
    }

    fn rate_desc(rate_per_s: f32, max_particles: u32) -> EmitterDesc {
        let mut d = desc(max_particles);
        d.rate_per_s = rate_per_s;
        d
    }

    /// The whole mutable state of a table, for "unchanged" assertions.
    #[derive(Clone, Debug, PartialEq)]
    struct Snapshot {
        entries: Vec<Option<EmitterEntry>>,
        records: Vec<EmitterGpu>,
        ranges: Vec<SlotRange>,
        capacity_used: u32,
        slots: Vec<u32>,
        spawns: Vec<[u32; 2]>,
        count: u32,
        steps: usize,
        sim_time: f64,
        gpu_dirty: bool,
    }

    fn snapshot(t: &EmitterTable) -> Snapshot {
        Snapshot {
            entries: t.entries.clone(),
            records: t.records.clone(),
            ranges: t.ranges.clone(),
            capacity_used: t.capacity_used,
            slots: t.slots.clone(),
            spawns: t.spawns.clone(),
            count: t.count(),
            steps: t.steps,
            sim_time: t.sim_time_s(),
            gpu_dirty: t.gpu_dirty,
        }
    }

    fn assert_unchanged(t: &EmitterTable, before: &Snapshot) {
        assert_eq!(&snapshot(t), before, "the refusal changed the table");
    }

    #[test]
    fn ids_reuse_the_lowest_free() {
        let mut t = EmitterTable::new();
        assert_eq!(t.add(desc(8)).unwrap(), 0);
        assert_eq!(t.add(desc(8)).unwrap(), 1);
        assert_eq!(t.add(desc(8)).unwrap(), 2);
        t.remove(1).unwrap();
        assert_eq!(t.add(desc(8)).unwrap(), 1);
        t.remove(0).unwrap();
        assert_eq!(t.add(desc(8)).unwrap(), 0);
        t.remove(2).unwrap();
        assert_eq!(t.add(desc(8)).unwrap(), 2);
        assert_eq!(t.len(), 3);
        assert_eq!(t.entries().len(), 3);
    }

    #[test]
    fn capacity_refusal_leaves_the_table_unchanged() {
        let mut t = EmitterTable::new();
        for _ in 0..MAX_EMITTERS {
            t.add(desc(1)).unwrap();
        }
        let before = snapshot(&t);
        let err = t.add(desc(1)).unwrap_err();
        assert_eq!(err.code(), "particles.capacity", "{err}");
        assert!(!err.message().is_empty());
        assert_unchanged(&t, &before);

        // The sum of max_particles is capped at TOTAL_CAPACITY_MAX.
        let mut t = EmitterTable::new();
        for _ in 0..4 {
            t.add(desc(65_536)).unwrap();
        }
        let before = snapshot(&t);
        let err = t.add(desc(1)).unwrap_err();
        assert_eq!(err.code(), "particles.capacity", "{err}");
        assert_unchanged(&t, &before);

        // set() refuses a new max_particles that would pass the total.
        let mut t = EmitterTable::new();
        for _ in 0..3 {
            t.add(desc(65_536)).unwrap();
        }
        let small = t.add(desc(1_000)).unwrap();
        t.add(desc(1)).unwrap();
        let before = snapshot(&t);
        let err = t.set(small, desc(65_536)).unwrap_err();
        assert_eq!(err.code(), "particles.capacity", "{err}");
        assert_unchanged(&t, &before);
    }

    #[test]
    fn unknown_id_refusals_leave_the_table_unchanged() {
        let mut t = EmitterTable::new();
        let id = t.add(desc(16)).unwrap();
        t.burst(id, 2).unwrap();
        assert_eq!(t.advance(1.0), 60);
        let before = snapshot(&t);
        let err = t.set(id + 1, desc(16)).unwrap_err();
        assert_eq!(err.code(), "particles.unknown_emitter", "{err}");
        let err = t.remove(id + 1).unwrap_err();
        assert_eq!(err.code(), "particles.unknown_emitter", "{err}");
        let err = t.burst(id + 1, 1).unwrap_err();
        assert_eq!(err.code(), "particles.unknown_emitter", "{err}");
        assert_unchanged(&t, &before);

        // An invalid description on a known id is refused too.
        let mut bad = desc(16);
        bad.rate_per_s = f32::NAN;
        let err = t.set(id, bad).unwrap_err();
        assert_eq!(err.code(), "particles.invalid", "{err}");
        assert_unchanged(&t, &before);
    }

    #[test]
    fn fresh_table_steps_to_one_second_is_60() {
        let mut t = EmitterTable::new();
        assert_eq!(t.advance(1.0), 60);
        assert_eq!(t.steps(), 60);
    }

    #[test]
    fn repeated_time_gives_zero_steps() {
        let mut t = EmitterTable::new();
        assert_eq!(t.advance(1.0), 60);
        let time = t.sim_time_s();
        assert_eq!(t.advance(1.0), 0);
        assert!(t.spawn_commands().is_empty());
        assert_eq!(t.sim_time_s(), time);
    }

    /// The m13 mud set's splash crown emitter (opaque chocolate blobs, see
    /// examples/lookdev/scenes/m13-liquids-particles.js): it spawns particles by the capture
    /// time and carries its body opacity into the GPU record.
    #[test]
    fn m13_mud_crown_emitter_spawns_opaque_droplets() {
        let mut d = EmitterDesc::for_kind(ParticleKind::Splash);
        d.position = [-1.45 + 0.68, 0.04, -0.55];
        d.direction = [0.45, 1.0, 0.0];
        d.spread_rad = 0.3;
        d.rate_per_s = 45.0;
        d.lifetime_s = [0.25, 0.5];
        d.speed = [1.0, 2.1];
        d.size = [0.07, 0.13];
        d.drag = 0.4;
        d.colors = vec![[0.147, 0.04, 0.008]];
        d.opacity = 1.0;
        d.volume_half = [0.03, 0.01, 0.03];
        d.max_particles = 64;
        let mut t = EmitterTable::new();
        let id = t.add(d.clone()).unwrap();
        assert_eq!(t.advance(1.0), 60);
        let spawned: u32 = (0..60)
            .map(|step| t.spawn_commands()[step * MAX_EMITTERS + id as usize][1])
            .sum();
        assert!(spawned > 0, "the crown emitter spawned nothing");
        assert_eq!(spawned, 45);
        assert!(t.count() > 0);
        assert_eq!(t.emitter_records()[id as usize].body[0], 1.0);
        // Every droplet leaves upward and is at least 7 cm across.
        for serial in 0..spawned {
            let s = crate::spawn::spawn_sample(&d, serial);
            assert!(s.velocity[1] > 0.0, "serial {serial}");
            assert!(s.size >= 0.07, "serial {serial}");
        }
    }

    #[test]
    fn rate_of_120_over_one_second_spawns_120() {
        let mut t = EmitterTable::new();
        let id = t.add(rate_desc(120.0, 2048)).unwrap();
        assert_eq!(t.advance(1.0), 60);
        assert_eq!(t.count(), 120);
        let mut serial = 0;
        for step in 0..60 {
            let cmd = t.spawn_commands()[step * MAX_EMITTERS + id as usize];
            assert_eq!(cmd, [serial, 2], "step {step}");
            serial += 2;
        }
    }

    #[test]
    fn count_caps_at_max_particles() {
        let mut t = EmitterTable::new();
        t.add(rate_desc(120.0, 100)).unwrap();
        assert_eq!(t.advance(1.0), 60);
        assert_eq!(t.count(), 100);
    }

    #[test]
    fn burst_lands_on_the_first_step() {
        let mut t = EmitterTable::new();
        let id = t.add(desc(64)).unwrap();
        t.burst(id, 5).unwrap();
        assert_eq!(t.advance(1.0), 60);
        let e = id as usize;
        assert_eq!(t.spawn_commands()[e], [0, 5]);
        assert_eq!(t.spawn_commands()[MAX_EMITTERS + e], [5, 0]);
        assert_eq!(t.count(), 5);

        // A burst waits through a 0-step frame and lands on the next stepped frame.
        t.burst(id, 3).unwrap();
        assert_eq!(t.advance(1.0), 0);
        assert_eq!(t.count(), 5);
        assert_eq!(t.advance(1.02), 1);
        assert_eq!(t.spawn_commands()[e], [5, 3]);
        assert_eq!(t.count(), 8);
    }

    #[test]
    fn first_advance_prewarms_at_four_seconds() {
        let mut t = EmitterTable::new();
        t.add(desc(8)).unwrap();
        assert_eq!(t.advance(10.0), MAX_STEPS_PER_FRAME);
        assert!((t.step_base_s() - 6.0).abs() < 1e-6);
        assert_eq!(t.advance(10.0), 0);

        // A frame far behind jumps forward instead of simulating every step.
        assert_eq!(t.advance(1000.0), MAX_STEPS_PER_FRAME);
    }

    #[test]
    fn slots_are_first_fit_and_contiguous() {
        let mut t = EmitterTable::new();
        let a = t.add(desc(4)).unwrap();
        assert_eq!(t.emitter(a).unwrap().slot_base, 0);
        assert_eq!(t.slot_emitters().to_vec(), vec![a; 4]);
        let b = t.add(desc(2)).unwrap();
        assert_eq!(t.emitter(b).unwrap().slot_base, 4);
        t.remove(a).unwrap();
        assert_eq!(
            t.slot_emitters().to_vec(),
            vec![SLOT_FREE, SLOT_FREE, SLOT_FREE, SLOT_FREE, b, b]
        );
        let c = t.add(desc(3)).unwrap();
        assert_eq!(t.emitter(c).unwrap().slot_base, 0);
        assert_eq!(t.slot_emitters().to_vec(), vec![c, c, c, SLOT_FREE, b, b]);
    }

    #[test]
    fn set_reallocates_only_when_max_particles_changes() {
        let mut t = EmitterTable::new();
        let a = t.add(desc(4)).unwrap();
        let b = t.add(desc(4)).unwrap();
        t.set(a, desc(4)).unwrap();
        assert_eq!(t.emitter(a).unwrap().slot_base, 0);
        t.set(a, desc(2)).unwrap();
        assert_eq!(t.emitter(a).unwrap().slot_base, 0);
        assert_eq!(
            t.slot_emitters().to_vec(),
            vec![a, a, SLOT_FREE, SLOT_FREE, b, b, b, b]
        );
        assert_eq!(t.capacity_used, 6);

        t.set(a, desc(8)).unwrap();
        assert_eq!(t.emitter(a).unwrap().slot_base, 8);
        assert_eq!(t.capacity_used, 12);
        assert_eq!(t.slot_emitters().to_vec()[0..4], [SLOT_FREE; 4]);
        assert_eq!(t.slot_emitters().to_vec()[4..8], [b; 4]);
        assert_eq!(t.slot_emitters().to_vec()[8..16], [a; 8]);

        // An invalid description does not reallocate or replace anything.
        let before = snapshot(&t);
        let mut bad = desc(16);
        bad.size = [2.0, 1.0];
        let err = t.set(a, bad).unwrap_err();
        assert_eq!(err.code(), "particles.invalid", "{err}");
        assert_unchanged(&t, &before);
    }

    #[test]
    fn remove_frees_the_range_and_reuses_it() {
        let mut t = EmitterTable::new();
        let a = t.add(desc(4)).unwrap();
        t.add(desc(2)).unwrap();
        t.remove(a).unwrap();
        assert_eq!(t.capacity_used, 2);
        assert_eq!(
            t.slot_emitters().to_vec(),
            vec![SLOT_FREE, SLOT_FREE, SLOT_FREE, SLOT_FREE, 1, 1]
        );
        let c = t.add(desc(4)).unwrap();
        assert_eq!(c, 0);
        assert_eq!(t.emitter(c).unwrap().slot_base, 0);
        assert_eq!(t.capacity_used, 6);
    }

    #[test]
    fn emitter_holes_spawn_nothing() {
        let mut t = EmitterTable::new();
        let a = t.add(rate_desc(60.0, 32)).unwrap();
        let b = t.add(rate_desc(60.0, 32)).unwrap();
        t.remove(a).unwrap();
        assert_eq!(t.advance(1.0), 60);
        assert_eq!(t.spawn_commands()[0], [0, 0]);
        assert_eq!(t.spawn_commands()[b as usize], [0, 1]);
        assert_eq!(t.spawn_commands()[MAX_EMITTERS], [0, 0]);
        assert_eq!(t.spawn_commands()[MAX_EMITTERS + b as usize], [1, 1]);
    }

    #[test]
    fn gpu_tables_are_marked_dirty() {
        let mut t = EmitterTable::new();
        assert!(t.take_gpu_dirty());
        assert!(!t.take_gpu_dirty());
        let id = t.add(desc(8)).unwrap();
        assert!(t.gpu_dirty());
        assert!(t.take_gpu_dirty());
        assert_eq!(t.advance(1.0), 60);
        assert!(!t.gpu_dirty(), "the simulation does not rewrite the tables");
        t.set(id, desc(8)).unwrap();
        assert!(t.take_gpu_dirty());
        t.remove(id).unwrap();
        assert!(t.take_gpu_dirty());
    }
}
