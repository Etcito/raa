//! Mutable S19 execution state: machine cursors, local clocks, phases,
//! crossfades and pending triggers. Kept outside [`Compiled`] so one compiled
//! graph drives many independent states.
use crate::error::{GraphError, codes};
use crate::load::Compiled;
use crate::params::Params;
use raa_det::Tick;
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crossfade {
    pub from: usize,
    /// Frozen source progress, in the source state's `motion` units.
    pub from_phase: f32,
    pub elapsed: Tick,
    pub ticks: Tick,
}
impl Crossfade {
    /// Destination weight; zero-duration fades are immediately complete.
    pub fn weight(&self) -> f32 {
        if self.ticks == 0 || self.elapsed >= self.ticks { 1.0 }
        else { self.elapsed as f32 / self.ticks as f32 }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct LayerState {
    pub machine: usize,
    pub state: usize,
    pub clock: Tick,
    /// Current-state progress (clip ticks, or normalized for sync spaces); see `motion`.
    pub phase: f32,
    pub fade: Option<Crossfade>,
}
/// All future-affecting execution data; `layers` mirrors `Graph::layers`.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphState {
    pub tick: Tick,
    pub layers: Vec<LayerState>,
    pub pending: Vec<bool>,
    /// Last applied inputs (including trigger pulses); `pending` tracks consumption separately.
    pub params: Params,
    pub(crate) binding: Vec<u8>,
}
fn bad(message: &str) -> GraphError {
    GraphError::new(codes::SCHEMA, "/state", message)
}
fn word(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn next(words: &mut std::slice::ChunksExact<'_, u8>) -> Result<u32, GraphError> {
    words.next().map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| bad("snapshot is truncated"))
}
fn boolean(word: u32) -> Result<bool, GraphError> {
    match word { 0 => Ok(false), 1 => Ok(true), _ => Err(bad("invalid snapshot bool")) }
}
impl GraphState {
    /// Entry state, zeroed clocks and no pending triggers.
    pub fn new(compiled: &Compiled) -> Self {
        let layers = compiled.references.layers.iter().map(|layer| LayerState {
            machine: layer.machine,
            state: compiled.references.machines[layer.machine].entry,
            clock: 0,
            phase: 0.0,
            fade: None,
        }).collect();
        Self { tick: 0, layers, pending: vec![false; compiled.graph.params.len()],
            params: Params::new(&compiled.graph), binding: crate::snapshot::binding(compiled) }
    }
    /// Fold one tick of fired triggers into the pending state.
    pub fn apply(&mut self, inputs: &Params) {
        self.params = inputs.clone();
        for (index, pending) in self.pending.iter_mut().enumerate() {
            *pending |= inputs.pending(index);
        }
    }
    /// Append every future-affecting field as little-endian 32-bit words.
    pub(crate) fn write_payload(&self, out: &mut Vec<u8>) {
        word(out, u32::from_le_bytes(*b"RGS1"));
        word(out, self.tick);
        word(out, self.layers.len() as u32);
        for layer in &self.layers {
            word(out, layer.state as u32);
            word(out, layer.clock);
            word(out, layer.phase.to_bits());
            match layer.fade {
                None => for _ in 0..5 { word(out, 0); },
                Some(fade) => {
                    word(out, 1);
                    word(out, fade.from as u32);
                    word(out, fade.from_phase.to_bits());
                    word(out, fade.elapsed);
                    word(out, fade.ticks);
                }
            }
        }
        word(out, self.pending.len() as u32);
        for &pending in &self.pending { word(out, pending as u32); }
    }
    /// Restore a snapshot, validating it against `compiled`.
    pub(crate) fn read_payload(compiled: &Compiled, bytes: &[u8]) -> Result<Self, GraphError> {
        if bytes.len() % 4 != 0 { return Err(bad("snapshot length is not word aligned")); }
        let mut words = bytes.chunks_exact(4);
        if next(&mut words)? != u32::from_le_bytes(*b"RGS1") {
            return Err(bad("unknown snapshot format"));
        }
        let tick = next(&mut words)?;
        let count = next(&mut words)? as usize;
        if count != compiled.graph.layers.len() { return Err(bad("snapshot layer count mismatch")); }
        let mut layers = Vec::with_capacity(count);
        for expected in &compiled.references.layers {
            let state = next(&mut words)? as usize;
            let clock = next(&mut words)?;
            let phase = f32::from_bits(next(&mut words)?);
            let fade = match next(&mut words)? {
                0 => { for _ in 0..4 {
                    if next(&mut words)? != 0 { return Err(bad("nonzero absent fade data")); }
                } None }
                1 => Some(Crossfade {
                    from: next(&mut words)? as usize,
                    from_phase: f32::from_bits(next(&mut words)?),
                    elapsed: next(&mut words)?,
                    ticks: next(&mut words)?,
                }),
                _ => return Err(bad("invalid crossfade flag")),
            };
            let states = &compiled.graph.machines[expected.machine].states;
            let progress = |p: f32| p.is_finite() && p >= 0.0;
            let invalid = state >= states.len() || !progress(phase)
                || fade.is_some_and(|f| f.from >= states.len() || !progress(f.from_phase)
                    || f.ticks == 0 || f.elapsed >= f.ticks);
            if invalid { return Err(bad("invalid layer snapshot")); }
            layers.push(LayerState { machine: expected.machine, state, clock, phase, fade });
        }
        let count = next(&mut words)? as usize;
        if count != compiled.graph.params.len() { return Err(bad("snapshot parameter count mismatch")); }
        let mut pending = Vec::with_capacity(count);
        for param in &compiled.graph.params {
            let value = boolean(next(&mut words)?)?;
            if value && param.ty != crate::ParamType::Trigger { return Err(bad("non-trigger pending")); }
            pending.push(value);
        }
        if words.next().is_some() { return Err(bad("snapshot has trailing words")); }
        Ok(Self { tick, layers, pending, params: Params::new(&compiled.graph),
            binding: crate::snapshot::binding(compiled) })
    }
}
