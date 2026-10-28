//! The public deterministic S19 tick: [`GraphRuntime::step`] is the contract's
//! `Graph::step(&mut state, &params) -> TickOut`, bound to the rig and clips the
//! graph was loaded with. Every call runs, in this order:
//!
//! 1. Parameter application: trigger pulses in `params` fold into
//!    `GraphState::pending`; float, int and bool values are read as given.
//! 2. Transitions: each machine evaluates once, current state then `*`, by
//!    priority then file order, gated on the pre-tick local clock. Existing
//!    fades advance one tick first; a firing transition enters its destination
//!    at clock and phase zero with a new fade at elapsed zero.
//! 3. Motion advancement: every layer not entered this tick (and not on the
//!    state's first tick) advances its current-state phase by this tick's speed;
//!    fading sources stay frozen.
//! 4. Crossfades: destination weight `elapsed / blendTicks`, so an entry tick
//!    shows the frozen source and the fade completes `blendTicks` calls later.
//! 5. Layers: composed in document order over the rig rest pose, with masks,
//!    weights and additive first-key deltas.
//! 6. Outputs: the root delta of the clip ticks crossed this tick, extracted axes
//!    restored to rest, and events in (previous sample, current sample]; an
//!    entered state reports zero delta and its tick-0 events.
//! 7. Matrices: S16 model and skin matrices through `Rig::pose`; the returned
//!    local pose is the one those matrices were built from.
//!
//! Inputs are validated before any state changes, so a refused call leaves the
//! state untouched. Equal states and inputs always give equal outputs.
use crate::load::Compiled;
use crate::{Graph, GraphError, GraphState, Params, TickOut, codes};
use raa_anim::{Clip, Mat4, Rig};

/// One tick's outputs plus the S16 matrices of its pose, in rig bone order.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub out: TickOut,
    pub model: Vec<Mat4>,
    pub skin: Vec<Mat4>,
}

/// A compiled graph with the rig and clips its references index.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphRuntime {
    compiled: Compiled,
    rig: Rig,
    clips: Vec<Clip>,
}

fn schema(path: &str, message: &str) -> GraphError {
    GraphError::new(codes::SCHEMA, path, message)
}

impl GraphRuntime {
    /// `Graph::load` keeping the compiled tables, rig and clips for stepping.
    pub fn load(text: &str, rig: &Rig, clips: &[Clip]) -> Result<Self, GraphError> {
        let compiled = Graph::load_compiled(text, rig, clips)?;
        Ok(Self { compiled, rig: rig.clone(), clips: clips.to_vec() })
    }
    pub fn compiled(&self) -> &Compiled { &self.compiled }
    pub fn graph(&self) -> &Graph { &self.compiled.graph }
    pub fn rig(&self) -> &Rig { &self.rig }
    pub fn clips(&self) -> &[Clip] { &self.clips }
    /// `GraphState::new` for this graph.
    pub fn new_state(&self) -> GraphState { GraphState::new(&self.compiled) }
    /// Declared parameter defaults for this graph.
    pub fn new_params(&self) -> Params { Params::new(&self.compiled.graph) }
    /// `GraphState::read` validated against this graph.
    pub fn read_state(&self, bytes: &[u8]) -> Result<GraphState, GraphError> {
        GraphState::read(&self.compiled, bytes)
    }

    /// Refuse a state or parameter set that does not belong to this graph.
    pub fn check(&self, state: &GraphState, params: &Params) -> Result<(), GraphError> {
        let graph = &self.compiled.graph;
        let same = params.len() == graph.params.len()
            && params.iter().zip(&graph.params).enumerate()
                .all(|(i, ((name, _), p))| name == p.name && params.ty(i) == Some(p.ty));
        if !same { return Err(GraphError::new(codes::PARAM, "/params", "parameters belong to another graph")); }
        if state.pending.len() != graph.params.len() {
            return Err(schema("/state/pending", "pending trigger count mismatch"));
        }
        if state.layers.len() != graph.layers.len() {
            return Err(schema("/state/layers", "layer state count mismatch"));
        }
        for (i, (layer, refs)) in state.layers.iter().zip(&self.compiled.references.layers).enumerate() {
            let states = graph.machines.get(layer.machine).map_or(0, |m| m.states.len());
            let valid = layer.machine == refs.machine && layer.state < states
                && layer.fade.is_none_or(|f| f.from < states);
            if !valid { return Err(schema(&format!("/state/layers/{i}"), "layer state does not match the graph")); }
        }
        Ok(())
    }

    /// Advance `state` by exactly one tick in the documented order.
    pub fn step(&self, state: &mut GraphState, params: &Params) -> Result<TickOut, GraphError> {
        self.frame(state, params).map(|frame| frame.out)
    }

    /// [`GraphRuntime::step`] with the S16 model and skin matrices of the pose.
    /// On an error after validation the state keeps its value from before the call.
    pub fn frame(&self, state: &mut GraphState, params: &Params) -> Result<Frame, GraphError> {
        self.check(state, params)?;
        let mut next = state.clone();
        let mut out = self.compiled.step_outputs(&self.rig, &self.clips, &mut next, params)?;
        let pose = self.rig.pose(&out.pose)?;
        out.pose = pose.local;
        *state = next;
        Ok(Frame { out, model: pose.model, skin: pose.skin })
    }
}
