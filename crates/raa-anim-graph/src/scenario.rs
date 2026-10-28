//! Scripted functional scenario over [`scenario_data`]: the reusable driver for
//! the graph-run selftest. [`run`] applies the 600-tick timeline one tick at a
//! time, records every [`TickOut`] with its state route and ordered events, and
//! takes a snapshot at the first tick inside a crossfade, then restores it and
//! replays the remaining ticks against the live trace. [`run_penguin`] loads the
//! preceding task's penguin rig, clips, graph and timeline for that script.
use crate::error::GraphError;
use crate::model::ParamType;
use crate::output::{GraphEvent, TickOut};
use crate::params::Params;
use crate::runtime::GraphRuntime;
use crate::scenario_data::{PENGUIN_GRAPH, PENGUIN_TIMELINE, Timeline, penguin_clips, penguin_rig};
use crate::state::GraphState;

/// The script is bounded to 600 ticks; a longer timeline stops here.
pub const SCENARIO_TICKS: u32 = 600;

/// A layer changed state at `tick`; the first entry of every layer is included.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteEntry { pub tick: u32, pub layer: usize, pub state: String }

/// One clip event with the absolute script tick of the step that reported it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioEvent { pub tick: u32, pub event: GraphEvent }

/// One scripted tick: its output plus the simulated state after the step.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioTick { pub tick: u32, pub out: TickOut }

/// Snapshot rollback taken inside a running crossfade and replayed to the end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rollback {
    pub checkpoint: u32,
    pub replayed: u32,
    /// Layers whose crossfade was active (elapsed > 0) when the snapshot was taken.
    pub fades: Vec<usize>,
    pub matched: bool,
    pub first_mismatch: Option<u32>,
}

/// A complete deterministic scenario run.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioRun {
    pub trace: Vec<ScenarioTick>,
    pub route: Vec<RouteEntry>,
    pub events: Vec<ScenarioEvent>,
    /// Sum of every tick's extracted root delta.
    pub travel: [f32; 3],
    pub rollback: Option<Rollback>,
}

struct Checkpoint { tick: u32, bytes: Vec<u8>, held: Params, fades: Vec<usize> }

fn active_fades(state: &GraphState) -> Vec<usize> {
    state.layers.iter().enumerate()
        .filter(|(_, l)| l.fade.is_some_and(|f| f.elapsed > 0))
        .map(|(i, _)| i).collect()
}

fn capture(state: &GraphState, tick: u32, held: &Params) -> Checkpoint {
    let mut bytes = Vec::new();
    state.write(&mut bytes);
    Checkpoint { tick, bytes, held: held.clone(), fades: active_fades(state) }
}

fn replay(rt: &GraphRuntime, timeline: &Timeline, trace: &[ScenarioTick], saved: &Checkpoint)
    -> Result<Rollback, GraphError> {
    let mut state = rt.read_state(&saved.bytes)?;
    let mut held = saved.held.clone();
    let (mut replayed, mut matched, mut first_mismatch) = (0, true, None);
    for step in &trace[saved.tick as usize + 1..] {
        let mut params = held.clone();
        timeline.apply(step.tick, &mut params)?;
        let out = rt.step(&mut state, &params)?;
        if out != step.out {
            matched = false;
            first_mismatch.get_or_insert(step.tick);
        }
        for index in 0..params.len() {
            if params.ty(index) == Some(ParamType::Trigger) { params.consume(index); }
        }
        held = params;
        replayed += 1;
    }
    Ok(Rollback { checkpoint: saved.tick, replayed, fades: saved.fades.clone(), matched, first_mismatch })
}

/// Drive `rt` with `timeline` from tick zero for `timeline.ticks` (at most
/// [`SCENARIO_TICKS`]), applying trigger keys once and holding other values.
pub fn run(rt: &GraphRuntime, timeline: &Timeline) -> Result<ScenarioRun, GraphError> {
    let total = timeline.ticks.min(SCENARIO_TICKS);
    let mut state = rt.new_state();
    let mut held = rt.new_params();
    let (mut trace, mut route, mut events) = (Vec::with_capacity(total as usize), Vec::new(), Vec::new());
    let mut last = vec![usize::MAX; state.layers.len()];
    let mut travel = [0.0f32; 3];
    let mut saved = None;
    for tick in 0..total {
        let mut params = held.clone();
        timeline.apply(tick, &mut params)?;
        let out = rt.step(&mut state, &params)?;
        for bone in 0..3 { travel[bone] += out.root_delta.t[bone]; }
        for (layer, cursor) in state.layers.iter().enumerate() {
            if cursor.state != last[layer] {
                last[layer] = cursor.state;
                let state = rt.graph().machines[cursor.machine].states[cursor.state].name.clone();
                route.push(RouteEntry { tick, layer, state });
            }
        }
        for event in &out.events { events.push(ScenarioEvent { tick, event: event.clone() }); }
        trace.push(ScenarioTick { tick, out });
        for index in 0..params.len() {
            if params.ty(index) == Some(ParamType::Trigger) { params.consume(index); }
        }
        held = params;
        if saved.is_none() && !active_fades(&state).is_empty() {
            saved = Some(capture(&state, tick, &held));
        }
    }
    let rollback = saved.as_ref().map(|s| replay(rt, timeline, &trace, s)).transpose()?;
    Ok(ScenarioRun { trace, route, events, travel, rollback })
}

/// Load the preceding task's penguin data and run the canonical script.
pub fn run_penguin() -> Result<ScenarioRun, GraphError> {
    let rig = penguin_rig();
    let rt = GraphRuntime::load(PENGUIN_GRAPH, &rig, &penguin_clips(&rig))?;
    let timeline = Timeline::load(PENGUIN_TIMELINE, &rt.new_params())?;
    run(&rt, &timeline)
}

impl ScenarioRun {
    /// Ticks actually stepped (never above [`SCENARIO_TICKS`]).
    pub fn steps(&self) -> u32 { self.trace.len() as u32 }
    /// State names entered by `layer`, in route order.
    pub fn states(&self, layer: usize) -> Vec<&str> {
        self.route.iter().filter(|r| r.layer == layer).map(|r| r.state.as_str()).collect()
    }
    /// FNV-1a over the full ordered trace, for the `graph-run` selftest record.
    /// The value is API data, never a cross-platform acceptance comparison.
    pub fn digest(&self) -> u32 {
        let mut hash = raa_det::DET_HASH;
        for step in &self.trace {
            hash.f32(step.tick as f32);
            hash.f32s(&step.out.root_delta.t);
            hash.f32s(&step.out.root_delta.r);
            for pose in &step.out.pose {
                hash.f32s(&pose.t);
                hash.f32s(&pose.r);
                hash.f32s(&pose.s);
            }
            for event in &step.out.events {
                hash.update(event.name.as_bytes());
                hash.f32(event.tick as f32);
                hash.f32(event.layer as f32);
                hash.f32(event.clip as f32);
            }
        }
        hash.finish()
    }
}
