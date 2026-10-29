//! S19 graph loading, ticking and rollback over the existing S16/S17 handles.
//! The graph borrows the caller's rig and clips; each state keeps its graph, so
//! steps and restores never mix graphs.
use crate::clips::{AnimationClip, AnimationRig};
use raa_anim_graph::{
    GraphError, GraphEvent, GraphRuntime, GraphState, ParamType, ParamValue, Params, RootDelta,
    codes,
};
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// A compiled S19 graph retaining the rig and clips it was loaded with.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct AnimationGraph {
    runtime: Arc<GraphRuntime>,
}

/// One independent S19 execution state created from an [`AnimationGraph`].
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct AnimationGraphState {
    graph: Arc<GraphRuntime>,
    state: GraphState,
}

/// One JavaScript parameter value before it is checked against its declared type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamInput {
    Number(f64),
    Bool(bool),
}

/// One tick for JavaScript: S16 skin matrices flattened column-major in rig bone
/// order (16 floats per bone, the S18 `setSkinPose` layout), root delta, events.
#[derive(Debug, Clone, PartialEq)]
pub struct StepOutput {
    pub skin: Vec<f32>,
    pub root_delta: RootDelta,
    pub events: Vec<GraphEvent>,
}

/// This tick's typed parameters: values persist from the state's last inputs and
/// triggers pulse only on `true`. `float` takes a finite number, `int` an integer
/// in i32 range, `bool` and `trigger` a boolean.
fn params_for<S: AsRef<str>>(
    state: &GraphState,
    inputs: &[(S, ParamInput)],
) -> Result<Params, GraphError> {
    use {ParamInput as I, ParamType as T};
    let mut params = state.params.clone();
    (0..params.len()).for_each(|i| _ = params.consume(i));
    for (name, input) in inputs {
        let name = name.as_ref();
        let unknown = || GraphError::new(codes::PARAM, "/params", format!("unknown {name:?}"));
        let i = params.index_of(name).ok_or_else(unknown)?;
        let int = |n: f64| n.fract() == 0.0 && (-2147483648.0..=2147483647.0).contains(&n);
        let value = match (params.ty(i), *input) {
            (Some(T::Trigger), I::Bool(false)) => continue,
            (Some(T::Trigger), I::Bool(true)) => {
                params.fire_at(i)?;
                continue;
            }
            // `set_at` refuses non-finite floats.
            (Some(T::Float), I::Number(n)) => Some(ParamValue::Float(n as f32)),
            (Some(T::Int), I::Number(n)) => int(n).then_some(ParamValue::Int(n as i32)),
            (Some(T::Bool), I::Bool(b)) => Some(ParamValue::Bool(b)),
            _ => None,
        };
        let mismatch = || GraphError::new(codes::PARAM, format!("/params/{i}"), "type mismatch");
        params.set_at(i, value.ok_or_else(mismatch)?)?;
    }
    Ok(params)
}

/// Load S19 graph JSON against the rig handle and the supplied clip handles.
/// The clip handles are borrowed and stay usable after this call.
pub fn load_graph(
    json: &str,
    rig: &AnimationRig,
    clips: &[&AnimationClip],
) -> Result<AnimationGraph, GraphError> {
    let mut owned = Vec::with_capacity(clips.len());
    for clip in clips {
        let (_, clip) = clip.parts();
        owned.push(clip.clone());
    }
    let runtime = Arc::new(GraphRuntime::load(json, rig.shared_rig(), &owned)?);
    Ok(AnimationGraph { runtime })
}

pub fn register(cases: &mut Vec<crate::Case>) {
    cases.push(crate::brick::GRAPH_RUN);
}

/// Run a prefix of the shared 600-tick script, capped at its length. Seed is
/// record metadata only: this scenario uses scripted inputs, not randomness.
/// The digest is API data, not a cross-platform acceptance comparison.
pub fn graph_run_selftest(_seed: u32, ticks: u32) -> u32 {
    let run = graph_run(ticks).expect("graph-run scenario");
    if let Some(rollback) = &run.rollback {
        assert!(rollback.matched, "graph-run rollback replay diverged");
    }
    run.digest()
}

fn graph_run(ticks: u32) -> Result<raa_anim_graph::ScenarioRun, GraphError> {
    use raa_anim_graph::scenario_data::{
        PENGUIN_GRAPH, PENGUIN_TIMELINE, Timeline, penguin_clips, penguin_rig,
    };
    let rig = penguin_rig();
    let runtime = GraphRuntime::load(PENGUIN_GRAPH, &rig, &penguin_clips(&rig))?;
    let mut timeline = Timeline::load(PENGUIN_TIMELINE, &runtime.new_params())?;
    timeline.ticks = ticks.min(timeline.ticks);
    raa_anim_graph::scenario::run(&runtime, &timeline)
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;
    use crate::{ApiError, SelftestResult, dispatch};

    #[test]
    fn graph_run_is_registered_once_and_dispatches_the_record() {
        let cases = crate::registered_cases();
        let entries: Vec<_> = cases.iter().filter(|c| c.name == "graph-run").collect();
        assert_eq!(entries.len(), 1);
        assert!(std::ptr::fn_addr_eq(
            entries[0].run,
            graph_run_selftest as fn(u32, u32) -> u32,
        ));
        for (seed, ticks) in [(0, 0), (1, 1), (305419896, 600), (u32::MAX, 601)] {
            let SelftestResult { case, seed: actual_seed, ticks: actual_ticks, hash } =
                dispatch("graph-run", seed.into(), ticks.into()).unwrap();
            assert_eq!((case, actual_seed, actual_ticks), ("graph-run", seed, ticks));
            assert_eq!(hash.len(), 8);
            assert!(hash.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        }
    }

    #[test]
    fn graph_run_rejects_invalid_dispatch_inputs() {
        for value in [f64::NAN, f64::INFINITY, -1.0, 0.5, 4294967296.0] {
            assert_eq!(dispatch("graph-run", value, 600.0), Err(ApiError::InvalidInput("seed")));
            assert_eq!(dispatch("graph-run", 0.0, value), Err(ApiError::InvalidInput("ticks")));
        }
        assert_eq!(dispatch("graph-run-extra", 0.0, 0.0),
            Err(ApiError::UnknownCase("graph-run-extra".into())));
    }

    #[test]
    fn graph_run_uses_the_shared_script_and_replays_rollback() {
        let empty = graph_run(0).unwrap();
        assert!(empty.trace.is_empty());
        assert!(empty.rollback.is_none());
        assert_eq!(graph_run(1).unwrap().steps(), 1);
        let full = graph_run(600).unwrap();
        assert_eq!(full.steps(), raa_anim_graph::SCENARIO_TICKS);
        assert!(!full.events.is_empty());
        assert!(full.travel.iter().any(|v| *v != 0.0));
        let rollback = full.rollback.as_ref().unwrap();
        assert!(rollback.matched);
        assert!(rollback.replayed > 0);
        assert!(!rollback.fades.is_empty());
        assert_eq!(full, raa_anim_graph::scenario::run_penguin().unwrap());
        assert_eq!(full, graph_run(u32::MAX).unwrap());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl AnimationGraph {
    /// `GraphState::new` for this graph; every returned state is independent.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = newState))]
    pub fn new_state(&self) -> AnimationGraphState {
        AnimationGraphState {
            graph: Arc::clone(&self.runtime),
            state: self.runtime.new_state(),
        }
    }
}

impl AnimationGraph {
    /// `Graph::step` on a state of this graph. A refused call (another graph's
    /// state, unknown or mistyped parameters, runtime errors) leaves it untouched.
    pub fn step<S: AsRef<str>>(
        &self,
        state: &mut AnimationGraphState,
        inputs: &[(S, ParamInput)],
    ) -> Result<StepOutput, GraphError> {
        if !Arc::ptr_eq(&self.runtime, &state.graph) {
            return Err(GraphError::new(codes::SCHEMA, "/state", "foreign state"));
        }
        let params = params_for(&state.state, inputs)?;
        let frame = self.runtime.frame(&mut state.state, &params)?;
        Ok(StepOutput {
            skin: frame.skin.into_iter().flatten().collect(),
            root_delta: frame.out.root_delta,
            events: frame.out.events,
        })
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl AnimationGraphState {
    /// `GraphState::write`: a complete snapshot for rollback, bound to this graph.
    pub fn save(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.state.write(&mut bytes);
        bytes
    }
}

impl AnimationGraphState {
    /// `GraphState::read` against this state's graph; the live state is replaced
    /// only after the whole snapshot validates.
    pub fn restore_bytes(&mut self, bytes: &[u8]) -> Result<(), GraphError> {
        self.state = self.graph.read_state(bytes)?;
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use wasm_bindgen::__rt::{RcRef, WasmPtr};
    use wasm_bindgen::convert::RefFromWasmAbi;

    fn error(e: GraphError) -> JsValue {
        let value = js_sys::Error::new(&e.message);
        js_sys::Reflect::set(&value, &"code".into(), &e.code.into()).expect("fresh error");
        js_sys::Reflect::set(&value, &"path".into(), &e.path.into()).expect("fresh error");
        value.into()
    }

    fn invalid(message: &str) -> JsValue {
        let value = js_sys::TypeError::new(message);
        js_sys::Reflect::set(&value, &"code".into(), &"api.invalid_input".into())
            .expect("fresh error");
        js_sys::Reflect::set(&value, &"path".into(), &"".into()).expect("fresh error");
        value.into()
    }

    // wasm-bindgen arrays of exported types take ownership of their elements.
    // Borrow the class instead so the caller's clip handles stay usable.
    fn borrow_clip(value: &JsValue) -> Result<RcRef<AnimationClip>, JsValue> {
        let pointer = js_sys::Reflect::get(value, &JsValue::from_str("__wbg_ptr"))
            .ok()
            .and_then(|pointer| pointer.as_f64())
            .filter(|pointer| {
                pointer.is_finite() && *pointer > 0.0 && *pointer <= f64::from(u32::MAX)
            })
            .ok_or_else(|| invalid("clips must contain AnimationClip handles"))?;
        let abi = WasmPtr::from_usize(pointer as usize);
        Ok(unsafe { <AnimationClip as RefFromWasmAbi>::ref_from_abi(abi) })
    }

    #[wasm_bindgen(js_name = loadGraph)]
    pub fn graph(
        json: &str,
        rig: &AnimationRig,
        clips: &js_sys::Array,
    ) -> Result<AnimationGraph, JsValue> {
        if !js_sys::Array::is_array(clips) {
            return Err(invalid("clips must be an array of AnimationClip handles"));
        }
        let mut anchors = Vec::with_capacity(clips.length() as usize);
        for value in clips.iter() {
            anchors.push(borrow_clip(&value)?);
        }
        let clips: Vec<&AnimationClip> = anchors.iter().map(|anchor| &**anchor).collect();
        super::load_graph(json, rig, &clips).map_err(error)
    }

    fn inputs(params: &JsValue) -> Result<Vec<(String, ParamInput)>, JsValue> {
        if params.is_undefined() || params.is_null() {
            return Ok(Vec::new());
        }
        if !params.is_object() || js_sys::Array::is_array(params) {
            return Err(invalid("params must be an object of parameter values"));
        }
        let entries = js_sys::Object::entries(params.unchecked_ref());
        let mut inputs = Vec::with_capacity(entries.length() as usize);
        for entry in entries.iter() {
            let entry: js_sys::Array = entry.unchecked_into();
            let (name, value) = (entry.get(0).as_string().unwrap_or_default(), entry.get(1));
            let message = || format!("parameter {name:?} must be a number or boolean");
            let input = (value.as_bool().map(ParamInput::Bool))
                .or(value.as_f64().map(ParamInput::Number))
                .ok_or_else(|| error(GraphError::new(codes::PARAM, "/params", message())))?;
            inputs.push((name, input));
        }
        Ok(inputs)
    }

    fn set(target: &js_sys::Object, key: &str, value: &JsValue) -> Result<(), JsValue> {
        js_sys::Reflect::set(target, &key.into(), value).map(drop)
    }

    fn output(out: StepOutput) -> Result<JsValue, JsValue> {
        let f32s = |values: &[f32]| js_sys::Float32Array::from(values).into();
        let root = js_sys::Object::new();
        set(&root, "t", &f32s(&out.root_delta.t))?;
        set(&root, "r", &f32s(&out.root_delta.r))?;
        let events = js_sys::Array::new();
        for event in out.events {
            let value = js_sys::Object::new();
            set(&value, "name", &event.name.into())?;
            set(&value, "tick", &event.tick.into())?;
            set(&value, "layer", &(event.layer as u32).into())?;
            set(&value, "clip", &(event.clip as u32).into())?;
            events.push(&value);
        }
        let result = js_sys::Object::new();
        set(&result, "skin", &f32s(&out.skin))?;
        set(&result, "rootDelta", &root)?;
        set(&result, "events", &events)?;
        Ok(result.into())
    }

    #[wasm_bindgen]
    impl AnimationGraph {
        /// `graph.step(state, params) -> {skin, rootDelta: {t, r}, events}`.
        #[wasm_bindgen(js_name = step)]
        pub fn js_step(
            &self,
            state: &mut AnimationGraphState,
            params: JsValue,
        ) -> Result<JsValue, JsValue> {
            output(self.step(state, &inputs(&params)?).map_err(error)?)
        }
    }

    #[wasm_bindgen]
    impl AnimationGraphState {
        /// `state.restore(bytes)`; a refused snapshot keeps the live state.
        #[wasm_bindgen(js_name = restore)]
        pub fn js_restore(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
            self.restore_bytes(bytes).map_err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clips::{load_clip, load_rig};
    use raa_anim_graph::scenario_data::{PENGUIN_CLIPS, PENGUIN_GRAPH, PENGUIN_RIG, clip_text};

    fn penguin() -> (AnimationRig, Vec<AnimationClip>) {
        let rig = load_rig(PENGUIN_RIG).expect("penguin rig");
        let clips = PENGUIN_CLIPS
            .iter()
            .map(|spec| load_clip(&rig, &clip_text(spec)).expect("penguin clip"))
            .collect();
        (rig, clips)
    }

    #[test]
    fn loads_graph_from_handles_and_creates_independent_states() {
        let (rig, clips) = penguin();
        let refs: Vec<&AnimationClip> = clips.iter().collect();
        let graph = load_graph(PENGUIN_GRAPH, &rig, &refs).expect("penguin graph");
        assert_eq!(graph.runtime.graph().name, "penguin");
        assert_eq!(graph.runtime.clips().len(), clips.len());
        let first = graph.new_state();
        let second = graph.new_state();
        assert_eq!(first.state, graph.runtime.new_state());
        assert_eq!(first.state, second.state);
        assert_eq!(first.state.tick, 0);
        assert_eq!(first.state.layers.len(), graph.runtime.graph().layers.len());
    }

    #[test]
    fn clip_handles_stay_usable_after_loading() {
        let (rig, clips) = penguin();
        let refs: Vec<&AnimationClip> = clips.iter().collect();
        for _ in 0..2 {
            load_graph(PENGUIN_GRAPH, &rig, &refs).expect("repeat load");
        }
        let (pose, _) = crate::clips::sample_clip(&clips[0], 0).expect("sample after load");
        assert!(!pose.skin.is_empty());
    }

    #[test]
    fn refuses_invalid_documents_without_creating_a_graph() {
        let (rig, _) = penguin();
        let error = load_graph("{}", &rig, &[])
            .err()
            .expect("empty graph refused");
        assert!(error.code.starts_with("graph."));
        let missing = load_graph(PENGUIN_GRAPH, &rig, &[])
            .err()
            .expect("missing clips refused");
        assert_eq!(missing.code, "graph.clip_unknown");
    }

    #[test]
    fn refuses_a_graph_for_another_rig() {
        let other = load_rig(include_str!(
            "../../../raa-anim/tests/fixtures/rig-basic.json"
        ))
        .expect("basic rig");
        let error = load_graph(PENGUIN_GRAPH, &other, &[])
            .err()
            .expect("rig mismatch refused");
        assert_eq!(error.code, "graph.schema");
    }

    fn load(json: &str) -> AnimationGraph {
        let (rig, clips) = penguin();
        let refs: Vec<&AnimationClip> = clips.iter().collect();
        load_graph(json, &rig, &refs).expect("penguin graph")
    }

    fn value(state: &AnimationGraphState, name: &str) -> ParamValue {
        let params = &state.state.params;
        params.value(params.index_of(name).unwrap()).unwrap()
    }

    #[test]
    fn typed_params_are_checked_persist_and_triggers_pulse() {
        use ParamInput::{Bool, Number};
        let graph = load(PENGUIN_GRAPH);
        let mut state = graph.new_state();
        let before = state.save();
        for bad in [
            ("speed", Bool(true)),
            ("speed", Number(1e39)),
            ("emote", Number(1.5)),
            ("emote", Number(2147483648.0)),
            ("grounded", Number(1.0)),
            ("jump", Number(1.0)),
            ("missing", Number(0.0)),
        ] {
            let error = graph.step(&mut state, &[bad]).err().expect("refused");
            assert_eq!(error.code, "graph.param", "{bad:?}");
        }
        assert_eq!(state.save(), before, "refused steps keep the state");
        let typed = [
            ("speed", Number(1.5)),
            ("emote", Number(-2.0)),
            ("jump", Bool(true)),
        ];
        graph.step(&mut state, &typed).unwrap();
        assert_eq!(value(&state, "jump"), ParamValue::Trigger(true));
        graph
            .step(&mut state, &[("grounded", Bool(false))])
            .unwrap();
        assert_eq!(value(&state, "speed"), ParamValue::Float(1.5));
        assert_eq!(value(&state, "emote"), ParamValue::Int(-2));
        assert_eq!(value(&state, "grounded"), ParamValue::Bool(false));
        assert_eq!(value(&state, "jump"), ParamValue::Trigger(false));
    }

    #[test]
    fn skin_layout_and_event_order_follow_the_runtime() {
        let graph = load(PENGUIN_GRAPH);
        let (mut state, mut mirror) = (graph.new_state(), graph.runtime.new_state());
        let mut params = graph.runtime.new_params();
        params.set("speed", ParamValue::Float(1.5)).unwrap();
        let bones = graph.runtime.rig().bones.len();
        let mut events = Vec::new();
        for tick in 0..120 {
            // `*` transitions re-fire while their condition holds: enter dance once.
            let emote = if tick == 0 { 2 } else { 0 };
            params.set("emote", ParamValue::Int(emote)).unwrap();
            let inputs = [
                ("speed", ParamInput::Number(1.5)),
                ("emote", ParamInput::Number(emote.into())),
            ];
            let out = graph.step(&mut state, &inputs).unwrap();
            let frame = graph.runtime.frame(&mut mirror, &params).unwrap();
            assert_eq!(out.skin.len(), bones * 16);
            assert_eq!(out.skin, frame.skin.concat());
            for m in out.skin.chunks_exact(16) {
                assert_eq!([m[3], m[7], m[11], m[15]], [0.0, 0.0, 0.0, 1.0]);
            }
            // Column-major: hips (bone 1) translation is m[12..15], offset from rest.
            assert!((out.skin[16 + 13] - (frame.out.pose[1].t[1] - 0.4)).abs() <= 1e-6);
            assert_eq!(out.root_delta, frame.out.root_delta);
            assert_eq!(out.events, frame.out.events);
            assert!(out.events.windows(2).all(|w| w[0].layer <= w[1].layer));
            events.extend(out.events.into_iter().map(|e| (e.layer, e.name)));
        }
        let steps: Vec<_> = events.iter().filter(|e| e.1.starts_with("step.")).collect();
        assert!(steps.len() >= 4 && steps.windows(2).all(|w| w[0].1 != w[1].1));
        assert!(
            events.iter().any(|e| e.0 == 1),
            "the emote layer reports events"
        );
    }

    #[test]
    fn snapshots_round_trip_and_stay_with_their_graph() {
        let graph = load(PENGUIN_GRAPH);
        let speed = [("speed", ParamInput::Number(4.0))];
        let jump = [("jump", ParamInput::Bool(true))];
        let run = |state: &mut AnimationGraphState| -> Vec<StepOutput> {
            let tick = |i| {
                graph
                    .step(state, if i == 3 { &jump } else { &speed })
                    .unwrap()
            };
            (0..20).map(tick).collect()
        };
        let mut state = graph.new_state();
        run(&mut state);
        let bytes = state.save();
        let first = run(&mut state);
        let advanced = state.save();
        state.restore_bytes(&bytes).unwrap();
        assert_eq!(state.save(), bytes, "byte round trip");
        assert_eq!(run(&mut state), first, "replay matches");
        let mut fresh = graph.new_state();
        fresh.restore_bytes(&bytes).unwrap();
        assert_eq!(run(&mut fresh), first, "restores into another state");
        let other = load(&PENGUIN_GRAPH.replacen("\"penguin\"", "\"other\"", 1));
        let foreign = other.new_state().save();
        let (short, long) = (&bytes[..bytes.len() - 1], [&bytes[..], &[0]].concat());
        for bad in [&[][..], &bytes[1..], short, &long, &foreign] {
            assert_eq!(state.restore_bytes(bad).err().unwrap().code, "graph.schema");
            assert_eq!(
                state.save(),
                advanced,
                "failed restore keeps the live state"
            );
        }
        let error = other.step(&mut state, &speed).err().unwrap();
        assert_eq!((error.code, state.save()), ("graph.schema", advanced));
    }
}
