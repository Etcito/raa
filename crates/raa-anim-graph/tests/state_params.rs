use raa_anim::{Clip, Rig};
use raa_anim_graph::{Compiled, Crossfade, Graph, GraphState, ParamValue, Params, codes};
const DOC: &str = r#"{"format":"raa.animgraph","version":1,"name":"g","rig":"rig-basic","params":[{"name":"speed","type":"float","default":1.5},{"name":"level","type":"int","default":3},{"name":"flag","type":"bool","default":true},{"name":"jump","type":"trigger"}],"blendSpaces":[],"layers":[{"name":"base","machine":"m","mask":null,"weight":1,"additive":false},{"name":"top","machine":"m","mask":["child"],"weight":0.5,"additive":true}],"machines":[{"name":"m","entry":"idle","states":[{"name":"idle","motion":"clip-linear-constant","loop":true,"speedParam":null},{"name":"jump","motion":"clip-bezier","loop":false,"speedParam":null}],"transitions":[{"from":"*","to":"jump","when":[{"param":"jump","op":"trigger"}],"blendTicks":6,"exitTick":null,"priority":0}]}],"rootMotion":{"bone":"root","mode":"none"}}"#;
fn compiled() -> Compiled {
    let rig = Rig::load(include_str!("../../raa-anim/tests/fixtures/rig-basic.json")).unwrap();
    let idle = Clip::load(include_str!("../../raa-anim/tests/fixtures/clip-linear-constant.json"), &rig).unwrap();
    let jump = Clip::load(include_str!("../../raa-anim/tests/fixtures/clip-bezier.json"), &rig).unwrap();
    Graph::load_compiled(DOC, &rig, &[idle, jump]).unwrap()
}
fn values(params: &Params) -> Vec<ParamValue> {
    (0..params.len()).map(|i| params.value(i).unwrap()).collect()
}
#[test]
fn defaults_and_initial_machine_states_follow_the_graph() {
    let compiled = compiled();
    let params = Params::new(&compiled.graph);
    assert_eq!(values(&params),
        [ParamValue::Float(1.5), ParamValue::Int(3), ParamValue::Bool(true), ParamValue::Trigger(false)]);
    assert_eq!(params.iter().map(|(name, _)| name).collect::<Vec<_>>(), ["speed", "level", "flag", "jump"]);
    let state = GraphState::new(&compiled);
    assert_eq!((state.tick, state.layers.len(), state.pending.len()), (0, 2, 4));
    assert!(state.layers.iter().all(|layer| {
        layer.machine == 0 && layer.state == 0 && layer.clock == 0 && layer.phase == 0.0 && layer.fade.is_none()
    }));
    assert_eq!(state.pending, vec![false; 4]);
}
#[test]
fn keeps_float_int_and_bool_values_distinct() {
    let compiled = compiled();
    let mut params = Params::new(&compiled.graph);
    params.set("speed", ParamValue::Float(2.5)).unwrap();
    params.set("level", ParamValue::Int(9)).unwrap();
    params.set("flag", ParamValue::Bool(false)).unwrap();
    assert_eq!(values(&params),
        [ParamValue::Float(2.5), ParamValue::Int(9), ParamValue::Bool(false), ParamValue::Trigger(false)]);
    for (name, value) in [("speed", ParamValue::Int(1)), ("level", ParamValue::Float(1.0)),
        ("flag", ParamValue::Float(1.0)), ("flag", ParamValue::Int(1)), ("jump", ParamValue::Bool(true))] {
        assert_eq!(params.set(name, value).unwrap_err().code, codes::PARAM);
    }
    assert_eq!(params.value(1), Some(ParamValue::Int(9)));
}
#[test]
fn rejects_invalid_runtime_values_with_graph_param() {
    let compiled = compiled();
    let mut params = Params::new(&compiled.graph);
    for value in [f32::NAN, f32::INFINITY] {
        assert_eq!(params.set("speed", ParamValue::Float(value)).unwrap_err().code, codes::PARAM);
    }
    assert_eq!(params.set("missing", ParamValue::Float(0.0)).unwrap_err().code, codes::PARAM);
    assert_eq!(params.set_at(64, ParamValue::Float(0.0)).unwrap_err().code, codes::PARAM);
    assert_eq!(params.fire("speed").unwrap_err().code, codes::PARAM);
    assert_eq!(params.fire("missing").unwrap_err().code, codes::PARAM);
    assert_eq!(params.value(0), Some(ParamValue::Float(1.5)));
    assert!(!params.consume(0));
}
#[test]
fn pending_triggers_stay_until_consumed_and_round_trip() {
    let compiled = compiled();
    let mut state = GraphState::new(&compiled);
    let mut inputs = Params::new(&compiled.graph);
    inputs.fire("jump").unwrap();
    state.apply(&inputs);
    assert!(state.pending[3]);
    state.apply(&Params::new(&compiled.graph));
    assert!(state.pending[3]);
    assert!(inputs.consume(3));
    assert!(!inputs.pending(3));
    assert!(!inputs.consume(3));
    let mut bytes = Vec::new();
    state.write(&mut bytes);
    let restored = GraphState::read(&compiled, &bytes).unwrap();
    assert_eq!(restored, state);
}
#[test]
fn one_graph_creates_independent_states_and_snapshots() {
    let compiled = compiled();
    let mut first = GraphState::new(&compiled);
    let pristine = GraphState::new(&compiled);
    first.tick = 4;
    first.layers[0].clock = 3;
    first.layers[0].phase = 0.5;
    first.layers[0].fade = Some(Crossfade { from: 1, from_phase: 0.25, elapsed: 2, ticks: 6 });
    first.pending[3] = true;
    assert_eq!(GraphState::new(&compiled), pristine);
    let mut bytes = Vec::new();
    first.write(&mut bytes);
    let copy = GraphState::read(&compiled, &bytes).unwrap();
    assert_eq!(copy, first);
    first.layers[0].clock = 99;
    first.layers[1].phase = 0.75;
    assert_eq!(copy.layers[0].clock, 3);
    assert_eq!(pristine.layers[1].phase, 0.0);
}
