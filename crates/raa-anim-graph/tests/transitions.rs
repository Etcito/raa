use raa_anim::{Clip, Rig};
use raa_anim_graph::{Compiled, Crossfade, Graph, GraphState, ParamValue, Params};

fn transition(from: &str, to: &str, priority: i32, fade: u32, exit: &str, when: &str) -> String {
    format!(r#"{{"from":"{from}","to":"{to}","priority":{priority},"blendTicks":{fade},"exitTick":{exit},"when":[{when}]}}"#)
}
fn graph(transitions: &[String]) -> Compiled {
    let rig = Rig::load(include_str!("../../raa-anim/tests/fixtures/rig-basic.json")).unwrap();
    let clip = Clip::load(include_str!("../../raa-anim/tests/fixtures/clip-linear-constant.json"), &rig).unwrap();
    let states = ["a", "b", "c"].map(|name| format!(r#"{{"name":"{name}","motion":"clip-linear-constant","loop":true,"speedParam":null}}"#)).join(",");
    let doc = format!(r#"{{"format":"raa.animgraph","version":1,"name":"g","rig":"rig-basic",
        "params":[{{"name":"go","type":"trigger"}},{{"name":"other","type":"trigger"}},
        {{"name":"f","type":"float","default":2}},{{"name":"i","type":"int","default":3}},{{"name":"b","type":"bool","default":true}}],
        "blendSpaces":[],"layers":[{{"name":"base","machine":"m","mask":null,"weight":1,"additive":false}},
        {{"name":"top","machine":"m","mask":null,"weight":1,"additive":false}}],
        "machines":[{{"name":"m","entry":"a","states":[{states}],"transitions":[{}]}}],
        "rootMotion":{{"bone":"root","mode":"none"}}}}"#, transitions.join(","));
    Graph::load_compiled(&doc, &rig, &[clip]).unwrap()
}
const GO: &str = r#"{"param":"go","op":"trigger"}"#;
#[test]
fn current_priority_file_order_and_wildcard_fallback() {
    for (ts, expected) in [
        (vec![transition("*", "c", -9, 0, "null", ""), transition("a", "b", 9, 0, "null", "")], 1),
        (vec![transition("a", "c", 9, 0, "null", ""), transition("a", "b", -1, 0, "null", "")], 1),
        (vec![transition("a", "b", 0, 0, "null", ""), transition("a", "c", 0, 0, "null", "")], 1),
        (vec![transition("a", "b", -9, 0, "1", ""), transition("*", "c", 0, 0, "null", "")], 2),
        (vec![transition("*", "c", 9, 0, "null", ""), transition("*", "b", 0, 0, "null", ""), transition("*", "c", 0, 0, "null", "")], 1),
    ] {
        let g = graph(&ts);
        let mut s = GraphState::new(&g);
        g.advance_transitions(&mut s, &Params::new(&g.graph));
        assert_eq!(s.layers[0].state, expected);
        assert_eq!(s.layers[0], s.layers[1]);
    }
}
#[test]
fn gates_pending_triggers_and_no_same_tick_chains() {
    let when = format!(r#"{GO},{{"param":"b","op":"==","value":true}}"#);
    let g = graph(&[transition("a", "b", 0, 0, "2", &when), transition("b", "c", 0, 0, "null", "")]);
    let mut s = GraphState::new(&g);
    let mut p = Params::new(&g.graph);
    p.fire("go").unwrap(); p.fire("other").unwrap();
    p.set("b", ParamValue::Bool(false)).unwrap();
    g.advance_transitions(&mut s, &p);
    p.consume(0); p.consume(1);
    g.advance_transitions(&mut s, &p);
    assert_eq!((s.layers[0].state, s.layers[0].clock), (0, 2));
    g.advance_transitions(&mut s, &p); // Gate open, condition false: keep trigger.
    assert!(s.pending[0]);
    p.set("b", ParamValue::Bool(true)).unwrap();
    g.advance_transitions(&mut s, &p);
    assert_eq!((s.layers[0].state, s.layers[0].clock), (1, 0));
    assert_eq!(&s.pending[..2], &[false, true]);
    assert!(s.layers[0].fade.is_none());
    g.advance_transitions(&mut s, &p);
    assert_eq!(s.layers[0].state, 2);
    let boundary = graph(&[transition("a", "b", 0, 0, "2", "")]);
    let mut s = GraphState::new(&boundary);
    for expected in [0, 0, 1] {
        boundary.advance_transitions(&mut s, &Params::new(&boundary.graph));
        assert_eq!(s.layers[0].state, expected);
    }
}
#[test]
fn every_comparison_operator_and_type() {
    for (param, value) in [("f", "2"), ("i", "3"), ("b", "true")] {
        for (op, expected) in [("==", true), ("!=", false), (">", false), ("<", false), (">=", true), ("<=", true)] {
            if param == "b" && op != "==" && op != "!=" { continue; }
            let when = format!(r#"{{"param":"{param}","op":"{op}","value":{value}}}"#);
            let g = graph(&[transition("a", "b", 0, 0, "null", &when)]);
            let mut s = GraphState::new(&g);
            g.advance_transitions(&mut s, &Params::new(&g.graph));
            assert_eq!(s.layers[0].state == 1, expected, "{param} {op}");
        }
    }
}
#[test]
fn exact_fade_endpoints_interruption_and_snapshot_continuation() {
    let g = graph(&[transition("a", "b", 0, 4, "null", ""), transition("b", "c", 0, 2, "null", GO)]);
    let mut s = GraphState::new(&g);
    let mut p = Params::new(&g.graph);
    s.layers[0].phase = 0.75;
    g.advance_transitions(&mut s, &p);
    let fade = s.layers[0].fade.unwrap();
    assert_eq!((fade.from, fade.from_phase, fade.weight()), (0, 0.75, 0.0));
    for expected in [0.25, 0.5, 0.75] {
        g.advance_transitions(&mut s, &p);
        assert_eq!(s.layers[0].fade.unwrap().weight(), expected);
    }
    g.advance_transitions(&mut s, &p);
    assert!(s.layers[0].fade.is_none());
    assert_eq!(Crossfade { elapsed: 4, ..fade }.weight(), 1.0);
    assert_eq!(Crossfade { ticks: 0, ..fade }.weight(), 1.0);
    s = GraphState::new(&g);
    g.advance_transitions(&mut s, &p);
    s.layers[0].phase = 0.25;
    p.fire("go").unwrap();
    g.advance_transitions(&mut s, &p);
    p.consume(0);
    let interrupted = s.layers[0].fade.unwrap();
    assert_eq!((interrupted.from, interrupted.from_phase, interrupted.weight()), (1, 0.25, 0.0));
    let mut bytes = Vec::new(); s.write(&mut bytes);
    let mut restored = GraphState::read(&g, &bytes).unwrap();
    for expected in [Some(0.5), None] {
        g.advance_transitions(&mut s, &p);
        g.advance_transitions(&mut restored, &p);
        assert_eq!(s, restored);
        assert_eq!(s.layers[0].fade.map(|f| f.weight()), expected);
    }
}
#[test]
fn competing_machines_consume_only_winning_triggers() {
    let other = r#"{"param":"other","op":"trigger"}"#;
    let mut g = graph(&[transition("a", "b", 0, 0, "null", GO), transition("a", "c", 1, 0, "null", other)]);
    let mut second = g.graph.machines[0].clone(); second.name = "second".into();
    g.graph.machines.push(second);
    g.references.machines.push(g.references.machines[0].clone());
    g.references.layers[1].machine = 1;
    let mut s = GraphState::new(&g);
    let mut p = Params::new(&g.graph);
    p.fire("go").unwrap(); p.fire("other").unwrap();
    g.advance_transitions(&mut s, &p);
    assert_eq!((s.layers[0].state, s.layers[1].state), (1, 2));
    assert_eq!(&s.pending[..2], &[false, false]);
}
#[test]
fn unequal_numeric_comparisons_and_zero_length_interruption() {
    for (op, threshold, expected) in [("!=", 1, true), (">", 1, true), ("<", 3, true), (">=", 3, false), ("<=", 1, false)] {
        let when = format!(r#"{{"param":"f","op":"{op}","value":{threshold}}}"#);
        let g = graph(&[transition("a", "b", 0, 0, "null", &when)]);
        let mut s = GraphState::new(&g);
        g.advance_transitions(&mut s, &Params::new(&g.graph));
        assert_eq!(s.layers[0].state == 1, expected);
    }
    let g = graph(&[transition("a", "b", 0, 8, "0", ""), transition("b", "c", 0, 0, "0", "")]);
    let mut s = GraphState::new(&g);
    let p = Params::new(&g.graph);
    g.advance_transitions(&mut s, &p);
    assert!(s.layers[0].fade.is_some());
    g.advance_transitions(&mut s, &p);
    assert_eq!(s.layers[0].state, 2);
    assert!(s.layers[0].fade.is_none());
}
