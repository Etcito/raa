use raa_anim::{Clip, Rig};
use raa_anim_graph::{Compiled, Crossfade, Graph, GraphState, ParamValue, Params};

const SYNC: usize = 0; const FREE: usize = 1; const LOOP: usize = 2; const ONCE: usize = 3; const ZERO: usize = 4;
/// Root translation x equals the sampled tick; child rotation is one constant key.
fn clip(rig: &Rig, name: &str, len: u32, r: &str) -> Clip {
    let text = format!(r#"{{"format":"raa.clip","version":2,"name":"{name}","rig":"rig-basic","lengthTicks":{len},"loop":true,"rootMotion":"none",
        "tracks":[{{"bone":"root","channel":"translation","keys":[{{"tick":0,"v":[0,0,0],"interp":"LINEAR"}},{{"tick":{len},"v":[{len},0,0],"interp":"LINEAR"}}]}},
        {{"bone":"child","channel":"rotation","keys":[{{"tick":0,"v":{r},"interp":"CONSTANT"}}]}}],"events":[]}}"#);
    Clip::load(&text, rig).unwrap()
}
fn setup() -> (Rig, Vec<Clip>, Compiled, Params) {
    let rig = Rig::load(include_str!("../../raa-anim/tests/fixtures/rig-basic.json")).unwrap();
    let clips = vec![clip(&rig, "c30", 30, "[0,0,0,-1]"), clip(&rig, "c48", 48, "[0,0.6,0,0.8]"), clip(&rig, "c0", 1, "[0,0,0,1]")];
    let space = |name: &str, a: &str, sync: bool| format!(r#"{{"name":"{name}","params":["x"],"points":[{{"clip":"{a}","at":[0]}},{{"clip":"c48","at":[1]}}],"sync":{sync}}}"#);
    let state = |name: &str, motion: &str, looping: bool| format!(r#"{{"name":"{name}","motion":"{motion}","loop":{looping},"speedParam":"rate"}}"#);
    let doc = format!(r#"{{"format":"raa.animgraph","version":1,"name":"g","rig":"rig-basic",
        "params":[{{"name":"x","type":"float","default":0.5}},{{"name":"rate","type":"float","default":1}}],
        "blendSpaces":[{},{},{}],"layers":[{{"name":"base","machine":"m","mask":null,"weight":1,"additive":false}}],
        "machines":[{{"name":"m","entry":"sync","states":[{},{},{},{},{}],"transitions":[]}}],
        "rootMotion":{{"bone":"root","mode":"none"}}}}"#,
        space("sync", "c30", true), space("free", "c30", false), space("zsync", "c0", true),
        state("sync", "sync", true), state("free", "free", true), state("loop", "c30", true),
        state("once", "c48", false), state("zero", "zsync", true));
    let compiled = Graph::load_compiled(&doc, &rig, &clips).unwrap();
    let params = Params::new(&compiled.graph);
    (rig, clips, compiled, params)
}
fn ticks(g: &Compiled, clips: &[Clip], state: usize, phase: f32, p: &Params) -> Vec<u32> {
    g.contributions(clips, 0, state, phase, p).unwrap().iter().map(|c| c.tick).collect()
}
fn close(a: f32, b: f32) -> bool { (a - b).abs() < 1e-4 }
fn start(g: &Compiled, state: usize) -> GraphState { let mut s = GraphState::new(g); s.layers[0].state = state; s }
#[test]
fn unequal_durations_share_one_normalized_phase() {
    let (rig, clips, g, p) = setup();
    assert_eq!(ticks(&g, &clips, SYNC, 0.5, &p), [15, 24]);
    assert_eq!(ticks(&g, &clips, SYNC, 0.999, &p), [29, 47]);
    // D = 0.5 * 30 + 0.5 * 48 = 39 ticks per shared cycle.
    let mut s = start(&g, SYNC);
    let seen: Vec<f32> = (0..39).map(|_| { g.step_motion(&rig, &clips, &mut s, &p).unwrap(); s.layers[0].phase }).collect();
    assert_eq!(seen[0], 1.0 / 39.0);
    assert!(seen[38] < 1e-5 || seen[38] > 1.0 - 1e-5, "{}", seen[38]);
    // Opposite-sign rotation keys blend through sign alignment, normalized.
    let pose = g.sample_state(&rig, &clips, 0, SYNC, 0.5, &p).unwrap();
    let r = pose[1].r;
    assert!(close(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3], 1.0));
    assert!(r[3] < -0.9 && r[1] < -0.3, "{r:?}");
    assert!(close(pose[0].t[0], 0.5 * 15.0 + 0.5 * 24.0));
}
#[test]
fn phase_stays_continuous_as_weights_change() {
    let (_, clips, g, mut p) = setup();
    let mut phase = 0.0;
    for i in 0..=40 {
        p.set("x", ParamValue::Float(i as f32 / 40.0)).unwrap();
        for c in g.contributions(&clips, 0, SYNC, phase, &p).unwrap() {
            assert_eq!(c.tick, (phase * clips[c.clip].length_ticks as f32) as u32);
        }
        let next = g.next_phase(&clips, 0, SYNC, phase, &p).unwrap();
        let step = if next < phase { next + 1.0 - phase } else { next - phase };
        assert!(step >= 1.0 / 48.0 - 1e-6 && step <= 1.0 / 30.0 + 1e-6, "tick {i}: {step}");
        phase = next;
    }
    p.set("x", ParamValue::Float(1.0)).unwrap();
    assert_eq!(g.next_phase(&clips, 0, SYNC, 0.25, &p).unwrap(), 0.25 + 1.0 / 48.0);
}
#[test]
fn unsynchronized_points_keep_their_own_clocks() {
    let (rig, clips, g, p) = setup();
    let mut s = start(&g, FREE);
    for k in 0..240u32 {
        assert_eq!(ticks(&g, &clips, FREE, s.layers[0].phase, &p), [k % 30, k % 48]);
        let pose = g.step_motion(&rig, &clips, &mut s, &p).unwrap();
        assert!(close(pose[0][0].t[0], 0.5 * (k % 30) as f32 + 0.5 * (k % 48) as f32));
    }
    assert_eq!(s.layers[0].phase, 0.0); // lcm(30, 48) = 240 wraps exactly.
}
#[test]
fn speed_changes_keep_fractional_progress() {
    let (_, clips, g, mut p) = setup();
    let mut s = start(&g, LOOP);
    let seen: Vec<u32> = [0.5, 0.5, 0.5, 2.0, 0.25, 0.25, 0.25, 0.25, 30.0, -1.0].map(|rate| {
        p.set("rate", ParamValue::Float(rate)).unwrap();
        g.advance_motion(&clips, &mut s, &p).unwrap();
        ticks(&g, &clips, LOOP, s.layers[0].phase, &p)[0]
    }).to_vec();
    assert_eq!((seen, s.layers[0].phase), (vec![0, 1, 1, 3, 3, 4, 4, 4, 4, 3], 3.5));
    for (rate, state, phase, next) in [(-5.0, LOOP, 3.5, 28.5), (7.0, ONCE, 45.0, 48.0), (0.0, SYNC, 0.4, 0.4)] {
        p.set("rate", ParamValue::Float(rate)).unwrap();
        assert_eq!(g.next_phase(&clips, 0, state, phase, &p).unwrap(), next);
    }
    assert_eq!(ticks(&g, &clips, ONCE, 48.0, &p), [48]);
}
#[test]
fn zero_duration_clips_are_safe() {
    let (rig, mut clips, g, mut p) = setup();
    clips[2].length_ticks = 0;
    p.set("x", ParamValue::Float(0.0)).unwrap();
    let mut s = start(&g, ZERO);
    for _ in 0..3 {
        let pose = g.step_motion(&rig, &clips, &mut s, &p).unwrap();
        assert_eq!((s.layers[0].phase, pose[0][0].t[0]), (0.0, 0.0));
    }
    p.set("x", ParamValue::Float(0.5)).unwrap();
    assert_eq!(ticks(&g, &clips, ZERO, 0.5, &p), [0, 24]);
    assert_eq!(g.next_phase(&clips, 0, ZERO, 0.0, &p).unwrap(), 1.0 / 24.0);
}
#[test]
fn crossfades_blend_frozen_source_and_snapshots_continue() {
    let (rig, clips, g, p) = setup();
    let mut s = start(&g, ONCE);
    s.layers[0].phase = 20.0;
    for (elapsed, expected) in [(0, 10.0), (2, 15.0), (4, 20.0)] {
        s.layers[0].fade = Some(Crossfade { from: LOOP, from_phase: 10.0, elapsed, ticks: 4 });
        let pose = g.sample_layer(&rig, &clips, &s, 0, &p).unwrap();
        assert!(close(pose[0].t[0], expected), "{elapsed}");
    }
    s.layers[0].fade = Some(Crossfade { from: LOOP, from_phase: 10.0, elapsed: 0, ticks: 4 });
    assert_eq!(g.sample_layer(&rig, &clips, &s, 0, &p).unwrap(), g.sample_state(&rig, &clips, 0, LOOP, 10.0, &p).unwrap());
    let mut bytes = Vec::new();
    s.write(&mut bytes);
    let mut copy = GraphState::read(&g, &bytes).unwrap();
    for _ in 0..6 { assert_eq!(g.step_motion(&rig, &clips, &mut s, &p), g.step_motion(&rig, &clips, &mut copy, &p)); }
    assert_eq!((s.layers[0].phase, s.layers[0].fade.map(|f| f.from_phase)), (26.0, None));
    assert_eq!(s, copy);
}
