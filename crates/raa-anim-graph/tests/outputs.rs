use raa_anim::{Clip, Rig, RootMotion};
use raa_anim_graph::{Compiled, Graph, GraphEvent, GraphState, ParamValue, Params, TickOut};

const WALK: usize = 0; const RUN: usize = 1; const ONCE: usize = 2;
const BASE: &str = r#"[{"name":"base","machine":"m","mask":null,"weight":1,"additive":false}]"#;
fn k(tick: u32, v: &str) -> String { format!(r#"{{"tick":{tick},"v":{v},"interp":"LINEAR"}}"#) }
fn clip(rig: &Rig, name: &str, len: u32, looping: bool, root: &[String], child: &[String], events: &str) -> Clip {
    let track = |bone: &str, keys: &[String]| format!(r#"{{"bone":"{bone}","channel":"translation","keys":[{}]}}"#, keys.join(","));
    Clip::load(&format!(r#"{{"format":"raa.clip","version":2,"name":"{name}","rig":"rig-basic","lengthTicks":{len},"loop":{looping},
        "rootMotion":"none","tracks":[{},{}],"events":{events}}}"#, track("root", root), track("child", child)), rig).unwrap()
}
/// walk: P = 0,1,2,5,8 in x (uneven, so seams differ from tick 1); run: x = 2 * tick; once: clamped 3 ticks.
fn setup(bone: &str, mode: &str, entry: &str, layers: &str) -> (Rig, Vec<Clip>, Compiled, Params, GraphState) {
    let rig = Rig::load(include_str!("../../raa-anim/tests/fixtures/rig-basic.json")).unwrap();
    let clips = vec![
        clip(&rig, "walk", 4, true, &[k(0, "[0,0,0]"), k(2, "[2,1,0]"), k(4, "[8,2,4]")], &[k(0, "[0,1,0]"), k(4, "[0,1,-4]")],
            r#"[{"tick":3,"name":"b"},{"tick":0,"name":"a"},{"tick":3,"name":"c"}]"#),
        clip(&rig, "run", 4, true, &[k(0, "[0,0,0]"), k(4, "[8,0,0]")], &[k(0, "[0,1,0]")], r#"[{"tick":0,"name":"r"}]"#),
        clip(&rig, "once", 3, false, &[k(0, "[0,0,0]"), k(3, "[3,3,3]")], &[k(0, "[0,1,0]")], r#"[{"tick":3,"name":"end"}]"#),
    ];
    let doc = format!(r#"{{"format":"raa.animgraph","version":1,"name":"g","rig":"rig-basic",
        "params":[{{"name":"x","type":"float","default":0}},{{"name":"rate","type":"float","default":1}},{{"name":"go","type":"trigger"}}],
        "blendSpaces":[{{"name":"mix","params":["x"],"points":[{{"clip":"walk","at":[0]}},{{"clip":"run","at":[1]}}],"sync":false}}],
        "layers":{layers},"machines":[{{"name":"m","entry":"{entry}","states":[
        {{"name":"walk","motion":"walk","loop":true,"speedParam":"rate"}},{{"name":"mix","motion":"mix","loop":true,"speedParam":"rate"}},
        {{"name":"once","motion":"once","loop":false,"speedParam":null}}],
        "transitions":[{{"from":"walk","to":"mix","when":[{{"param":"go","op":"trigger"}}],"blendTicks":4,"exitTick":null,"priority":0}}]}}],
        "rootMotion":{{"bone":"{bone}","mode":"{mode}"}}}}"#);
    let g = Graph::load_compiled(&doc, &rig, &clips).unwrap();
    let (p, s) = (Params::new(&g.graph), GraphState::new(&g));
    (rig, clips, g, p, s)
}
/// S17 reference: the clip's own extraction delta in `mode`.
fn reference(rig: &Rig, clip: &Clip, mode: RootMotion, tick: u32) -> [f32; 3] { Clip { root_motion: mode, ..clip.clone() }.root_delta_at(rig, tick).unwrap() }
fn at(rate: f32, g: &Compiled, rig: &Rig, clips: &[Clip], s: &mut GraphState, p: &mut Params) -> TickOut {
    p.set("rate", ParamValue::Float(rate)).unwrap();
    g.step_outputs(rig, clips, s, p).unwrap()
}
fn names(j: u32, out: &TickOut) -> Vec<String> { out.events.iter().map(|e| format!("{j}{}", e.name)).collect() }
#[test]
fn looping_deltas_match_s17_at_tick_zero_and_seams() {
    let (rig, clips, g, p, mut s) = setup("root", "xz", "walk", BASE);
    let (mut seen, mut xs) = (Vec::new(), Vec::new());
    for j in 0..10 {
        let out = g.step_outputs(&rig, &clips, &mut s, &p).unwrap();
        assert_eq!(out.root_delta.t, reference(&rig, &clips[WALK], RootMotion::Xz, j), "tick {j}");
        assert_eq!(out.root_delta.r, [0.0, 0.0, 0.0, 1.0]);
        // x and z return to rest [1, _, 3]; y keeps its sampled value.
        assert_eq!(out.pose[0].t, [1.0, clips[WALK].sample_track(0, j).unwrap()[1], 3.0]);
        xs.push(out.root_delta.t[0]);
        seen.extend(names(j, &out));
    }
    assert_eq!(xs, [0.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0]);
    assert_eq!(seen, ["0a", "3b", "3c", "4a", "7b", "7c", "8a"]);
}
#[test]
fn crossed_ticks_follow_speed_and_direction_once() {
    let (rig, clips, g, mut p, mut s) = setup("root", "xz", "walk", BASE);
    let mut seen = Vec::new();
    for j in 0..9 { seen.extend(names(j, &at(0.5, &g, &rig, &clips, &mut s, &mut p))); }
    assert_eq!(seen, ["0a", "6b", "6c", "8a"]);
    // Five ticks from 0: one full lap plus a tick, every event at most once.
    let out = at(5.0, &g, &rig, &clips, &mut s, &mut p);
    assert_eq!((out.root_delta.t, names(0, &out)), ([9.0, 0.0, 4.0], vec!["0b".into(), "0c".into(), "0a".into()]));
    // Backwards from 1 to 3 enters 0 then 3, across the seam.
    let out = at(-2.0, &g, &rig, &clips, &mut s, &mut p);
    assert_eq!((out.root_delta.t, names(0, &out)), ([-4.0, 0.0, -2.0], vec!["0a".into(), "0b".into(), "0c".into()]));
    let out = at(0.0, &g, &rig, &clips, &mut s, &mut p);
    assert_eq!((out.root_delta.t, out.events.len()), ([0.0; 3], 0));
}
#[test]
fn clamped_state_stops_after_its_end() {
    let (rig, clips, g, p, mut s) = setup("root", "xyz", "once", BASE);
    let mut seen = Vec::new();
    for j in 0..6 {
        let out = g.step_outputs(&rig, &clips, &mut s, &p).unwrap();
        let want = if (1..=3).contains(&j) { [1.0; 3] } else { [0.0; 3] };
        assert_eq!(out.root_delta.t, want);
        assert_eq!(out.root_delta.t, reference(&rig, &clips[ONCE], RootMotion::Xyz, j));
        assert_eq!(out.pose[0].t, [1.0, 2.0, 3.0]);
        seen.extend(names(j, &out));
    }
    assert_eq!(seen, ["3end"]);
}
#[test]
fn graph_bone_and_mode_select_extraction() {
    let second = |bone: &str, mode: &str| {
        let (rig, clips, g, p, mut s) = setup(bone, mode, "walk", BASE);
        g.step_outputs(&rig, &clips, &mut s, &p).unwrap();
        g.step_outputs(&rig, &clips, &mut s, &p).unwrap()
    };
    let out = second("child", "xyz");
    assert_eq!((out.root_delta.t, out.pose[1].t, out.pose[0].t), ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.5, 0.0]));
    let out = second("root", "none");
    assert_eq!((out.root_delta.t, out.root_delta.r, out.pose[0].t), ([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0, 0.5, 0.0]));
}
#[test]
fn blend_weight_changes_blend_deltas_and_dominant_events() {
    let (rig, clips, g, mut p, mut s) = setup("root", "xz", "mix", BASE);
    let mut seen = Vec::new();
    for j in 0..9 {
        let x = j as f32 / 8.0;
        p.set("x", ParamValue::Float(x)).unwrap();
        let out = g.step_outputs(&rig, &clips, &mut s, &p).unwrap();
        let (walk, run) = (reference(&rig, &clips[WALK], RootMotion::Xz, j), reference(&rig, &clips[RUN], RootMotion::Xz, j));
        for axis in 0..3 { assert!((out.root_delta.t[axis] - (walk[axis] * (1.0 - x) + run[axis] * x)).abs() < 1e-5, "{j}"); }
        seen.extend(names(j, &out));
    }
    assert_eq!(seen, ["0a", "3b", "3c", "4a", "8r"]);
}
#[test]
fn crossfade_scales_target_motion_and_silences_source() {
    let (rig, clips, g, mut p, mut s) = setup("root", "xz", "walk", BASE);
    p.set("x", ParamValue::Float(1.0)).unwrap();
    for _ in 0..3 { g.step_outputs(&rig, &clips, &mut s, &p).unwrap(); }
    p.fire("go").unwrap();
    let (mut seen, mut xs) = (Vec::new(), Vec::new());
    for j in 0..6 {
        let out = g.step_outputs(&rig, &clips, &mut s, &p).unwrap();
        xs.push(out.root_delta.t[0]);
        seen.extend(names(j, &out));
    }
    assert_eq!((xs, seen), (vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.0], vec!["0r".to_string(), "4r".into()]));
}
#[test]
fn layers_compose_deltas_and_shared_machines_emit_once() {
    let layers = |mask: &str| format!(r#"[{{"name":"base","machine":"m","mask":null,"weight":1,"additive":false}},
        {{"name":"top","machine":"m","mask":{mask},"weight":0.5,"additive":true}}]"#);
    for (mask, x) in [("null", 1.5), (r#"["child"]"#, 1.0)] {
        let (rig, clips, g, p, mut s) = setup("root", "xz", "walk", &layers(mask));
        let first = g.step_outputs(&rig, &clips, &mut s, &p).unwrap();
        assert_eq!(first.events, [GraphEvent { layer: 0, clip: WALK, tick: 0, name: "a".into() }]);
        assert_eq!(g.step_outputs(&rig, &clips, &mut s, &p).unwrap().root_delta.t, [x, 0.0, 0.0]);
    }
}
