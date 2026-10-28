use raa_anim::{Clip, Rig};
use raa_anim_graph::{Frame, GraphRuntime, GraphState, ParamValue, Params, codes};

const RIG: &str = r#"{"format":"raa.rig","version":1,"name":"rig-run","bones":[
{"name":"root","parent":null,"rest":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}},
{"name":"spine","parent":"root","rest":{"t":[0,1,0],"r":[0,0,0,1],"s":[1,1,1]}},
{"name":"tip","parent":"spine","rest":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}}],
"roles":{},"ikChains":[],"limits":[],"attach":{}}"#;

fn track(bone: &str, keys: &[(u32, [f32; 3])]) -> String {
    let keys: Vec<_> = keys.iter().map(|(tick, v)|
        format!(r#"{{"tick":{tick},"v":[{},{},{}],"interp":"LINEAR"}}"#, v[0], v[1], v[2])).collect();
    format!(r#"{{"bone":"{bone}","channel":"translation","keys":[{}]}}"#, keys.join(","))
}
fn clip(rig: &Rig, name: &str, tracks: &[String], events: &str) -> Clip {
    Clip::load(&format!(r#"{{"format":"raa.clip","version":2,"name":"{name}","rig":"rig-run","lengthTicks":4,
        "loop":true,"rootMotion":"none","tracks":[{}],"events":{events}}}"#, tracks.join(",")), rig).unwrap()
}
/// Base machine: walk (root x = tick), then on `go` a 2-tick fade to a walk/run
/// blend space on `w` (run: root x = 2 * tick). An additive layer masked to the
/// spine adds half of wave's z = tick from its first key. Root motion is xz.
fn runtime() -> GraphRuntime {
    let rig = Rig::load(RIG).unwrap();
    let clips = vec![
        clip(&rig, "walk", &[track("root", &[(0, [0.0; 3]), (4, [4.0, 0.0, 0.0])]), track("spine", &[(0, [0.0, 1.0, 0.0])])],
            r#"[{"tick":2,"name":"step"}]"#),
        clip(&rig, "run", &[track("root", &[(0, [0.0; 3]), (4, [8.0, 0.0, 0.0])]), track("spine", &[(0, [0.0, 3.0, 0.0])])],
            r#"[{"tick":0,"name":"kick"}]"#),
        clip(&rig, "wave", &[track("spine", &[(0, [0.0, 1.0, 0.0]), (4, [0.0, 1.0, 4.0])])], r#"[{"tick":1,"name":"wave"}]"#),
    ];
    let doc = r#"{"format":"raa.animgraph","version":1,"name":"g","rig":"rig-run",
        "params":[{"name":"w","type":"float","default":0.5},{"name":"go","type":"trigger"}],
        "blendSpaces":[{"name":"mix","params":["w"],"points":[{"clip":"walk","at":[0]},{"clip":"run","at":[1]}],"sync":false}],
        "layers":[{"name":"base","machine":"base","mask":null,"weight":1,"additive":false},
                  {"name":"upper","machine":"up","mask":["spine"],"weight":0.5,"additive":true}],
        "machines":[{"name":"base","entry":"walk","states":[
            {"name":"walk","motion":"walk","loop":true,"speedParam":null},{"name":"mix","motion":"mix","loop":true,"speedParam":null}],
            "transitions":[{"from":"walk","to":"mix","when":[{"param":"go","op":"trigger"}],"blendTicks":2,"exitTick":null,"priority":0}]},
          {"name":"up","entry":"wave","states":[{"name":"wave","motion":"wave","loop":true,"speedParam":null}],"transitions":[]}],
        "rootMotion":{"bone":"root","mode":"xz"}}"#;
    GraphRuntime::load(doc, &rig, &clips).unwrap()
}
/// Step `ticks` times with `w`, firing `go` on the listed ticks only.
fn run(rt: &GraphRuntime, state: &mut GraphState, w: f32, fire: &[u32], ticks: std::ops::Range<u32>) -> Vec<Frame> {
    ticks.map(|j| {
        let mut params = rt.new_params();
        params.set("w", ParamValue::Float(w)).unwrap();
        if fire.contains(&j) { params.fire("go").unwrap(); }
        rt.frame(state, &params).unwrap()
    }).collect()
}
fn events(frames: &[Frame]) -> Vec<String> {
    frames.iter().enumerate()
        .flat_map(|(j, f)| f.out.events.iter().map(move |e| format!("{j}:{}{}", e.layer, e.name))).collect()
}
#[test]
fn transitions_blending_and_masks_compose_in_tick_order() {
    let rt = runtime();
    let frames = run(&rt, &mut rt.new_state(), 0.5, &[3], 0..8);
    // Entry samples tick 0 with zero delta; the fade entered at tick 3 shows the
    // frozen source (zero delta), then half and full weight of 0.5 * 1 + 0.5 * 2.
    let dx: Vec<f32> = frames.iter().map(|f| f.out.root_delta.t[0]).collect();
    assert_eq!(dx, [0.0, 1.0, 1.0, 0.0, 0.75, 1.5, 1.5, 1.5]);
    for f in &frames {
        assert_eq!(f.out.root_delta.r, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(&f.out.root_delta.t[1..], [0.0, 0.0]);
        // Extracted x and z stay at rest; the root carries no other motion.
        assert_eq!(f.out.pose[0].t, [0.0; 3]);
    }
    // Spine: base y crossfades walk 1 to mix 2; the additive layer adds 0.5 * wave z.
    let spine: Vec<[f32; 3]> = frames.iter().map(|f| f.out.pose[1].t).collect();
    assert_eq!(spine, [[0.0, 1.0, 0.0], [0.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 1.0, 1.5],
        [0.0, 1.5, 0.0], [0.0, 2.0, 0.5], [0.0, 2.0, 1.0], [0.0, 2.0, 1.5]]);
    // Tip is masked in but untracked: rest from both layers.
    assert!(frames.iter().all(|f| f.out.pose[2].t == [0.0; 3]));
    // Events: walk's step, the fade entry's dominant walk tick 0 (none), wave per lap.
    assert_eq!(events(&frames), ["1:1wave", "2:0step", "5:0step", "5:1wave"]);
}
#[test]
fn matrices_come_from_raa_anim_pose() {
    let rt = runtime();
    for f in run(&rt, &mut rt.new_state(), 0.5, &[3], 0..6) {
        let pose = rt.rig().pose(&f.out.pose).unwrap();
        assert_eq!((&pose.local, &pose.model, &pose.skin), (&f.out.pose, &f.model, &f.skin));
        let spine = f.out.pose[1].t;
        assert_eq!(&f.model[1][12..15], spine);
        // Skin removes the spine's rest translation [0, 1, 0].
        assert_eq!(&f.skin[1][12..15], [spine[0], spine[1] - 1.0, spine[2]]);
    }
}
#[test]
fn entered_state_delivers_its_tick_zero_events_once() {
    let rt = runtime();
    let frames = run(&rt, &mut rt.new_state(), 1.0, &[2], 0..8);
    // At w = 1 run dominates: kick on entry (tick 2) and again after a lap.
    let base: Vec<_> = events(&frames).into_iter().filter(|e| e.contains(":0")).collect();
    assert_eq!(base, ["2:0kick", "6:0kick"]);
    let dx: Vec<f32> = frames.iter().map(|f| f.out.root_delta.t[0]).collect();
    assert_eq!(dx, [0.0, 1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
}
#[test]
fn repeated_inputs_and_restored_snapshots_repeat_outputs() {
    let rt = runtime();
    let (mut a, mut b) = (rt.new_state(), rt.new_state());
    let first = run(&rt, &mut a, 0.25, &[3, 9], 0..12);
    assert_eq!(first, run(&rt, &mut b, 0.25, &[3, 9], 0..12));
    assert_eq!(a, b);
    // Snapshot mid-fade, then continue both the live and the restored state.
    let mut live = rt.new_state();
    let head = run(&rt, &mut live, 0.25, &[3, 9], 0..4);
    assert_eq!(head[..], first[..4]);
    let mut bytes = Vec::new();
    live.write(&mut bytes);
    let mut restored = rt.read_state(&bytes).unwrap();
    assert_eq!(restored, live);
    let tail = run(&rt, &mut live, 0.25, &[3, 9], 4..12);
    assert_eq!(tail, run(&rt, &mut restored, 0.25, &[3, 9], 4..12));
    assert_eq!(tail[..], first[4..]);
}
#[test]
fn refused_inputs_leave_the_state_untouched() {
    let rt = runtime();
    let mut state = rt.new_state();
    run(&rt, &mut state, 0.5, &[], 0..2);
    let before = state.clone();
    let foreign = Params::new(&raa_anim_graph::Graph { params: Vec::new(), ..rt.graph().clone() });
    assert_eq!(rt.step(&mut state, &foreign).unwrap_err().code, codes::PARAM);
    let mut broken = state.clone();
    broken.layers.pop();
    assert_eq!(rt.step(&mut broken, &rt.new_params()).unwrap_err().code, codes::SCHEMA);
    broken = state.clone();
    broken.layers[1].state = 1;
    assert_eq!(rt.step(&mut broken, &rt.new_params()).unwrap_err().code, codes::SCHEMA);
    assert_eq!(state, before);
    assert_eq!(rt.step(&mut state, &rt.new_params()).unwrap(), run(&rt, &mut before.clone(), 0.5, &[], 2..3)[0].out);
}
