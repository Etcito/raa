use raa_anim::{Clip, LocalPose, Rig};
use raa_anim_graph::{Graph, GraphError, GraphState, Params};

const RIG: &str = r#"{"format":"raa.rig","version":1,"name":"rig-layers","bones":[
{"name":"root","parent":null,"rest":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}},
{"name":"spine","parent":"root","rest":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}},
{"name":"tip","parent":"spine","rest":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}}],
"roles":{},"ikChains":[],"limits":[],"attach":{}}"#;

fn rig() -> Rig { Rig::load(RIG).unwrap() }
fn vec3(v: [f32; 3]) -> String { format!("[{},{},{}]", v[0], v[1], v[2]) }
fn vec4(v: [f32; 4]) -> String { format!("[{},{},{},{}]", v[0], v[1], v[2], v[3]) }
fn track(bone: &str, channel: &str, values: &[(u32, String)]) -> String {
    let list: Vec<_> = values.iter()
        .map(|(tick, v)| format!(r#"{{"tick":{tick},"v":{v},"interp":"LINEAR"}}"#)).collect();
    format!(r#"{{"bone":"{bone}","channel":"{channel}","keys":[{}]}}"#, list.join(","))
}
fn clip(rig: &Rig, name: &str, length: u32, tracks: Vec<String>) -> Clip {
    let text = format!(r#"{{"format":"raa.clip","version":2,"name":"{name}","rig":"rig-layers","lengthTicks":{length},"loop":true,"rootMotion":"none","tracks":[{}],"events":[]}}"#, tracks.join(","));
    Clip::load(&text, rig).unwrap()
}
/// Constant local TRS at tick 0 for each bone.
fn constant(rig: &Rig, name: &str, bones: &[(&str, [f32; 3], [f32; 4], [f32; 3])]) -> Clip {
    let mut tracks = Vec::new();
    for (bone, t, r, s) in bones {
        tracks.push(track(bone, "translation", &[(0, vec3(*t))]));
        tracks.push(track(bone, "rotation", &[(0, vec4(*r))]));
        tracks.push(track(bone, "scale", &[(0, vec3(*s))]));
    }
    clip(rig, name, 10, tracks)
}
fn state(name: &str, motion: &str) -> String {
    format!(r#"{{"name":"{name}","motion":"{motion}","loop":true,"speedParam":null}}"#)
}
fn layer(name: &str, mask: &str, weight: f32, additive: bool) -> String {
    format!(r#"{{"name":"{name}","machine":"m","mask":{mask},"weight":{weight},"additive":{additive}}}"#)
}
fn document(layers: &[String], states: &[String], params: &str, spaces: &str) -> String {
    format!(r#"{{"format":"raa.animgraph","version":1,"name":"g","rig":"rig-layers","params":[{params}],"blendSpaces":[{spaces}],"layers":[{}],"machines":[{{"name":"m","entry":"s0","states":[{}],"transitions":[]}}],"rootMotion":{{"bone":"root","mode":"none"}}}}"#, layers.join(","), states.join(","))
}
/// Load `doc`, set each layer's state index and phase, then compose the layers.
fn run(rig: &Rig, doc: &str, clips: &[Clip], states: &[usize], phases: &[f32])
    -> Result<Vec<LocalPose>, GraphError> {
    let compiled = Graph::load_compiled(doc, rig, clips).unwrap();
    let params = Params::new(&compiled.graph);
    let mut state = GraphState::new(&compiled);
    for (l, (&state_index, &phase)) in state.layers.iter_mut().zip(states.iter().zip(phases)) {
        l.state = state_index;
        l.phase = phase;
    }
    compiled.compose_layers(rig, clips, &state, &params)
}
fn close(a: f32, b: f32) -> bool { (a - b).abs() < 1e-5 }
fn v3(v: [f32; 3], x: f32, y: f32, z: f32) -> bool {
    close(v[0], x) && close(v[1], y) && close(v[2], z)
}
fn q4(v: [f32; 4], x: f32, y: f32, z: f32, w: f32) -> bool {
    close(v[0], x) && close(v[1], y) && close(v[2], z) && close(v[3], w)
}

#[test]
fn weights_blend_and_document_order_decide_the_pose() {
    let rig = rig();
    let (id, one) = ([0.0, 0.0, 0.0, 1.0], [1.0; 3]);
    let clips = [
        constant(&rig, "a", &[("root", [10.0, 0.0, 0.0], id, one),
            ("spine", [0.0, 20.0, 0.0], id, one), ("tip", [0.0, 0.0, 30.0], id, one)]),
        constant(&rig, "b", &[("root", [-10.0, 0.0, 0.0], id, one),
            ("spine", [0.0, -20.0, 0.0], id, one), ("tip", [0.0, 0.0, -30.0], id, one)]),
    ];
    let states = [state("s0", "a"), state("s1", "b")];
    for (weight, expected) in [(0.0, [10.0, 20.0, 30.0]), (0.5, [0.0, 0.0, 0.0]),
        (1.0, [-10.0, -20.0, -30.0])] {
        let layers = [layer("base", "null", 1.0, false), layer("top", "null", weight, false)];
        let pose = run(&rig, &document(&layers, &states, "", ""), &clips, &[0, 1], &[0.0, 0.0]).unwrap();
        assert!(v3(pose[0].t, expected[0], 0.0, 0.0), "{weight}: {:?}", pose[0].t);
        assert!(v3(pose[1].t, 0.0, expected[1], 0.0), "{weight}: {:?}", pose[1].t);
        assert!(v3(pose[2].t, 0.0, 0.0, expected[2]), "{weight}: {:?}", pose[2].t);
    }
    // Reversing the document order makes the later layer win at full weight.
    let layers = [layer("base", "null", 1.0, false), layer("top", "null", 1.0, false)];
    let pose = run(&rig, &document(&layers, &states, "", ""), &clips, &[1, 0], &[0.0, 0.0]).unwrap();
    assert!(v3(pose[0].t, 10.0, 0.0, 0.0));
    assert!(v3(pose[2].t, 0.0, 0.0, 30.0));
}

#[test]
fn compiled_masks_include_descendants_and_later_layers_overlap() {
    let rig = rig();
    let (id, one) = ([0.0, 0.0, 0.0, 1.0], [1.0; 3]);
    let clips = [
        constant(&rig, "base", &[("root", [10.0, 0.0, 0.0], id, one),
            ("spine", [0.0, 20.0, 0.0], id, one), ("tip", [0.0, 0.0, 30.0], id, one)]),
        constant(&rig, "mid", &[("root", [-1.0; 3], id, one),
            ("spine", [-2.0; 3], id, one), ("tip", [-3.0; 3], id, one)]),
        constant(&rig, "top", &[("root", [5.0; 3], id, one),
            ("spine", [6.0; 3], id, one), ("tip", [7.0; 3], id, one)]),
    ];
    let states = [state("s0", "base"), state("s1", "mid"), state("s2", "top")];
    let layers = [layer("base", "null", 1.0, false), layer("mid", "[\"spine\"]", 1.0, false)];
    let pose = run(&rig, &document(&layers, &states, "", ""), &clips, &[0, 1], &[0.0, 0.0]).unwrap();
    assert!(v3(pose[0].t, 10.0, 0.0, 0.0)); // root is not in the mask
    assert!(v3(pose[1].t, -2.0, -2.0, -2.0)); // spine is selected directly
    assert!(v3(pose[2].t, -3.0, -3.0, -3.0)); // tip is selected as a descendant
    let layers = [layer("base", "null", 1.0, false), layer("mid", "[\"spine\"]", 1.0, false),
        layer("top", "[\"tip\"]", 1.0, false)];
    let pose = run(&rig, &document(&layers, &states, "", ""), &clips, &[0, 1, 2], &[0.0; 3]).unwrap();
    assert!(v3(pose[0].t, 10.0, 0.0, 0.0));
    assert!(v3(pose[1].t, -2.0, -2.0, -2.0));
    assert!(v3(pose[2].t, 7.0, 7.0, 7.0)); // a later overlapping layer wins
}

#[test]
fn additive_layer_is_neutral_at_the_first_key_and_adds_each_channel() {
    let rig = rig();
    let (id, one) = ([0.0, 0.0, 0.0, 1.0], [1.0; 3]);
    let clips = [
        constant(&rig, "base", &[("root", [100.0, 0.0, 0.0], id, [2.0; 3])]),
        clip(&rig, "add", 10, vec![
            // The first translation key is at tick 2, so phase 0 and 1 both clamp to it.
            track("root", "translation", &[(2, vec3([4.0, 5.0, 6.0])),
                (10, vec3([14.0, 15.0, 16.0]))]),
            track("root", "rotation", &[(0, vec4([0.0, 0.0, 0.0, 1.0])),
                (10, vec4([0.0, 0.0, 1.0, 0.0]))]),
            track("root", "scale", &[(0, vec3(one)), (10, vec3([3.0, 5.0, 7.0]))]),
        ]),
    ];
    let states = [state("s0", "base"), state("s1", "add")];
    let layers = [layer("base", "null", 1.0, false), layer("add", "null", 1.0, true)];
    let doc = document(&layers, &states, "", "");
    let pose = run(&rig, &doc, &clips, &[0, 1], &[0.0, 0.0]).unwrap();
    assert!(v3(pose[0].t, 100.0, 0.0, 0.0));
    assert!(v3(pose[0].s, 2.0, 2.0, 2.0));
    assert!(q4(pose[0].r, 0.0, 0.0, 0.0, 1.0));
    // Before the translation's first key the whole additive layer is neutral.
    let pose = run(&rig, &doc, &clips, &[0, 1], &[0.0, 1.0]).unwrap();
    assert!(v3(pose[0].t, 100.0, 0.0, 0.0), "{:?}", pose[0].t);
    let pose = run(&rig, &doc, &clips, &[0, 1], &[0.0, 5.0]).unwrap();
    assert!(v3(pose[0].t, 103.75, 3.75, 3.75)); // + (7.75, 8.75, 9.75) - (4, 5, 6)
    assert!(v3(pose[0].s, 3.0, 4.0, 5.0)); // + (2, 3, 4) - (1, 1, 1)
    assert!(q4(pose[0].r, 0.0, 0.0, 0.70710677, 0.70710677));
    let layers = [layer("base", "null", 1.0, false), layer("add", "null", 0.5, true)];
    let pose = run(&rig, &document(&layers, &states, "", ""), &clips, &[0, 1], &[0.0, 5.0]).unwrap();
    assert!(v3(pose[0].t, 101.875, 1.875, 1.875));
    assert!(v3(pose[0].s, 2.5, 3.0, 3.5));
    assert!(q4(pose[0].r, 0.0, 0.0, 0.38268343, 0.9238795));
    let layers = [layer("base", "null", 1.0, false), layer("add", "null", 0.0, true)];
    let pose = run(&rig, &document(&layers, &states, "", ""), &clips, &[0, 1], &[0.0, 5.0]).unwrap();
    assert!(v3(pose[0].t, 100.0, 0.0, 0.0));
    assert!(v3(pose[0].s, 2.0, 2.0, 2.0));
}

#[test]
fn additive_blend_space_delta_uses_the_blended_first_keys() {
    let rig = rig();
    let (id, one) = ([0.0, 0.0, 0.0, 1.0], [1.0; 3]);
    let clips = [
        constant(&rig, "base", &[("root", [100.0, 0.0, 0.0], id, [2.0; 3])]),
        clip(&rig, "p0", 10, vec![
            track("root", "translation", &[(0, vec3([0.0; 3])), (10, vec3([10.0, 0.0, 0.0]))]),
            track("root", "rotation", &[(0, vec4(id)), (10, vec4([0.0, 0.0, 0.6, 0.8]))]),
            track("root", "scale", &[(0, vec3(one)), (10, vec3([2.0, 1.0, 1.0]))]),
        ]),
        clip(&rig, "p1", 10, vec![
            track("root", "translation", &[(0, vec3([0.0; 3])), (10, vec3([0.0, 10.0, 0.0]))]),
            track("root", "rotation", &[(0, vec4(id)), (10, vec4([0.0, 0.6, 0.0, 0.8]))]),
            track("root", "scale", &[(0, vec3(one)), (10, vec3([1.0, 3.0, 1.0]))]),
        ]),
    ];
    let spaces = r#"{"name":"locomotion","params":["x"],"points":[{"clip":"p0","at":[0]},{"clip":"p1","at":[1]}],"sync":true}"#;
    let params = r#"{"name":"x","type":"float","default":0.5}"#;
    let states = [state("s0", "base"), state("s1", "locomotion")];
    let layers = [layer("base", "null", 1.0, false), layer("add", "null", 1.0, true)];
    let doc = document(&layers, &states, params, spaces);
    let pose = run(&rig, &doc, &clips, &[0, 1], &[0.0, 0.0]).unwrap();
    assert!(v3(pose[0].t, 100.0, 0.0, 0.0) && v3(pose[0].s, 2.0, 2.0, 2.0));
    assert!(q4(pose[0].r, 0.0, 0.0, 0.0, 1.0));
    // Tick 5 blends (5,0,0) and (0,5,0); both first keys are zero.
    let pose = run(&rig, &doc, &clips, &[0, 1], &[0.0, 0.5]).unwrap();
    assert!(v3(pose[0].t, 102.5, 2.5, 0.0));
    assert!(v3(pose[0].s, 2.25, 2.5, 2.0));
    assert!(q4(pose[0].r, 0.0, 0.16222, 0.16222, 0.97333));
}
