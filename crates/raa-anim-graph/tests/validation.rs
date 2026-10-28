use raa_anim::{Clip, JsonValue as J, Rig, parse, to_canonical_json};
use raa_anim_graph::{GRAPH_CAPS as C, Graph};
use raa_anim_graph::load::validate::Motion;
const DOC: &str = r#"{"format":"raa.animgraph","version":1,"name":"g","rig":"rig-basic","params":[{"name":"x","type":"float","default":0},{"name":"y","type":"int","default":0},{"name":"b","type":"bool","default":false},{"name":"t","type":"trigger"}],"blendSpaces":[{"name":"blend","params":["x"],"points":[{"clip":"idle","at":[0]},{"clip":"run","at":[1]}]}],"layers":[{"name":"base","machine":"m","mask":null,"weight":1,"additive":false}],"machines":[{"name":"m","entry":"s","states":[{"name":"s","motion":"blend","loop":true,"speedParam":"y"},{"name":"other","motion":"run","loop":false,"speedParam":null}],"transitions":[{"from":"*","to":"other","when":[{"param":"t","op":"trigger"}],"blendTicks":6,"exitTick":null,"priority":0}]}],"rootMotion":{"bone":"root","mode":"none"}}"#;
fn node<'a>(v: &'a mut J, path: &str) -> &'a mut J {
    let mut v = v;
    for key in path.split('/').filter(|s| !s.is_empty()) {
        v = match v { J::Object(o) => &mut o.iter_mut().find(|(k, _)| k == key).unwrap().1,
            J::Array(a) => &mut a[key.parse::<usize>().unwrap()], _ => panic!("bad path {path}") };
    }
    v
}
fn set(v: &mut J, path: &str, value: &str) { *node(v, path) = parse(value).unwrap(); }
fn assets() -> (Rig, Vec<Clip>) {
    let mut rig = Rig::load(include_str!("../../raa-anim/tests/fixtures/rig-basic.json")).unwrap();
    for i in 2..C.max_mask_bones {
        let mut bone = rig.bones[1].clone(); bone.name = format!("b{i}"); bone.parent = Some(if i == 2 { 0 } else { i - 1 }); rig.bones.push(bone);
    }
    let clips = ["idle", "run"].iter().map(|name| Clip::load(&format!(r#"{{"format":"raa.clip","version":2,"name":"{name}","rig":"rig-basic","lengthTicks":10,"loop":true,"rootMotion":"none","tracks":[],"events":[]}}"#), &rig).unwrap()).collect();
    (rig, clips)
}
#[test]
fn indices_and_masks_follow_supplied_order() {
    let (rig, mut clips) = assets(); clips.reverse();
    let mut doc = parse(DOC).unwrap();
    set(&mut doc, "/layers/0/mask", r#"["b3","child","b2"]"#);
    set(&mut doc, "/machines/0/transitions/0/from", r#""s""#);
    let (g, refs) = Graph::load_resolved(&to_canonical_json(&doc), &rig, &clips).unwrap();
    assert_eq!(g.layers[0].mask.as_ref().unwrap(), &["b3", "child", "b2"]);
    assert_eq!(refs.layers[0].bones, (1..C.max_mask_bones).collect::<Vec<_>>());
    assert_eq!((refs.layers[0].machine, refs.root_bone, refs.machines[0].entry), (0, 0, 0));
    assert_eq!((&refs.blends[0].params, &refs.blends[0].clips), (&vec![0], &vec![1, 0]));
    assert_eq!((refs.machines[0].states[0].motion, refs.machines[0].states[0].speed_param), (Motion::BlendSpace(0), Some(1)));
    assert_eq!(refs.machines[0].states[1].motion, Motion::Clip(0));
    let t = &refs.machines[0].transitions[0]; assert_eq!((t.from, t.to, &t.params), (Some(0), 1, &vec![3]));
    for (mask, expected) in [("null", (0..C.max_mask_bones).collect()), ("[]", vec![]), (r#"["child"]"#, vec![1])] {
        set(&mut doc, "/layers/0/mask", mask);
        let (_, refs) = Graph::load_resolved(&to_canonical_json(&doc), &rig, &clips).unwrap(); assert_eq!(refs.layers[0].bones, expected);
    }
    let (_, refs) = Graph::load_resolved(DOC, &rig, &clips).unwrap(); assert_eq!(refs.machines[0].transitions[0].from, None);
}
#[test]
fn refuses_unknown_duplicate_and_ambiguous_references() {
    let (rig, clips) = assets();
    for (path, value, code) in [
        ("/rig", r#""other""#, "graph.schema"),
        ("/machines/0/states/0/motion", r#""missing""#, "graph.clip_unknown"),
        ("/blendSpaces/0/points/0/clip", r#""missing""#, "graph.clip_unknown"),
        ("/blendSpaces/0/name", r#""idle""#, "graph.blend_space"),
        ("/blendSpaces/0/params/0", r#""missing""#, "graph.blend_space"),
        ("/machines/0/states/0/speedParam", r#""missing""#, "graph.param"),
        ("/machines/0/states/0/speedParam", r#""b""#, "graph.param"),
        ("/machines/0/entry", r#""missing""#, "graph.state_unknown"),
        ("/machines/0/transitions/0/from", r#""missing""#, "graph.state_unknown"),
        ("/machines/0/transitions/0/to", r#""missing""#, "graph.state_unknown"),
        ("/layers/0/machine", r#""missing""#, "graph.state_unknown"),
        ("/rootMotion/bone", r#""missing""#, "graph.mask"),
        ("/layers/0/mask", r#"["missing"]"#, "graph.mask"),
        ("/layers/0/mask", r#"["root","root"]"#, "graph.mask"),
        ("/params/1/name", r#""x""#, "graph.param"),
        ("/machines/0/states/1/name", r#""s""#, "graph.schema"),
    ] {
        let mut doc = parse(DOC).unwrap(); set(&mut doc, path, value);
        assert_eq!(Graph::load(&to_canonical_json(&doc), &rig, &clips).unwrap_err().code, code, "{path}");
    }
    for (path, code) in [("/machines", "graph.schema"), ("/layers", "graph.schema"), ("/blendSpaces", "graph.blend_space")] {
        let mut doc = parse(DOC).unwrap(); let J::Array(a) = node(&mut doc, path) else { panic!() }; a.push(a[0].clone());
        assert_eq!(Graph::load(&to_canonical_json(&doc), &rig, &clips).unwrap_err().code, code);
    }
    let mut bad = clips.clone(); bad.push(bad[0].clone()); assert_eq!(Graph::load(DOC, &rig, &bad).unwrap_err().code, "graph.clip_unknown");
    bad = clips.clone(); bad[0].rig = "other".into(); assert_eq!(Graph::load(DOC, &rig, &bad).unwrap_err().code, "graph.clip_unknown");
    bad = clips.clone();
    let mut tracked = Clip::load(include_str!("../../raa-anim/tests/fixtures/clip-linear-constant.json"), &rig).unwrap();
    tracked.tracks[0].bone = "missing".into(); bad.push(tracked);
    assert_eq!(Graph::load(DOC, &rig, &bad).unwrap_err().code, "graph.clip_unknown");
    let mut bad_rig = rig.clone(); bad_rig.bones[1].name = "root".into(); assert_eq!(Graph::load(DOC, &bad_rig, &clips).unwrap_err().code, "graph.mask");
}
#[test]
fn condition_operands_and_geometry() {
    let (rig, clips) = assets();
    for (param, op, value, valid) in [("x", ">", "1.5", true), ("y", "<=", "2", true), ("b", "!=", "true", true), ("t", "trigger", "", true), ("missing", "==", "0", false), ("x", "==", "true", false), ("y", ">", "1.5", false), ("b", "==", "1", false), ("b", ">", "false", false), ("x", "trigger", "", false), ("t", "==", "1", false), ("t", "trigger", "0", false)] {
        let mut doc = parse(DOC).unwrap(); let operand = if value.is_empty() { String::new() } else { format!(",\"value\":{value}") };
        set(&mut doc, "/machines/0/transitions/0/when", &format!(r#"[{{"param":"{param}","op":"{op}"{operand}}}]"#));
        let result = Graph::load(&to_canonical_json(&doc), &rig, &clips);
        if valid { result.unwrap(); } else { assert_eq!(result.unwrap_err().code, "graph.param"); }
    }
    for (params, coords, valid) in [(r#"["x"]"#, vec!["[1]", "[0]"], true), (r#"["x"]"#, vec!["[0]", "[0]"], false), (r#"["x"]"#, vec!["[1]", "[1.00000001]"], false), (r#"["x"]"#, vec!["[-3e38]", "[3e38]"], false), (r#"["x","y"]"#, vec!["[0,0]", "[1,1]"], false), (r#"["x","y"]"#, vec!["[0,0]", "[1,1]", "[2,2]"], false), (r#"["x","y"]"#, vec!["[0,0]", "[1,0]", "[0,1]", "[2,0]"], true), (r#"["x","y"]"#, vec!["[0,0]", "[3e38,0]", "[0,3e38]"], false)] {
        let mut doc = parse(DOC).unwrap(); set(&mut doc, "/blendSpaces/0/params", params);
        let points = coords.iter().map(|at| format!(r#"{{"clip":"idle","at":{at}}}"#)).collect::<Vec<_>>().join(","); set(&mut doc, "/blendSpaces/0/points", &format!("[{points}]"));
        let result = Graph::load(&to_canonical_json(&doc), &rig, &clips);
        if valid { result.unwrap(); } else { assert_eq!(result.unwrap_err().code, "graph.blend_space"); }
    }
}
#[test]
fn every_cap_is_inclusive_and_never_truncates() {
    let (rig, clips) = assets();
    for (path, cap) in [("/params", C.max_params), ("/blendSpaces", C.max_blend_spaces), ("/blendSpaces/0/points", C.max_points_per_space), ("/layers", C.max_layers), ("/machines", C.max_machines), ("/machines/0/states", C.max_states_per_machine), ("/machines/0/transitions", C.max_transitions_per_machine), ("/machines/0/transitions/0/when", C.max_conditions_per_transition), ("/layers/0/mask", C.max_mask_bones)] {
        for count in [cap, cap + 1] {
            let mut doc = parse(DOC).unwrap();
            if path.ends_with("mask") { *node(&mut doc, path) = J::Array((0..count).map(|i| J::String(rig.bones[i % rig.bones.len()].name.clone())).collect()); }
            else {
                let J::Array(a) = node(&mut doc, path) else { panic!() }; let original = a.clone();
                *a = (0..count).map(|i| {
                    let mut item = original[i % original.len()].clone();
                    if i >= original.len() && !path.ends_with("transitions") && !path.ends_with("when") && !path.ends_with("points") { *node(&mut item, "name") = J::String(format!("n{i}")); }
                    if path.ends_with("points") { *node(&mut item, "at/0") = J::Number((i as u32).into()); } item
                }).collect();
            }
            let result = Graph::load_resolved(&to_canonical_json(&doc), &rig, &clips);
            if count > cap { let e = result.unwrap_err(); assert_eq!((e.code, e.path.as_str()), ("graph.cap", path)); continue; }
            let (g, r) = result.unwrap();
            let retained = match path {
                "/params" => g.params.len(), "/blendSpaces" => r.blends.len(), "/blendSpaces/0/points" => r.blends[0].clips.len(),
                "/layers" => r.layers.len(), "/machines" => r.machines.len(), "/machines/0/states" => r.machines[0].states.len(),
                "/machines/0/transitions" => r.machines[0].transitions.len(), "/machines/0/transitions/0/when" => r.machines[0].transitions[0].params.len(), _ => r.layers[0].bones.len(),
            }; assert_eq!(retained, cap, "{path}");
        }
    }
}
