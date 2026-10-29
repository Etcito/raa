use raa_anim_graph::scenario_data::{PENGUIN_CLIPS, PENGUIN_GRAPH, PENGUIN_TIMELINE, Timeline, penguin_clips, penguin_rig};
use raa_anim_graph::{GraphRuntime, ParamType, codes};

const S19_NAMES: [&str; 15] = ["idle", "waddle", "run", "jump", "dive", "bellySlide", "handPush", "grab", "hang",
    "climbUp", "fall", "emote.wave", "emote.dance", "emote.cheer", "emote.laugh"];

fn runtime() -> GraphRuntime {
    let rig = penguin_rig();
    GraphRuntime::load(PENGUIN_GRAPH, &rig, &penguin_clips(&rig)).unwrap()
}
#[test]
fn documents_load_with_every_penguin_clip() {
    let rt = runtime();
    let mut names: Vec<_> = PENGUIN_CLIPS.iter().map(|s| s.0).collect();
    let mut want = S19_NAMES.to_vec();
    names.sort();
    want.sort();
    assert_eq!(names, want);
    let g = rt.graph();
    assert_eq!(g.blend_spaces.iter().map(|b| b.params.len()).collect::<Vec<_>>(), [1, 2]);
    assert!(g.layers[1].additive && g.layers[1].mask.is_some());
}
#[test]
fn timeline_boundaries_are_valid() {
    let rt = runtime();
    let t = Timeline::load(PENGUIN_TIMELINE, &rt.new_params()).unwrap();
    assert_eq!(t.ticks, 600);
    assert_eq!(t.keys[0].tick, 0);
    assert!(t.keys.windows(2).all(|w| w[0].tick <= w[1].tick) && t.keys.iter().all(|k| k.tick < 600));
    let bad = PENGUIN_TIMELINE.replace(r#""tick":580"#, r#""tick":600"#);
    assert_eq!(Timeline::load(&bad, &rt.new_params()).unwrap_err().code, codes::SCHEMA);
    let bad = PENGUIN_TIMELINE.replace(r#""tick":40"#, r#""tick":0"#).replace(r#""tick":0,"param":"speed","value":0"#, r#""tick":41,"param":"speed","value":0"#);
    assert!(Timeline::load(&bad, &rt.new_params()).is_err());
    let bad = PENGUIN_TIMELINE.replace(r#""value":1.5"#, r#""value":true"#);
    assert_eq!(Timeline::load(&bad, &rt.new_params()).unwrap_err().code, codes::PARAM);
}
#[test]
fn timeline_drives_every_state_with_motion_and_events() {
    let rt = runtime();
    let t = Timeline::load(PENGUIN_TIMELINE, &rt.new_params()).unwrap();
    let (mut state, mut held) = (rt.new_state(), rt.new_params());
    let (mut base, mut emote, mut events, mut travel) = (vec![], vec![], 0, 0.0f32);
    for tick in 0..t.ticks {
        let mut params = held.clone();
        t.apply(tick, &mut params).unwrap();
        let out = rt.step(&mut state, &params).unwrap();
        events += out.events.len();
        travel += out.root_delta.t[2];
        for (i, s) in [(0, &mut base), (1, &mut emote)] {
            let name = &rt.graph().machines[state.layers[i].machine].states[state.layers[i].state].name;
            if !s.contains(name) { s.push(name.clone()); }
        }
        for i in 0..params.len() { if params.ty(i) == Some(ParamType::Trigger) { params.consume(i); } }
        held = params;
    }
    assert_eq!(base, ["locomotion", "jump", "fall", "dive", "bellySlide", "grab", "hang", "climbUp", "handPush"]);
    assert_eq!(emote.len(), 4);
    assert!(events > 20 && travel > 5.0, "events {events} travel {travel}");
}
