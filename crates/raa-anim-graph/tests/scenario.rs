use raa_anim_graph::scenario_data::{
    PENGUIN_GRAPH, PENGUIN_TIMELINE, Timeline, penguin_clips, penguin_rig,
};
use raa_anim_graph::{GraphRuntime, ScenarioRun, run, run_penguin};

const BASE_ROUTE: [(u32, &str); 13] = [
    (0, "locomotion"), (150, "jump"), (163, "fall"), (200, "locomotion"), (240, "dive"),
    (251, "bellySlide"), (330, "locomotion"), (360, "grab"), (369, "hang"), (420, "climbUp"),
    (443, "locomotion"), (470, "handPush"), (520, "locomotion"),
];
const EMOTE_ROUTE: [(u32, &str); 5] = [
    (0, "emote.wave"), (300, "emote.dance"), (450, "emote.cheer"),
    (550, "emote.laugh"), (580, "emote.wave"),
];
const EVENT_SCRIPT: [&str; 49] = [
    "10:1:wave", "20:0:breathe", "30:1:wave", "46:0:step.l", "50:1:wave",
    "58:0:step.r", "70:0:step.l", "70:1:wave", "82:0:step.r", "90:1:wave",
    "94:0:step.l", "104:0:step.r", "110:1:wave", "112:0:step.l", "120:0:step.r",
    "128:0:step.l", "130:1:wave", "136:0:step.r", "144:0:step.l", "150:1:wave",
    "152:0:takeoff", "169:0:flail", "170:1:wave", "181:0:flail", "190:1:wave",
    "193:0:flail", "210:1:wave", "220:0:breathe", "230:1:wave", "248:0:splash",
    "250:1:wave", "259:0:splash", "270:1:wave", "271:0:splash", "283:0:splash",
    "290:1:wave", "295:0:splash", "307:0:splash", "319:0:splash", "350:0:breathe",
    "365:0:grab", "384:0:sway", "414:0:sway", "440:0:ledge", "463:0:breathe",
    "480:0:push", "500:0:push", "540:0:breathe", "581:0:breathe",
];

fn route(run: &ScenarioRun, layer: usize) -> Vec<(u32, String)> {
    run.route.iter().filter(|r| r.layer == layer)
        .map(|r| (r.tick, r.state.clone())).collect()
}
fn route_matches(run: &ScenarioRun, layer: usize, expected: &[(u32, &str)]) -> bool {
    route(run, layer) == expected.iter().map(|&(tick, name)| (tick, name.to_string())).collect::<Vec<_>>()
}
fn window(run: &ScenarioRun, a: u32, b: u32) -> f32 {
    run.trace[a as usize..b as usize].iter().map(|s| s.out.root_delta.t[2]).sum()
}
fn base_only() -> ScenarioRun {
    let rig = penguin_rig();
    let graph = PENGUIN_GRAPH.replace("\"weight\":0.8", "\"weight\":0");
    assert_ne!(graph, PENGUIN_GRAPH, "emote layer weight must exist to mask it out");
    let rt = GraphRuntime::load(&graph, &rig, &penguin_clips(&rig)).unwrap();
    let timeline = Timeline::load(PENGUIN_TIMELINE, &rt.new_params()).unwrap();
    run(&rt, &timeline).unwrap()
}

#[test]
fn route_and_root_movement_follow_the_script() {
    let run = run_penguin().unwrap();
    assert_eq!(run.steps(), 600);
    assert!(route_matches(&run, 0, &BASE_ROUTE), "base route: {:?}", route(&run, 0));
    assert!(route_matches(&run, 1, &EMOTE_ROUTE), "emote route: {:?}", route(&run, 1));
    assert_eq!(window(&run, 0, 40), 0.0);
    assert!(window(&run, 40, 100) > 2.0, "waddle travel");
    assert!(window(&run, 100, 150) > window(&run, 40, 100), "run must outrun waddle");
    assert!(window(&run, 150, 163) > 0.0, "jump travel");
    assert!(window(&run, 163, 200) > 0.0, "fall travel");
    assert_eq!(window(&run, 200, 240), 0.0);
    assert!(window(&run, 240, 251) > 0.5, "dive travel");
    assert!(window(&run, 251, 330) > 5.5, "belly slide travel");
    assert!(window(&run, 360, 369) > 0.0, "grab travel");
    assert!(window(&run, 420, 443) > 0.2, "climb travel");
    assert!(window(&run, 470, 520) > 0.9, "hand push travel");
    assert_eq!(window(&run, 550, 600), 0.0, "emotes must not travel");
    assert!((run.travel[2] - 16.725273).abs() < 1e-4, "total {:?}", run.travel);
    assert_eq!(&run.travel[..2], [0.0, 0.0]);
    for step in &run.trace {
        assert_eq!((step.out.pose[0].t[0], step.out.pose[0].t[2]), (0.0, 0.0),
            "extracted root axes return to rest at {}", step.tick);
        assert_eq!(step.out.root_delta.r, [0.0, 0.0, 0.0, 1.0]);
        assert!(step.out.root_delta.t[2] >= 0.0, "root must not move backwards at {}", step.tick);
    }
}

#[test]
fn emote_layer_is_masked_to_spine_and_its_descendants() {
    let full = run_penguin().unwrap();
    let base = base_only();
    assert_eq!(base.steps(), 600);
    let (mut moved, mut masked_events) = (0, 0);
    for (a, b) in full.trace.iter().zip(&base.trace) {
        for bone in [0, 1, 6, 7] {
            assert_eq!(a.out.pose[bone], b.out.pose[bone], "unmasked bone {bone} at {}", a.tick);
        }
        moved += u32::from(a.out.pose[4] != b.out.pose[4]);
    }
    for event in &full.events {
        if event.event.layer == 1 { masked_events += 1; }
        assert!(base.events.iter().all(|e| e.event.layer == 0));
    }
    assert!(moved > 100, "masked flipper barely moved: {moved}");
    assert_eq!(masked_events, 15);
}

#[test]
fn events_are_reported_once_per_tick_in_crossing_order() {
    let run = run_penguin().unwrap();
    let got: Vec<String> = run.events.iter()
        .map(|e| format!("{}:{}:{}", e.tick, e.event.layer, e.event.name)).collect();
    assert_eq!(got, EVENT_SCRIPT);
    assert!(run.events.windows(2).all(|w| w[0].tick <= w[1].tick));
    assert_eq!(run.trace.iter().map(|s| s.out.events.len()).sum::<usize>(), EVENT_SCRIPT.len());
}

#[test]
fn rollback_from_a_checkpoint_inside_a_crossfade_replays_identically() {
    let run = run_penguin().unwrap();
    let rollback = run.rollback.as_ref().expect("the script must exercise rollback");
    assert_eq!(rollback.checkpoint, 151);
    assert_eq!(rollback.replayed, 448);
    assert_eq!(rollback.fades, [0], "checkpoint must be inside a running crossfade");
    assert!(rollback.matched, "restored replay diverged at {:?}", rollback.first_mismatch);
    assert_eq!(rollback.first_mismatch, None);
    let base = route(&run, 0);
    assert!(base.iter().any(|(tick, state)| *tick == 150 && state == "jump"));
    assert!(base.iter().any(|(tick, state)| *tick == 163 && state == "fall"));
    assert!(run.trace[151].out.pose != run.trace[150].out.pose, "fade is progressing");
}

#[test]
fn repeated_runs_are_identical_and_bounded() {
    let first = run_penguin().unwrap();
    let second = run_penguin().unwrap();
    assert_eq!(first, second);
    assert_eq!(first.digest(), second.digest());
    assert!(first.steps() <= raa_anim_graph::SCENARIO_TICKS);
}
