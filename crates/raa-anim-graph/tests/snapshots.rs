use raa_anim::{Clip, Rig};
use raa_anim_graph::{GraphRuntime, GraphState, ParamValue};

fn runtime() -> GraphRuntime {
    let rig = Rig::load(include_str!("../../raa-anim/tests/fixtures/rig-basic.json")).unwrap();
    let clips: Vec<_> = [("slow", 4), ("fast", 8)].iter().map(|(name, length)| {
        Clip::load(&format!(r#"{{"format":"raa.clip","version":2,"name":"{name}","rig":"rig-basic",
            "lengthTicks":{length},"loop":true,"rootMotion":"none","tracks":[{{"bone":"root",
            "channel":"translation","keys":[{{"tick":0,"v":[0,0,0],"interp":"LINEAR"}},
            {{"tick":{length},"v":[8,0,0],"interp":"LINEAR"}}]}}],
            "events":[{{"tick":0,"name":"entry"}},{{"tick":2,"name":"step"}}]}}"#), &rig).unwrap()
    }).collect();
    let doc = r#"{"format":"raa.animgraph","version":1,"name":"snap","rig":"rig-basic",
        "params":[{"name":"speed","type":"float","default":0.75},{"name":"i","type":"int","default":0},
          {"name":"flag","type":"bool","default":false},{"name":"go","type":"trigger"}],
        "blendSpaces":[{"name":"mix","params":["speed"],"points":[{"clip":"slow","at":[0]},
          {"clip":"fast","at":[1]}],"sync":true}],
        "layers":[{"name":"base","machine":"m","mask":null,"weight":1,"additive":false}],
        "machines":[{"name":"m","entry":"a","states":[
          {"name":"a","motion":"mix","loop":true,"speedParam":"speed"},
          {"name":"b","motion":"fast","loop":true,"speedParam":"speed"}],"transitions":[
          {"from":"*","to":"b","when":[{"param":"go","op":"trigger"}],"blendTicks":3,"exitTick":3,"priority":0}]}],
        "rootMotion":{"bone":"root","mode":"xz"}}"#;
    GraphRuntime::load(doc, &rig, &clips).unwrap()
}
fn bytes(s: &GraphState) -> Vec<u8> { let mut b = Vec::new(); s.write(&mut b); b }
fn set(b: &mut [u8], offset: usize, v: u32) { b[offset..offset + 4].copy_from_slice(&v.to_le_bytes()); }
fn offsets(b: &[u8]) -> (usize, usize) {
    let p = 8 + u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    (p, p + 16) // Four parameter words precede the cursor payload.
}
#[test]
fn rollback_repeats_frames_across_triggers_fades_sync_and_event_boundaries() {
    let rt = runtime();
    let mut live = rt.new_state();
    let mut input = rt.new_params();
    input.set("i", ParamValue::Int(i32::MIN)).unwrap();
    input.set("flag", ParamValue::Bool(true)).unwrap();
    for tick in 0..24 {
        if tick == 1 || tick == 5 { input.fire("go").unwrap(); }
        rt.frame(&mut live, &input).unwrap();
        let saved = bytes(&live);
        let mut restored = rt.read_state(&saved).unwrap();
        assert_eq!(restored, live);
        assert_eq!(bytes(&restored), saved);
        let mut expected = live.clone();
        let mut future = restored.params.clone();
        future.consume(3);
        for _ in 0..18 {
            assert_eq!(rt.frame(&mut expected, &future).unwrap(), rt.frame(&mut restored, &future).unwrap());
        }
        assert_eq!(restored, expected);
        input.consume(3);
    }
}
#[test]
fn stable_little_endian_words_and_append_semantics() {
    let rt = runtime();
    let mut s = rt.new_state();
    s.tick = 0x12345678;
    s.params.set("speed", ParamValue::Float(-0.0)).unwrap();
    s.params.set("i", ParamValue::Int(-2)).unwrap();
    let b = bytes(&s);
    let (p, c) = offsets(&b);
    assert_eq!(&b[..4], b"RGS2");
    assert_eq!(&b[p..p + 8], &[0, 0, 0, 128, 254, 255, 255, 255]);
    assert_eq!(&b[c..c + 12], &[82, 71, 83, 49, 120, 86, 52, 18, 1, 0, 0, 0]);
    assert_eq!(bytes(&rt.read_state(&b).unwrap()), b);
    let mut appended = vec![42]; s.write(&mut appended);
    assert_eq!(&appended[1..], b);
}
#[test]
fn malformed_and_incompatible_snapshots_never_mutate_live_state() {
    let rt = runtime();
    let mut live = rt.new_state();
    rt.step(&mut live, &rt.new_params()).unwrap();
    let before = live.clone();
    let good = bytes(&live);
    let (p, c) = offsets(&good);
    let mut reject = |b: &[u8]| {
        assert!(live.restore(rt.compiled(), b).is_err());
        assert_eq!(live, before);
    };
    for end in 0..good.len() { reject(&good[..end]); }
    for (offset, value) in [(0, 0), (4, u32::MAX), (p, f32::NAN.to_bits()),
        (p + 8, 2), (p + 12, 2), (c + 8, 99), (c + 12, 99),
        (c + 20, f32::INFINITY.to_bits()), (c + 20, (-1.0f32).to_bits()),
        (c + 24, 2), (c + 28, 1), (c + 44, 99), (c + 48, 1), (c + 60, 2)] {
        let mut b = good.clone(); set(&mut b, offset, value); reject(&b);
    }
    let mut b = good.clone(); b.extend_from_slice(&[0; 4]); reject(&b);
    for (elapsed, ticks) in [(0, 0), (4, 4), (5, 4)] {
        let mut b = good.clone(); set(&mut b, c + 24, 1);
        set(&mut b, c + 36, elapsed); set(&mut b, c + 40, ticks); reject(&b);
    }
    let mut foreign = rt.compiled().clone(); foreign.graph.name.push('x');
    assert!(GraphState::read(&foreign, &good).is_err());
    foreign = rt.compiled().clone(); foreign.references.blends[0].clips.swap(0, 1);
    assert!(GraphState::read(&foreign, &good).is_err());
    live.restore(rt.compiled(), &bytes(&rt.new_state())).unwrap();
    assert_eq!(live, rt.new_state());
}
