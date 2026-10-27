use raa_anim::{CLIP_CAPS, Channel, Clip, Interp, JsonValue, Rig, RootMotion, parse};
use std::fs;

const RIG: &str = include_str!("fixtures/rig-basic.json");
const VALID: &str = r#"{"format":"raa.clip","version":2,"name":"waddle","rig":"rig-basic","lengthTicks":48,"loop":true,"rootMotion":"none","tracks":[{"bone":"child","channel":"rotation","keys":[{"tick":0,"v":[0,0,0,1],"interp":"BEZIER","hl":[-4,0,0,0,0],"hr":[4,0,0.02,0,0]},{"tick":24,"v":[0,0,0,-1],"interp":"LINEAR"}]},{"bone":"root","channel":"translation","keys":[{"tick":0,"v":[0,0,0],"interp":"CONSTANT"}]},{"bone":"root","channel":"scale","keys":[{"tick":0,"v":[1,1,1],"interp":"LINEAR"}]}],"events":[{"tick":12,"name":"step.l"},{"tick":24,"name":"step.r"}]}"#;

fn cap_rig() -> Rig {
    let mut bones = String::new();
    for i in 0..86 {
        if i > 0 {
            bones.push(',');
        }
        let parent = if i == 0 {
            "null".to_owned()
        } else {
            "\"b000\"".to_owned()
        };
        bones.push_str(&format!(
            r#"{{"name":"b{i:03}","parent":{parent},"rest":{{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}}}}"#
        ));
    }
    Rig::load(&format!(
        r#"{{"format":"raa.rig","version":1,"name":"rig-cap","bones":[{bones}]}}"#
    ))
    .unwrap()
}

fn cap_track(bone: &str, channel: &str) -> String {
    let v = match channel {
        "rotation" => "[0,0,0,1]",
        "scale" => "[1,1,1]",
        _ => "[0,0,0]",
    };
    format!(
        r#"{{"bone":"{bone}","channel":"{channel}","keys":[{{"tick":0,"v":{v},"interp":"LINEAR"}}]}}"#
    )
}

fn cap_tracks(count: usize) -> String {
    let channels = ["translation", "rotation", "scale"];
    let mut tracks = Vec::with_capacity(count);
    'outer: for i in 0..86 {
        let bone = format!("b{i:03}");
        for channel in channels {
            if tracks.len() == count {
                break 'outer;
            }
            tracks.push(cap_track(&bone, channel));
        }
    }
    tracks.join(",")
}

fn clip_doc(length_ticks: u32, tracks: &str, events: &str) -> String {
    format!(
        r#"{{"format":"raa.clip","version":2,"name":"cap","rig":"rig-cap","lengthTicks":{length_ticks},"loop":true,"rootMotion":"none","tracks":[{tracks}],"events":[{events}]}}"#
    )
}

#[test]
fn clip_schema() {
    let rig = Rig::load(RIG).unwrap();
    let clip = Clip::load(VALID, &rig).unwrap();
    assert_eq!(clip.name, "waddle");
    assert_eq!(clip.rig, "rig-basic");
    assert_eq!(clip.length_ticks, 48);
    assert!(clip.looping);
    assert_eq!(clip.root_motion, RootMotion::None);
    assert_eq!(clip.tracks.len(), 3);
    assert_eq!(clip.tracks[0].bone, "child");
    assert_eq!(clip.tracks[0].channel, Channel::Rotation);
    assert_eq!(clip.tracks[0].channel.width(), 4);
    assert_eq!(clip.tracks[0].keys[0].interp, Interp::Bezier);
    assert_eq!(clip.tracks[0].keys[0].tick, 0);
    assert!(clip.tracks[0].keys[0].hl.is_some());
    assert_eq!(clip.tracks[2].keys[0].interp, Interp::Linear);
    assert_eq!(clip.events[0].tick, 12);
    assert_eq!(clip.events[1].name, "step.r");

    for (fixture, code) in [
        (
            include_str!("fixtures/invalid-clip-rig-mismatch.json"),
            "clip.rig_mismatch",
        ),
        (
            include_str!("fixtures/invalid-clip-bone.json"),
            "clip.bone_unknown",
        ),
        (
            include_str!("fixtures/invalid-clip-channel.json"),
            "clip.channel",
        ),
        (
            include_str!("fixtures/invalid-clip-duplicate-track.json"),
            "clip.duplicate_track",
        ),
        (
            include_str!("fixtures/invalid-clip-key-order.json"),
            "clip.key_order",
        ),
    ] {
        let error = Clip::load(fixture, &rig).unwrap_err();
        assert_eq!(error.code, code, "{error}");
    }

    let invalid = fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"))
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("invalid-"))
        .count();
    assert_eq!(invalid, 12);
}

#[test]
fn clip_validation() {
    let rig = Rig::load(RIG).unwrap();
    let doc = |fields: &str| {
        format!(
            r#"{{"format":"raa.clip","version":2,"name":"x","rig":"rig-basic","lengthTicks":48,"loop":true,"rootMotion":"none",{fields}}}"#
        )
    };
    let track = |key: &str| {
        format!(
            r#""tracks":[{{"bone":"root","channel":"translation","keys":[{key}]}}],"events":[]"#
        )
    };
    let code = |text: &str| Clip::load(text, &rig).unwrap_err().code;

    let empty = doc(r#""tracks":[],"events":[]"#);
    assert!(Clip::load(&empty, &rig).is_ok());
    assert_eq!(
        code(&empty.replace("\"raa.clip\"", "\"raa.rig\"")),
        "clip.format"
    );
    assert_eq!(
        code(&empty.replace("\"version\":2", "\"version\":1")),
        "clip.version"
    );
    assert_eq!(
        code(&doc(r#""extra":0,"tracks":[],"events":[]"#)),
        "clip.schema"
    );
    assert_eq!(
        code(&empty.replace("\"lengthTicks\":48", "\"lengthTicks\":0")),
        "clip.schema"
    );
    assert_eq!(
        code(&doc(r#""tracks":[],"events":[{"tick":0}]"#)),
        "clip.schema"
    );
    assert_eq!(
        code(&doc(&track(r#"{"tick":0,"v":[0,0],"interp":"LINEAR"}"#))),
        "clip.channel"
    );
    assert_eq!(
        code(&doc(&track(
            r#"{"tick":0,"v":[0,0,0],"interp":"QUADRATIC"}"#
        ))),
        "clip.interp"
    );
    assert_eq!(
        code(&doc(&track(
            r#"{"tick":0,"v":[0,0,0],"interp":"LINEAR","hl":[0,0,0,0]}"#
        ))),
        "clip.handles"
    );
    assert_eq!(
        code(&doc(&track(
            r#"{"tick":0,"v":[0,0,0],"interp":"BEZIER","hl":[0,0,0]}"#
        ))),
        "clip.handles"
    );
    assert_eq!(
        code(&doc(&track(
            r#"{"tick":0.5,"v":[0,0,0],"interp":"LINEAR"}"#
        ))),
        "clip.schema"
    );
    assert!(
        Clip::load(
            &doc(&track(r#"{"tick":0,"v":[0,0,0],"interp":"BEZIER"}"#)),
            &rig
        )
        .is_ok()
    );
    let long = empty.replace("\"lengthTicks\":48", "\"lengthTicks\":4000000000");
    assert_eq!(Clip::load(&long, &rig).unwrap().length_ticks, 4_000_000_000);
}

#[test]
fn clip_caps() {
    assert_eq!(CLIP_CAPS.max_tracks, 256);
    assert_eq!(CLIP_CAPS.max_keys_per_track, 4096);
    assert_eq!(CLIP_CAPS.max_events, 1024);
    let rig = cap_rig();

    let at = clip_doc(48, &cap_tracks(CLIP_CAPS.max_tracks), "");
    let clip = Clip::load(&at, &rig).unwrap();
    assert_eq!(clip.tracks.len(), CLIP_CAPS.max_tracks);
    let over = clip_doc(48, &cap_tracks(CLIP_CAPS.max_tracks + 1), "");
    let error = Clip::load(&over, &rig).unwrap_err();
    assert_eq!(error.code, "clip.cap");
    assert_eq!(error.path, "/tracks");

    let fixture = include_str!("fixtures/invalid-clip-cap.json");
    let error = Clip::load(fixture, &rig).unwrap_err();
    assert_eq!(error.code, "clip.cap");
    assert_eq!(error.path, "/tracks");
    let error = Clip::load(fixture, &Rig::load(RIG).unwrap()).unwrap_err();
    assert_eq!(error.code, "clip.cap", "the cap refuses before any rig use");
    let JsonValue::Object(fields) = parse(fixture).unwrap() else {
        panic!("fixture is not an object");
    };
    let JsonValue::Array(tracks) = &fields.iter().find(|(k, _)| k == "tracks").unwrap().1 else {
        panic!("fixture tracks is not an array");
    };
    assert_eq!(tracks.len(), 257);
    let mut targets = Vec::with_capacity(tracks.len());
    for track in tracks {
        let JsonValue::Object(fields) = track else {
            panic!("track is not an object");
        };
        let field = |name: &str| match &fields.iter().find(|(k, _)| k == name).unwrap().1 {
            JsonValue::String(value) => value.clone(),
            _ => panic!("{name} is not a string"),
        };
        targets.push((field("bone"), field("channel")));
    }
    targets.sort();
    targets.dedup();
    assert_eq!(targets.len(), 257, "track targets must be distinct");
    assert_eq!(
        fixture,
        clip_doc(48, &cap_tracks(257), "").replace(r#""name":"cap""#, r#""name":"bad-cap""#),
        "the fixture must differ from a valid clip only by its 257th track"
    );

    let keys = |count: usize| {
        (0..count)
            .map(|i| format!(r#"{{"tick":{i},"v":[0,0,0],"interp":"LINEAR"}}"#))
            .collect::<Vec<_>>()
            .join(",")
    };
    let key_doc = |count: usize| {
        let tracks = format!(
            r#"{{"bone":"b000","channel":"translation","keys":[{}]}}"#,
            keys(count)
        );
        clip_doc(4096, &tracks, "")
    };
    let clip = Clip::load(&key_doc(CLIP_CAPS.max_keys_per_track), &rig).unwrap();
    assert_eq!(clip.tracks[0].keys.len(), CLIP_CAPS.max_keys_per_track);
    let error = Clip::load(&key_doc(CLIP_CAPS.max_keys_per_track + 1), &rig).unwrap_err();
    assert_eq!(error.code, "clip.cap");
    assert_eq!(error.path, "/tracks/0/keys");

    let events = |count: usize| {
        (0..count)
            .map(|i| format!(r#"{{"tick":{i},"name":"e{i}"}}"#))
            .collect::<Vec<_>>()
            .join(",")
    };
    let event_doc = |count: usize| clip_doc(4096, &cap_tracks(1), &events(count));
    let clip = Clip::load(&event_doc(CLIP_CAPS.max_events), &rig).unwrap();
    assert_eq!(clip.events.len(), CLIP_CAPS.max_events);
    let error = Clip::load(&event_doc(CLIP_CAPS.max_events + 1), &rig).unwrap_err();
    assert_eq!(error.code, "clip.cap");
    assert_eq!(error.path, "/events");
}

#[test]
fn clip_canonical() {
    let rig = Rig::load(RIG).unwrap();
    let fixture = include_str!("fixtures/clip-linear-constant.json");
    let clip = Clip::load(fixture, &rig).unwrap();
    assert_eq!(clip.to_canonical_json().as_bytes(), fixture.as_bytes());
    assert_eq!(Clip::load(&clip.to_canonical_json(), &rig).unwrap(), clip);
    // Intervals cover before-first, exact keys, linear midpoint, constant
    // midpoint and after-last; child deliberately has no animated channel.
    assert_eq!(
        clip.tracks[0]
            .keys
            .iter()
            .map(|k| k.tick)
            .collect::<Vec<_>>(),
        [4, 24, 44]
    );
    assert!(clip.tracks.iter().all(|track| track.bone != "child"));
    assert!(!clip.looping);
    assert_eq!(clip.events.last().unwrap().tick, clip.length_ticks);
    let text = Clip::load(VALID, &rig).unwrap().to_canonical_json();
    let mut at = 0;
    for field in [
        "\"format\"",
        "\"version\"",
        "\"name\"",
        "\"rig\"",
        "\"lengthTicks\"",
        "\"loop\"",
        "\"rootMotion\"",
        "\"tracks\"",
        "\"events\"",
    ] {
        at += text[at..].find(field).unwrap_or_else(|| panic!("{field}"));
        at += field.len();
    }
    assert!(text.contains("\"tick\": 24"));
    assert!(text.contains("\"hl\": ["));
    assert_eq!(Clip::load(&text, &rig).unwrap().to_canonical_json(), text);

    let spelled = VALID.replace("[0,0,0,-1]", "[0,1e0,-0,-1.0005]");
    let canonical = Clip::load(&spelled, &rig).unwrap().to_canonical_json();
    assert!(canonical.contains("-1.0005"), "{canonical}");
    assert!(!canonical.contains("1e0"), "{canonical}");
    assert_eq!(
        Clip::load(&canonical, &rig).unwrap().to_canonical_json(),
        canonical
    );

    let key_doc = |key: &str| {
        format!(
            r#"{{"events":[],"tracks":[{{"keys":[{key}],"channel":"translation","bone":"root"}}],"rootMotion":"none","loop":false,"lengthTicks":48,"rig":"rig-basic","name":"x","version":2,"format":"raa.clip"}}"#
        )
    };
    for (key, code, suffix) in [
        (
            r#"{"tick":0,"v":[0,0,0],"interp":"linear"}"#,
            "clip.interp",
            "/interp",
        ),
        (
            r#"{"tick":0,"v":[0,0,0],"interp":"CONSTANT","hr":[1,0,0,0]}"#,
            "clip.handles",
            "",
        ),
        (
            r#"{"tick":0,"v":[0,0,0],"interp":"BEZIER","hr":[1,0,0]}"#,
            "clip.handles",
            "/hr",
        ),
        (
            r#"{"tick":0,"v":[0,0,0],"interp":"BEZIER","hl":null}"#,
            "clip.handles",
            "/hl",
        ),
        (
            r#"{"tick":0,"v":[0,0,0],"interp":"BEZIER","hl":[false,0,0,0]}"#,
            "clip.handles",
            "/hl/0",
        ),
        (
            r#"{"tick":0,"v":[0,0,0],"interp":"LINEAR","extra":0}"#,
            "clip.schema",
            "/extra",
        ),
        (
            r#"{"tick":0,"v":[1e999,0,0],"interp":"LINEAR"}"#,
            "clip.not_finite",
            "/v/0",
        ),
        (
            r#"{"tick":0,"v":[0,0,0],"interp":"BEZIER","hl":[-1e999,0,0,0]}"#,
            "clip.not_finite",
            "/hl/0",
        ),
        (
            r#"{"tick":0,"v":[0,0,0],"interp":"BEZIER","hr":[1,0,1e999,0]}"#,
            "clip.not_finite",
            "/hr/2",
        ),
    ] {
        let error = Clip::load(&key_doc(key), &rig).unwrap_err();
        assert_eq!(error.code, code, "{error}");
        assert_eq!(error.path, format!("/tracks/0/keys/0{suffix}"));
    }
    let source = key_doc(
        r#"{"hr":[2.75,1e21,1e-7,-0],"interp":"BEZIER","v":[0.30000000000000004,1e-6,1e20],"tick":0,"hl":[-4.25,0,0,0]}"#,
    );
    let mut clip = Clip::load(&source, &rig).unwrap();
    let canonical = clip.to_canonical_json();
    assert_eq!(Clip::load(&canonical, &rig).unwrap(), clip);
    assert_eq!(
        Clip::load(&canonical, &rig).unwrap().to_canonical_json(),
        canonical
    );
    for number in [
        "-4.25",
        "2.75",
        "1e+21",
        "1e-7",
        "0.000001",
        "100000000000000000000",
        "0.30000000000000004",
    ] {
        assert!(canonical.contains(number), "{number}");
    }
    let mut at = 0;
    for field in ["tick", "v", "interp", "hl", "hr"] {
        at += canonical[at..].find(&format!("\"{field}\":")).unwrap() + field.len() + 3;
    }
    for interp in [Interp::Linear, Interp::Constant] {
        clip.tracks[0].keys[0].interp = interp;
        let text = clip.to_canonical_json();
        assert!(!text.contains("\"hl\"") && !text.contains("\"hr\""));
        assert!(Clip::load(&text, &rig).is_ok());
    }
}

#[test]
fn sample_basic() {
    use raa_anim::address_tick;

    let rig = Rig::load(RIG).unwrap();
    let fixture = include_str!("fixtures/clip-linear-constant.json");
    let clip = Clip::load(fixture, &rig).unwrap();
    let linear = |a: [f32; 3], b: [f32; 3], from: u32, to: u32, tick: u32| {
        let u = (tick - from) as f32 / (to - from) as f32;
        [
            a[0] + (b[0] - a[0]) * u,
            a[1] + (b[1] - a[1]) * u,
            a[2] + (b[2] - a[2]) * u,
        ]
    };
    assert_eq!(address_tick(48, 48, true), 0);
    assert_eq!(address_tick(95, 48, true), 47);
    assert_eq!(address_tick(48, 48, false), 48);

    // Keys sit at 4 (LINEAR), 24 (CONSTANT) and 44 (LINEAR).
    assert_eq!(clip.sample_track(0, 0).unwrap(), [0.0, 0.0, 0.0]);
    assert_eq!(clip.sample_track(0, 4).unwrap(), [0.0, 0.0, 0.0]);
    let expected = linear([0.0, 0.0, 0.0], [10.0, 2.0, -4.0], 4, 24, 12);
    assert_eq!(clip.sample_track(0, 12).unwrap(), expected);
    assert_eq!(clip.sample_track(0, 24).unwrap(), [10.0, 2.0, -4.0]);
    assert_eq!(clip.sample_track(0, 34).unwrap(), [10.0, 2.0, -4.0]);
    assert_eq!(clip.sample_track(0, 43).unwrap(), [10.0, 2.0, -4.0]);
    assert_eq!(clip.sample_track(0, 44).unwrap(), [20.0, 0.0, 4.0]);
    assert_eq!(clip.sample_track(0, 47).unwrap(), [20.0, 0.0, 4.0]);
    assert_eq!(clip.sample_track(0, 480).unwrap(), [20.0, 0.0, 4.0]);

    // Tracked channels are replaced; the rest value survives everywhere else.
    let rest = rig.local_rest_pose().unwrap();
    let local = clip.local_pose_at(&rig, 12).unwrap();
    assert_eq!(local[0].t, expected);
    assert_eq!(local[0].r, rest[0].r);
    assert_eq!(local[0].s, rest[0].s);
    assert_eq!(local[1], rest[1], "child has no track");
    let pose = clip.pose_at(&rig, 12).unwrap();
    assert_eq!(&pose.model[0][12..15], &expected);
    assert_eq!(pose.model.len(), rig.bones.len());
    assert!(pose.skin.iter().flatten().all(|v| v.is_finite()));

    let still = r#"{"format":"raa.clip","version":2,"name":"still","rig":"rig-basic","lengthTicks":48,"loop":true,"rootMotion":"none","tracks":[],"events":[]}"#;
    let still = Clip::load(still, &rig).unwrap();
    assert_eq!(still.pose_at(&rig, 5).unwrap(), rig.rest_pose().unwrap());

    // Looping wraps at lengthTicks before key lookup and clamping.
    let looping = Clip::load(&fixture.replace("\"loop\": false", "\"loop\": true"), &rig).unwrap();
    assert_eq!(looping.sample_track(0, 48).unwrap(), [0.0, 0.0, 0.0]);
    assert_eq!(looping.sample_track(0, 49).unwrap(), [0.0, 0.0, 0.0]);
    assert_eq!(looping.sample_track(0, 52).unwrap(), [0.0, 0.0, 0.0]);
    assert_eq!(
        looping.sample_track(0, 58).unwrap(),
        linear([0.0, 0.0, 0.0], [10.0, 2.0, -4.0], 4, 24, 10)
    );
    assert_eq!(looping.sample_track(0, 72).unwrap(), [10.0, 2.0, -4.0]);
    assert_eq!(looping.sample_track(0, 92).unwrap(), [20.0, 0.0, 4.0]);
    assert_eq!(looping.sample_track(0, 95).unwrap(), [20.0, 0.0, 4.0]);
    assert_eq!(looping.sample_track(0, 96).unwrap(), [0.0, 0.0, 0.0]);

    // Rotation keys sign-align to the previous key, then normalize with sqrt.
    let rotate = r#"{"format":"raa.clip","version":2,"name":"rotate","rig":"rig-basic","lengthTicks":10,"loop":false,"rootMotion":"none","tracks":[{"bone":"child","channel":"rotation","keys":[{"tick":0,"v":[0,0,0,2],"interp":"LINEAR"},{"tick":10,"v":[0,0,0,-2],"interp":"LINEAR"}]}],"events":[]}"#;
    let rotate = Clip::load(rotate, &rig).unwrap();
    assert_eq!(rotate.sample_track(0, 5).unwrap(), [0.0, 0.0, 0.0, 1.0]);
}
