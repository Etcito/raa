use raa_anim::{CLIP_CAPS, Channel, Clip, Interp, JsonValue, Rig, RootMotion, parse};
use std::fs;

fn field<'a>(value: &'a JsonValue, name: &str) -> &'a JsonValue {
    let JsonValue::Object(fields) = value else {
        panic!("expected object")
    };
    &fields.iter().find(|(key, _)| key == name).unwrap().1
}

fn array(value: &JsonValue) -> &[JsonValue] {
    let JsonValue::Array(values) = value else {
        panic!("expected array")
    };
    values
}

fn string(value: &JsonValue) -> &str {
    let JsonValue::String(value) = value else {
        panic!("expected string")
    };
    value
}

#[test]
fn sample_reference() {
    let reference = parse(include_str!("fixtures/samples.json")).unwrap();
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let load =
        |record: &JsonValue| fs::read_to_string(base.join(string(field(record, "file")))).unwrap();
    let rig = Rig::load(&load(field(&reference, "rig"))).unwrap();
    let mut maximum = 0.0;
    let mut count = 0;
    let mut compare = |actual: &[f32], expected: &JsonValue| {
        let expected = array(expected);
        assert_eq!(actual.len(), expected.len());
        for (&actual, expected) in actual.iter().zip(expected) {
            // Widen native values into document-number precision, never round
            // the independent reference down to runtime precision.
            let JsonValue::Number(actual) = JsonValue::Number(actual.into()) else {
                unreachable!()
            };
            let JsonValue::Number(expected) = expected else {
                panic!("expected number")
            };
            let difference = (actual - expected).abs();
            assert!(difference.is_finite());
            if difference > maximum {
                maximum = difference;
            }
            count += 1;
        }
    };
    let clips = array(field(&reference, "clips"));
    assert_eq!(clips.len(), 3);
    for record in clips {
        let clip = Clip::load(&load(record), &rig).unwrap();
        let ticks = array(field(record, "ticks"));
        let poses = array(field(record, "poses"));
        assert!(!ticks.is_empty());
        assert_eq!(ticks.len(), poses.len());
        for (index, (tick, expected)) in ticks.iter().zip(poses).enumerate() {
            assert_eq!(tick, field(expected, "tick"));
            let JsonValue::Number(tick) = tick else {
                panic!("expected tick")
            };
            let tick = *tick as u32;
            let pose = clip.pose_at(&rig, tick).unwrap();
            compare(
                &clip.root_delta_at(&rig, tick).unwrap(),
                field(expected, "rootDelta"),
            );
            let locals = array(field(expected, "local"));
            assert_eq!(pose.local.len(), locals.len());
            for (local, expected) in pose.local.iter().zip(locals) {
                compare(&local.t, field(expected, "t"));
                compare(&local.r, field(expected, "r"));
                compare(&local.s, field(expected, "s"));
            }
            for (name, matrices) in [("model", &pose.model), ("skin", &pose.skin)] {
                let expected = array(field(expected, name));
                assert_eq!(matrices.len(), expected.len());
                for (matrix, expected) in matrices.iter().zip(expected) {
                    compare(matrix, expected);
                }
            }
            let tracks = array(field(record, "tracks"));
            assert_eq!(clip.tracks.len(), tracks.len());
            for (track, expected) in tracks.iter().enumerate() {
                let values = array(field(expected, "values"));
                assert_eq!(ticks.len(), values.len());
                compare(&clip.sample_track(track, tick).unwrap(), &values[index]);
            }
        }
    }
    println!("sample_reference: maximum absolute difference = {maximum:.12e} ({count} components)");
    assert!(
        maximum <= 1e-6,
        "maximum absolute difference {maximum} exceeds 1e-6"
    );
}

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

#[test]
fn sample_curves() {
    let rig = Rig::load(RIG).unwrap();
    let fixture = include_str!("fixtures/clip-bezier.json");
    let clip = Clip::load(fixture, &rig).unwrap();
    assert_eq!(clip.to_canonical_json().as_bytes(), fixture.as_bytes());

    let bezier = |p0: f32, p1: f32, p2: f32, p3: f32, u: f32| {
        let v = 1.0 - u;
        ((v * v * v) * p0 + (3.0 * v * v * u) * p1) + (3.0 * v * u * u) * p2 + (u * u * u) * p3
    };
    assert_eq!(
        clip.sample_track(0, 12).unwrap(),
        [
            bezier(0.0, 1.5, 9.0, 10.0, 0.5),
            bezier(0.0, -2.0, 4.5, 4.0, 0.5),
            bezier(0.0, 0.5, 0.0, -2.0, 0.5),
        ]
    );
    assert_eq!(clip.sample_track(0, 0).unwrap(), [0.0, 0.0, 0.0]);
    assert_eq!(clip.sample_track(0, 24).unwrap(), [10.0, 4.0, -2.0]);

    // Handle dt offsets are stored but never sampled.
    let shifted = fixture
        .replace("\"hr\": [\n            4,", "\"hr\": [\n            31.5,")
        .replace(
            "\"hl\": [\n            -8,",
            "\"hl\": [\n            -0.25,",
        );
    let shifted = Clip::load(&shifted, &rig).unwrap();
    for tick in [0, 1, 7, 12, 23, 24] {
        assert_eq!(clip.sample_track(0, tick), shifted.sample_track(0, tick));
    }

    let flip_fixture = include_str!("fixtures/clip-rotation-flip.json");
    let flip = Clip::load(flip_fixture, &rig).unwrap();
    assert_eq!(flip.to_canonical_json().as_bytes(), flip_fixture.as_bytes());
    assert_eq!(flip.sample_track(0, 16).unwrap(), [0.0, 0.0, 0.0, 1.0]);

    let normalized = |v: [f32; 4]| {
        let squared = ((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]) + v[3] * v[3];
        let inverse = 1.0 / squared.sqrt();
        v.map(|x| x * inverse)
    };
    // Incoming flipped handle: p2 = aligned b.v + (-b.hl.dv).
    let incoming = [
        bezier(0.0, 0.1, 0.05, 0.0, 0.5),
        bezier(0.0, 0.2, -0.1, 0.0, 0.5),
        bezier(0.0, 0.3, 0.2, 0.0, 0.5),
        bezier(1.0, 1.0, 1.0, 1.0, 0.5),
    ];
    assert_eq!(flip.sample_track(0, 8).unwrap(), normalized(incoming));
    // Outgoing flipped handle: p1 = aligned a.v + (-a.hr.dv).
    let outgoing = [
        bezier(0.0, 0.1, 0.2, 0.2, 0.25),
        0.0,
        0.0,
        bezier(1.0, 1.0, 0.9, 0.9, 0.25),
    ];
    assert_eq!(flip.sample_track(0, 20).unwrap(), normalized(outgoing));
    for tick in [0, 8, 16, 20, 32] {
        let v = flip.sample_track(0, tick).unwrap();
        let squared = ((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]) + v[3] * v[3];
        assert!((squared - 1.0).abs() < 1e-6, "tick {tick}: {v:?}");
    }
}

#[test]
fn root_motion_adjacent_ticks() {
    let rig = Rig::load(RIG).unwrap();
    let rest = rig.local_rest_pose().unwrap();
    // Different first/last slopes distinguish endpoint seams from wrapped resets
    // and from an assumed constant velocity. Root rest is [1, 2, 3].
    let source = r#"{"format":"raa.clip","version":2,"name":"motion","rig":"rig-basic","lengthTicks":4,"loop":true,"rootMotion":"xz","tracks":[{"bone":"root","channel":"translation","keys":[{"tick":0,"v":[10,20,30],"interp":"LINEAR"},{"tick":2,"v":[12,24,36],"interp":"LINEAR"},{"tick":4,"v":[18,32,46],"interp":"LINEAR"}]}],"events":[]}"#;
    let mut clip = Clip::load(source, &rig).unwrap();
    for (tick, delta, y) in [
        (0, [0.0, 0.0, 0.0], 20.0),
        (1, [1.0, 0.0, 3.0], 22.0),
        (2, [1.0, 0.0, 3.0], 24.0),
        (3, [3.0, 0.0, 5.0], 28.0),
        (4, [3.0, 0.0, 5.0], 20.0),
        (5, [1.0, 0.0, 3.0], 22.0),
        (8, [3.0, 0.0, 5.0], 20.0),
        (9, [1.0, 0.0, 3.0], 22.0),
        (4_000_000_000, [3.0, 0.0, 5.0], 20.0),
        (u32::MAX, [3.0, 0.0, 5.0], 28.0),
    ] {
        assert_eq!(clip.root_delta_at(&rig, tick).unwrap(), delta, "{tick}");
        let pose = clip.pose_at(&rig, tick).unwrap();
        assert_eq!(pose.local[0].t, [1.0, y, 3.0]);
        assert_eq!(pose.local[0].r, rest[0].r);
        assert_eq!(pose.local[0].s, rest[0].s);
        assert_eq!(pose.local[1], rest[1]);
    }
    assert_eq!(clip.sample_track(0, 4).unwrap(), [10.0, 20.0, 30.0]);
    for tick in [9, 0, 4, 1, 8, 5, 4, 0] {
        let fresh = Clip::load(source, &rig).unwrap();
        assert_eq!(
            clip.root_delta_at(&rig, tick),
            fresh.root_delta_at(&rig, tick)
        );
        assert_eq!(clip.pose_at(&rig, tick), fresh.pose_at(&rig, tick));
    }
    clip.root_motion = RootMotion::Xyz;
    assert_eq!(clip.root_delta_at(&rig, 4).unwrap(), [3.0, 4.0, 5.0]);
    assert_eq!(clip.local_pose_at(&rig, 3).unwrap()[0].t, rest[0].t);
    clip.looping = false;
    assert_eq!(clip.root_delta_at(&rig, 4).unwrap(), [3.0, 4.0, 5.0]);
    for tick in [5, 8, 100, u32::MAX] {
        assert_eq!(clip.root_delta_at(&rig, tick).unwrap(), [0.0; 3]);
    }
    clip.root_motion = RootMotion::None;
    assert_eq!(
        clip.local_pose_at(&rig, 4).unwrap()[0].t,
        [18.0, 32.0, 46.0]
    );
    assert_eq!(clip.root_delta_at(&rig, 4).unwrap(), [0.0; 3]);
    clip.root_motion = RootMotion::Xyz;
    clip.tracks.clear();
    assert_eq!(clip.root_delta_at(&rig, 3).unwrap(), [0.0; 3]);
    assert_eq!(clip.local_pose_at(&rig, 3).unwrap(), rest);
    // No zero-duration JSON clip is currently accepted; keep that validation.
    assert!(
        Clip::load(
            &source.replace("\"lengthTicks\":4", "\"lengthTicks\":0"),
            &rig
        )
        .is_err()
    );
    // Public runtime duration can be changed; the defensive zero case must
    // neither divide by zero nor underflow, with either addressing mode.
    clip = Clip::load(source, &rig).unwrap();
    clip.length_ticks = 0;
    for looping in [false, true] {
        clip.looping = looping;
        for tick in [0, 1, 100, u32::MAX] {
            assert_eq!(clip.root_delta_at(&rig, tick).unwrap(), [0.0; 3]);
        }
    }
}
