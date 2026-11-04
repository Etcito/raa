//! Jiggle solver: format, clamp, damping decay, frame-rate independence, world bookkeeping.
use raa_anim::jiggle::{JIGGLE_SUBSTEP_HZ, JiggleChain, JiggleWorld};
use raa_anim::{Mat4, Rig};

const IDENTITY: Mat4 = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// A hat root on the mount with one horn (jiggle) and its tip, the horn pointing +X.
fn hat(jiggle: &str) -> Rig {
    Rig::load(&format!(
        r#"{{"format":"raa.rig","version":1,"name":"hat","parentBone":"head","bones":[
        {{"name":"hat","parent":null,"rest":{{"t":[0,1,0],"r":[0,0,0,1],"s":[1,1,1]}}}},
        {{"name":"horn","parent":"hat","rest":{{"t":[0.1,0,0],"r":[0,0,0,1],"s":[1,1,1]}},"jiggle":{jiggle}}},
        {{"name":"tip","parent":"horn","rest":{{"t":[0.5,0,0],"r":[0,0,0,1],"s":[1,1,1]}}}}]}}"#
    ))
    .unwrap()
}

fn translation(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = IDENTITY;
    m[12] = x;
    m[13] = y;
    m[14] = z;
    m
}

/// The horn's swing away from its animated direction, degrees, read from its palette matrix (the X column).
fn swing(palette: &[f32]) -> f32 {
    let m = &palette[16..32];
    let x = [m[0], m[1], m[2]];
    let l = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    (x[0] / l).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Signed vertical swing (degrees, + = tip up) of the horn.
fn pitch(palette: &[f32]) -> f32 {
    let m = &palette[16..32];
    m[1].atan2(m[0]).to_degrees()
}

#[test]
fn format_reads_jiggle_and_parent_bone_with_defaults() {
    let rig = hat("{}");
    assert_eq!(rig.parent_bone.as_deref(), Some("head"));
    let j = rig.bones[1].jiggle.unwrap();
    assert_eq!(
        (j.max_angle, j.stiffness, j.damping, j.gravity),
        (30.0, 0.5, 0.3, 0.0)
    );
    assert!(rig.bones[0].jiggle.is_none());
    let rig = hat(r#"{"maxAngle":45,"stiffness":0.2,"damping":0.9,"gravity":0.5}"#);
    let j = rig.bones[1].jiggle.unwrap();
    assert_eq!(
        (j.max_angle, j.stiffness, j.damping, j.gravity),
        (45.0, 0.2, 0.9, 0.5)
    );
    // Canonical JSON keeps both fields and round-trips.
    let text = rig.to_canonical_json();
    assert!(text.contains("\"parentBone\": \"head\""), "{text}");
    assert_eq!(Rig::load(&text).unwrap(), rig);
}

#[test]
fn format_refuses_bad_jiggle() {
    for (jiggle, path) in [
        (r#"{"maxAngle":181}"#, "/bones/1/jiggle/maxAngle"),
        (r#"{"stiffness":-0.1}"#, "/bones/1/jiggle/stiffness"),
        (r#"{"damping":2}"#, "/bones/1/jiggle/damping"),
        (r#"{"damping":"x"}"#, "/bones/1/jiggle/damping"),
        (r#"{"spring":1}"#, "/bones/1/jiggle/spring"),
    ] {
        let text = format!(
            r#"{{"format":"raa.rig","version":1,"name":"h","bones":[{{"name":"a","parent":null,"rest":{{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}}}},{{"name":"b","parent":"a","rest":{{"t":[0,1,0],"r":[0,0,0,1],"s":[1,1,1]}},"jiggle":{jiggle}}}]}}"#
        );
        let error = Rig::load(&text).unwrap_err();
        assert_eq!(error.path, path, "{jiggle}: {error}");
    }
    let error = Rig::load(r#"{"format":"raa.rig","version":1,"name":"h","parentBone":"no space","bones":[{"name":"a","parent":null,"rest":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}}]}"#).unwrap_err();
    assert_eq!(
        (error.code, error.path.as_str()),
        ("rig.bone_name", "/parentBone")
    );
}

#[test]
fn unjiggled_palette_is_the_mount() {
    let mut chain = JiggleChain::new(&hat("{}")).unwrap();
    let mount = translation(0.0, 0.2, 0.0);
    let palette = chain
        .step(
            4,
            1.0 / JIGGLE_SUBSTEP_HZ,
            &translation(5.0, 0.0, 3.0),
            &mount,
        )
        .unwrap();
    for bone in 0..3 {
        for (a, b) in palette[bone * 16..bone * 16 + 16].iter().zip(mount) {
            assert!((a - b).abs() < 1e-5, "bone {bone}: {a} vs {b}");
        }
    }
}

/// Run a chain under a world placement given per frame, at a fixed frame rate, sampling the horn swing.
fn run(jiggle: &str, fps: f32, seconds: f32, place: impl Fn(f32) -> Mat4) -> Vec<(f32, f32)> {
    let mut world = JiggleWorld::new();
    let h = world.add(&hat(jiggle)).unwrap();
    let frames = (seconds * fps).round() as usize;
    let mut out = Vec::new();
    for f in 0..=frames {
        let t = f as f32 / fps;
        let mut input = place(t).to_vec();
        input.extend_from_slice(&IDENTITY);
        let palette = world
            .step(if f == 0 { 0.0 } else { 1.0 / fps }, &[h], &input)
            .unwrap();
        out.push((t, pitch(&palette)));
    }
    out
}

#[test]
fn clamp_holds_max_angle_under_violent_motion() {
    // The hat is thrown up and down 1 m at 6 Hz: far more than a 20 degree swing would follow.
    let samples = run(
        r#"{"maxAngle":20,"stiffness":0.1,"damping":0.05}"#,
        60.0,
        2.0,
        |t| translation(0.0, (t * 6.0 * std::f32::consts::TAU).sin(), 0.0),
    );
    let peak = samples.iter().map(|(_, a)| a.abs()).fold(0.0, f32::max);
    assert!(peak <= 20.0 + 1e-3, "peak swing {peak}");
    assert!(
        peak > 15.0,
        "the motion should reach the clamp, peak {peak}"
    );
}

#[test]
fn damping_decays_the_swing() {
    // A sudden stop: the hat moves up at 2 m/s for 0.2 s, then halts; the horn swings and settles.
    let place = |t: f32| translation(0.0, 2.0 * t.min(0.2), 0.0);
    let samples = run(
        r#"{"maxAngle":60,"stiffness":0.3,"damping":0.15}"#,
        60.0,
        4.0,
        place,
    );
    let after = |from: f32, to: f32| {
        samples
            .iter()
            .filter(|(t, _)| *t >= from && *t < to)
            .map(|(_, a)| a.abs())
            .fold(0.0, f32::max)
    };
    let first = after(0.2, 1.0);
    let later = after(1.0, 2.0);
    let end = after(3.5, 4.01);
    assert!(first > 2.0, "the stop should swing the horn, peak {first}");
    assert!(
        later < first * 0.6,
        "swing should decay: {first} then {later}"
    );
    assert!(end < 0.2, "swing should settle, still {end} degrees");
    // Critical damping does not swing back past the animated pose by more than a sliver.
    let critical = run(
        r#"{"maxAngle":60,"stiffness":0.3,"damping":1}"#,
        60.0,
        3.0,
        place,
    );
    let lowest = critical
        .iter()
        .filter(|(t, _)| *t > 0.2)
        .map(|(_, a)| *a)
        .fold(f32::MAX, f32::min);
    let highest = critical
        .iter()
        .filter(|(t, _)| *t > 0.2)
        .map(|(_, a)| *a)
        .fold(f32::MIN, f32::max);
    let overshoot = if highest.abs() > lowest.abs() {
        lowest
    } else {
        highest
    };
    assert!(
        overshoot.abs() < 0.5,
        "critical damping overshoot {overshoot}"
    );
}

#[test]
fn swing_is_frame_rate_independent() {
    let place = |t: f32| translation(0.3 * (t * 3.0).sin(), 0.4 * (t * 5.0).sin(), 0.0);
    let jiggle = r#"{"maxAngle":45,"stiffness":0.4,"damping":0.2}"#;
    let reference = run(jiggle, 240.0, 3.0, place);
    for fps in [30.0, 60.0, 144.0] {
        let samples = run(jiggle, fps, 3.0, place);
        let mut worst: f32 = 0.0;
        let mut compared = 0;
        // Common instants: every 1/6 s is a whole frame at 30, 60, 144 and 240 fps.
        for (t, a) in &samples {
            let k = (t * 6.0).round();
            if (t * 6.0 - k).abs() > 1e-4 || *t < 0.2 {
                continue;
            }
            let (_, b) = reference
                .iter()
                .find(|(u, _)| (u - t).abs() < 1e-4)
                .unwrap();
            worst = worst.max((a - b).abs());
            compared += 1;
        }
        let range = reference.iter().map(|(_, a)| a.abs()).fold(0.0, f32::max);
        assert!(compared >= 15, "{fps} fps compared {compared} instants");
        assert!(
            range > 3.0,
            "the motion should swing the horn, range {range}"
        );
        assert!(
            worst < 0.15 * range,
            "{fps} fps differs by {worst} degrees (swing range {range})"
        );
    }
}

#[test]
fn turning_head_swings_the_horn_and_returns() {
    // A yaw turn of the mount over 0.25 s: the horn lags, then comes back to the animated pose.
    let mut world = JiggleWorld::new();
    let h = world
        .add(&hat(r#"{"maxAngle":40,"stiffness":0.4,"damping":0.3}"#))
        .unwrap();
    let mut peak: f32 = 0.0;
    let mut last = 0.0;
    for f in 0..=180 {
        let t = f as f32 / 60.0;
        let yaw = (t / 0.25).min(1.0) * std::f32::consts::FRAC_PI_2;
        let (s, c) = yaw.sin_cos();
        let mount: Mat4 = [
            c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut input = IDENTITY.to_vec();
        input.extend_from_slice(&mount);
        let palette = world
            .step(if f == 0 { 0.0 } else { 1.0 / 60.0 }, &[h], &input)
            .unwrap();
        // Swing relative to the rigid mount: compare the horn's X column with the mount's.
        let m = &palette[16..32];
        let d = (m[0] * c - m[2] * s).clamp(-1.0, 1.0).acos().to_degrees();
        peak = peak.max(d);
        last = d;
    }
    assert!(peak > 5.0, "a head turn should swing the horn, peak {peak}");
    assert!(
        last < 0.3,
        "the horn should settle after the turn, still {last}"
    );
}

#[test]
fn skipped_chain_restarts_at_its_animated_pose() {
    let mut world = JiggleWorld::new();
    let rig = hat(r#"{"maxAngle":60,"stiffness":0.2,"damping":0.1}"#);
    let a = world.add(&rig).unwrap();
    let b = world.add(&rig).unwrap();
    let frame = |y: f32| {
        let mut v = translation(0.0, y, 0.0).to_vec();
        v.extend_from_slice(&IDENTITY);
        v
    };
    let both = |y: f32| [frame(y), frame(y)].concat();
    world.step(0.0, &[a, b], &both(0.0)).unwrap();
    // Both swing after a jump; then `b` is culled for a while and comes back somewhere else.
    let palette = world.step(1.0 / 60.0, &[a, b], &both(0.05)).unwrap();
    assert!(swing(&palette[..48]) > 0.5);
    for i in 0..10 {
        world
            .step(1.0 / 60.0, &[a], &frame(0.05 + 0.05 * i as f32))
            .unwrap();
    }
    let palette = world.step(1.0 / 60.0, &[b], &frame(3.0)).unwrap();
    assert!(
        swing(&palette) < 1e-3,
        "a restarted chain starts at rest, swing {}",
        swing(&palette)
    );
    // Handles: removal frees the slot, a stale handle is refused, a duplicate in one call too.
    world.remove(a);
    assert_eq!(
        world.step(0.01, &[a], &frame(0.0)).unwrap_err().code,
        "jiggle.handle"
    );
    assert_eq!(
        world.step(0.01, &[b, b], &both(0.0)).unwrap_err().code,
        "jiggle.input"
    );
    assert_eq!(world.add(&rig).unwrap(), a);
    assert_eq!(world.len(), 2);
}

#[test]
fn animated_pose_input_drives_the_target() {
    // The armature's own animation (the hat root nodding) swings the horn like a mount motion does.
    let rig = hat(r#"{"maxAngle":40,"stiffness":0.3,"damping":0.2}"#);
    let mut chain = JiggleChain::new(&rig).unwrap();
    let rest = rig.rest_pose().unwrap();
    let mut peak: f32 = 0.0;
    for f in 0..60 {
        let angle = if f < 5 { 0.0 } else { 0.5_f32 };
        let (s, c) = angle.sin_cos();
        let nod: Mat4 = [
            c, s, 0.0, 0.0, -s, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let model: Vec<f32> = rest
            .model
            .iter()
            .flat_map(|m| raa_anim::jiggle::mat_mul(&nod, m))
            .collect();
        chain.set_pose(&model).unwrap();
        let palette = chain
            .step(4, 1.0 / JIGGLE_SUBSTEP_HZ, &IDENTITY, &IDENTITY)
            .unwrap()
            .to_vec();
        let horn = &palette[16..32];
        let animated = raa_anim::jiggle::mat_mul(&nod, &IDENTITY);
        let d = (horn[0] * animated[0] + horn[1] * animated[1])
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        peak = peak.max(d);
    }
    assert!(peak > 3.0, "a nod should swing the horn, peak {peak}");
}

#[test]
fn skin_input_is_the_same_pose_as_model_input() {
    // An editor preview only holds the skin palette (model * inverse bind): set_skin must pose the chain identically.
    let rig = hat(r#"{"maxAngle":40,"stiffness":0.3,"damping":0.2}"#);
    let rest = rig.rest_pose().unwrap();
    let (s, c) = 0.4_f32.sin_cos();
    let nod: Mat4 = [
        c, s, 0.0, 0.0, -s, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0, 0.0, 1.0,
    ];
    let model: Vec<f32> = rest
        .model
        .iter()
        .flat_map(|m| raa_anim::jiggle::mat_mul(&nod, m))
        .collect();
    let skin: Vec<f32> = rest
        .model
        .iter()
        .zip(rig.inverse_bind_matrices().unwrap())
        .flat_map(|(m, ib)| raa_anim::jiggle::mat_mul(&raa_anim::jiggle::mat_mul(&nod, m), &ib))
        .collect();
    let mut a = JiggleChain::new(&rig).unwrap();
    let mut b = JiggleChain::new(&rig).unwrap();
    a.set_pose(&model).unwrap();
    b.set_skin(&skin).unwrap();
    for _ in 0..30 {
        let pa = a
            .step(4, 1.0 / JIGGLE_SUBSTEP_HZ, &IDENTITY, &IDENTITY)
            .unwrap()
            .to_vec();
        let pb = b
            .step(4, 1.0 / JIGGLE_SUBSTEP_HZ, &IDENTITY, &IDENTITY)
            .unwrap()
            .to_vec();
        for (x, y) in pa.iter().zip(&pb) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }
    assert!(b.set_skin(&[0.0; 5]).is_err());
    b.set_skin(&[]).unwrap();
}
