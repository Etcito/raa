//! S16/S17 bindings. All validation and runtime arithmetic belong to raa-anim.
use raa_anim::{AnimError, Clip, Pose, Rig};
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct AnimationRig {
    rig: Arc<Rig>,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct AnimationClip {
    rig: Arc<Rig>,
    clip: Clip,
}

pub fn load_rig(json: &str) -> Result<AnimationRig, AnimError> {
    Ok(AnimationRig {
        rig: Arc::new(Rig::load(json)?),
    })
}

pub fn load_clip(rig: &AnimationRig, json: &str) -> Result<AnimationClip, AnimError> {
    Ok(AnimationClip {
        clip: Clip::load(json, &rig.rig)?,
        rig: Arc::clone(&rig.rig),
    })
}

pub fn sample_clip(clip: &AnimationClip, tick: u32) -> Result<(Pose, [f32; 3]), AnimError> {
    Ok((
        clip.clip.pose_at(&clip.rig, tick)?,
        clip.clip.root_delta_at(&clip.rig, tick)?,
    ))
}

pub fn register(cases: &mut Vec<crate::Case>) {
    cases.push(crate::brick::CLIP_SAMPLE);
}

/// Exercise the S17 fixtures at consecutive ticks starting at seed (wrapping u32).
/// DET_HASH is API data, not a cross-platform acceptance comparison.
pub fn clip_sample_selftest(seed: u32, ticks: u32) -> u32 {
    let mut hash = raa_det::DET_HASH;
    visit_clip_samples(seed, ticks, |_, _, pose, _| {
        for matrix in &pose.skin {
            hash.f32s(matrix);
        }
    });
    hash.finish()
}

// Keep native tests and the dispatched worker on the same ordinary runtime path.
fn visit_clip_samples(seed: u32, ticks: u32, mut visit: impl FnMut(&str, u32, &Pose, [f32; 3])) {
    let rig = load_rig(include_str!(
        "../../../raa-anim/tests/fixtures/rig-basic.json"
    ))
    .expect("selftest rig");
    let clips = [
        include_str!("../../../raa-anim/tests/fixtures/clip-linear-constant.json"),
        include_str!("../../../raa-anim/tests/fixtures/clip-bezier.json"),
        include_str!("../../../raa-anim/tests/fixtures/clip-rotation-flip.json"),
    ]
    .map(|json| load_clip(&rig, json).expect("selftest clip"));
    // Even the zero-tick input samples all fixtures once, at its starting tick.
    for offset in 0..ticks.max(1) {
        let tick = seed.wrapping_add(offset);
        for clip in &clips {
            let (pose, delta) = sample_clip(clip, tick).expect("selftest sample");
            assert!(delta.iter().all(|v| v.is_finite()));
            for matrix in pose.model.iter().chain(&pose.skin) {
                assert!(matrix.iter().all(|v| v.is_finite()));
            }
            visit(&clip.clip.name, tick, &pose, delta);
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;

    fn error(e: AnimError) -> JsValue {
        let value = js_sys::Error::new(&e.message);
        js_sys::Reflect::set(&value, &"code".into(), &e.code.into()).expect("fresh error");
        js_sys::Reflect::set(&value, &"path".into(), &e.path.into()).expect("fresh error");
        value.into()
    }

    #[wasm_bindgen(js_name = loadRig)]
    pub fn rig(json: &str) -> Result<AnimationRig, JsValue> {
        load_rig(json).map_err(error)
    }

    #[wasm_bindgen(js_name = loadClip)]
    pub fn clip(rig: &AnimationRig, json: &str) -> Result<AnimationClip, JsValue> {
        load_clip(rig, json).map_err(error)
    }

    #[wasm_bindgen(js_name = sampleClip)]
    pub fn sample(clip: &AnimationClip, tick: JsValue) -> Result<JsValue, JsValue> {
        let tick = tick
            .as_f64()
            .and_then(|n| crate::boundary_u32(n, "tick").ok())
            .ok_or_else(|| {
                let e = js_sys::TypeError::new("tick must be a u32 number");
                js_sys::Reflect::set(&e, &"code".into(), &"api.invalid_input".into())
                    .expect("fresh error");
                JsValue::from(e)
            })?;
        let (pose, delta) = sample_clip(clip, tick).map_err(error)?;
        let local: Vec<f32> = pose
            .local
            .iter()
            .flat_map(|p| p.t.into_iter().chain(p.r).chain(p.s))
            .collect();
        let model: Vec<f32> = pose.model.into_iter().flatten().collect();
        let skin: Vec<f32> = pose.skin.into_iter().flatten().collect();
        let result = js_sys::Object::new();
        for (key, values) in [
            ("local", local.as_slice()),
            ("model", model.as_slice()),
            ("skin", skin.as_slice()),
            ("rootDelta", delta.as_slice()),
        ] {
            js_sys::Reflect::set(&result, &key.into(), &js_sys::Float32Array::from(values))?;
        }
        Ok(result.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_anim::JsonValue;
    const RIG: &str = include_str!("../../../raa-anim/tests/fixtures/rig-basic.json");
    const CLIP: &str = include_str!("../../../raa-anim/tests/fixtures/clip-linear-constant.json");

    #[test]
    fn rig_cap_is_delegated_without_truncation() {
        for count in [256, 257] {
            let bones = (0..count).map(|i| format!(
                r#"{{"name":"b{i}","parent":{},"rest":{{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}}}}"#,
                if i == 0 { "null" } else { "\"b0\"" }
            )).collect::<Vec<_>>().join(",");
            let json = format!(
                r#"{{"format":"raa.rig","version":1,"name":"cap","bones":[{bones}],"roles":{{}},"ikChains":[],"limits":[],"attach":{{}}}}"#
            );
            let loaded = load_rig(&json);
            if count == 256 {
                assert_eq!(loaded.unwrap().rig.bones.len(), 256);
            } else {
                assert_eq!(loaded.err().unwrap().code, "rig.cap_bones");
            }
        }
    }

    #[test]
    fn sampling_retains_rig_and_stateless_root_motion() {
        let rig = load_rig(RIG).unwrap();
        let json = CLIP.replace("\"none\"", "\"xz\"").replace("false", "true");
        let clip = load_clip(&rig, &json).unwrap();
        drop(rig);
        for tick in [0, 14, 48, 49, 96, 14] {
            let (pose, delta) = sample_clip(&clip, tick).unwrap();
            assert_eq!(pose, clip.clip.pose_at(&clip.rig, tick).unwrap());
            assert_eq!(delta, clip.clip.root_delta_at(&clip.rig, tick).unwrap());
            assert_eq!(pose.local[0].t[0], 1.0);
            assert_eq!(pose.local[0].t[2], 3.0);
            assert_eq!(delta[1], 0.0);
            if tick == 14 {
                assert_eq!(pose.local[0].t[1], 1.0);
                assert_eq!(delta[0], 0.5);
            }
        }
        assert_eq!(sample_clip(&clip, 0).unwrap().1, [0.0; 3]);
    }

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

    fn compare(actual: &[f32], expected: &JsonValue) {
        let expected = array(expected);
        assert_eq!(actual.len(), expected.len());
        for (&actual, expected) in actual.iter().zip(expected) {
            let JsonValue::Number(expected) = expected else {
                panic!("expected number")
            };
            assert!(actual.is_finite());
            assert!(
                (f64::from(actual) - expected).abs() <= 1e-6,
                "sample {actual} differs from reference {expected}"
            );
        }
    }

    #[test]
    fn clip_sample_executes_canonical_cases_and_checks_values() {
        // Independently generated f64 expectations, never runtime-derived hashes.
        let reference = raa_anim::parse(include_str!(
            "../../../raa-anim/tests/fixtures/samples.json"
        ))
        .unwrap();
        for (seed, ticks) in raa_det::DET_CASES {
            let mut visited = 0;
            let mut compared = [0; 3];
            visit_clip_samples(seed, ticks, |name, tick, pose, delta| {
                visited += 1;
                let records = array(field(&reference, "clips"));
                let index = records
                    .iter()
                    .position(|r| field(r, "name") == &JsonValue::String(name.into()))
                    .unwrap();
                let poses = array(field(&records[index], "poses"));
                let last = poses.last().unwrap();
                // These committed fixtures are non-looping: high seeds clamp.
                let expected = poses
                    .iter()
                    .find(|p| field(p, "tick") == &JsonValue::Number(tick.into()))
                    .or_else(|| {
                        let JsonValue::Number(end) = field(last, "tick") else {
                            unreachable!()
                        };
                        (f64::from(tick) > *end).then_some(last)
                    });
                if let Some(expected) = expected {
                    compared[index] += 1;
                    compare(&delta, field(expected, "rootDelta"));
                    let locals = array(field(expected, "local"));
                    assert_eq!(pose.local.len(), locals.len());
                    for (local, expected) in pose.local.iter().zip(locals) {
                        compare(&local.t, field(expected, "t"));
                        compare(&local.r, field(expected, "r"));
                        compare(&local.s, field(expected, "s"));
                    }
                    assert_eq!(pose.model.len(), locals.len());
                    assert_eq!(pose.skin.len(), locals.len());
                }
            });
            assert_eq!(visited, ticks.max(1) * 3);
            assert!(compared.iter().all(|&count| count > 0));
            let result = crate::dispatch("clip-sample", seed.into(), ticks.into()).unwrap();
            assert_eq!(
                (result.case, result.seed, result.ticks),
                ("clip-sample", seed, ticks)
            );
            assert_eq!(result.hash.len(), 8);
            assert!(
                result
                    .hash
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            );
        }
    }
}
