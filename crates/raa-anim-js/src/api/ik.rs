//! S20 visual IK over S17 poses, exposed as the JavaScript `ik` namespace.
use raa_anim::{AnimError, LocalPose, Pose, Rig};
use raa_ik::{IkPose, ccd, fabrik, two_bone};
#[cfg(target_arch = "wasm32")]
use {crate::clips::AnimationRig, wasm_bindgen::prelude::*};
pub const DEFAULT_TOLERANCE: f32 = 0.001;
pub const DEFAULT_WEIGHT: f32 = 1.0;
/// A corrected pose plus the outcome of iterative solvers.
#[derive(Debug, Clone, PartialEq)]
pub struct IkOutput {
    pub local: Vec<f32>,
    pub model: Vec<f32>,
    pub skin: Vec<f32>,
    pub converged: Option<bool>,
    pub passes: Option<usize>,
    pub distance: Option<f32>,
}
fn invalid(message: impl Into<String>) -> AnimError {
    AnimError::new("api.invalid_input", "/pose", message)
}
/// Convert the flat S17 local layout (t xyz, r xyzw, s xyz) into bone locals.
pub fn local_pose(rig: &Rig, values: &[f32]) -> Result<Vec<LocalPose>, AnimError> {
    let expected = rig.bones.len() * 10;
    if values.len() != expected {
        return Err(invalid(format!("expected {expected} local values, got {}", values.len())));
    }
    Ok(values
        .chunks_exact(10)
        .map(|v| LocalPose {
            t: [v[0], v[1], v[2]],
            r: [v[3], v[4], v[5], v[6]],
            s: [v[7], v[8], v[9]],
        })
        .collect())
}
fn flatten(pose: Pose) -> IkOutput {
    let local = pose.local.iter().flat_map(|p| p.t.into_iter().chain(p.r).chain(p.s)).collect();
    IkOutput {
        local,
        model: pose.model.into_iter().flatten().collect(),
        skin: pose.skin.into_iter().flatten().collect(),
        converged: None,
        passes: None,
        distance: None,
    }
}
/// `two_bone` on the S16 chain named `name`.
pub fn solve_two_bone(rig: &Rig, name: &str, values: &[f32], target: [f32; 3],
    pole: Option<[f32; 3]>, weight: f32) -> Result<IkOutput, AnimError> {
    let chain = raa_ik::chain(rig, name)?;
    let mut ik = IkPose::new(rig, &local_pose(rig, values)?)?;
    two_bone(&mut ik, chain, target, pole, weight)?;
    Ok(flatten(ik.into_pose()))
}
fn solve_iterative(rig: &Rig, name: &str, values: &[f32], target: [f32; 3],
    iterations: usize, tol: f32, fab: bool) -> Result<IkOutput, AnimError> {
    let chain = raa_ik::chain(rig, name)?;
    let mut ik = IkPose::new(rig, &local_pose(rig, values)?)?;
    let (converged, passes, distance) = if fab {
        let out = fabrik(&mut ik, chain, target, iterations, tol)?;
        (out.converged, out.passes, out.distance)
    } else {
        let out = ccd(&mut ik, chain, target, iterations, tol)?;
        (out.converged, out.passes, out.distance)
    };
    let mut output = flatten(ik.into_pose());
    output.converged = Some(converged);
    output.passes = Some(passes);
    output.distance = Some(distance);
    Ok(output)
}

/// `ccd` on the S16 chain named `name`; iterations above IK_RULE are refused.
pub fn solve_ccd(rig: &Rig, name: &str, values: &[f32], target: [f32; 3],
    iterations: usize, tol: f32) -> Result<IkOutput, AnimError> {
    solve_iterative(rig, name, values, target, iterations, tol, false)
}
/// `fabrik` on the S16 chain named `name`; iterations above IK_RULE are refused.
pub fn solve_fabrik(rig: &Rig, name: &str, values: &[f32], target: [f32; 3],
    iterations: usize, tol: f32) -> Result<IkOutput, AnimError> {
    solve_iterative(rig, name, values, target, iterations, tol, true)
}
pub fn register(_cases: &mut Vec<crate::Case>) {}
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use raa_anim::rig::IkChain;

    fn error(e: AnimError) -> JsValue {
        let value = js_sys::Error::new(&e.message);
        js_sys::Reflect::set(&value, &"code".into(), &e.code.into()).expect("fresh error");
        js_sys::Reflect::set(&value, &"path".into(), &e.path.into()).expect("fresh error");
        value.into()
    }

    fn invalid(message: &str) -> JsValue {
        let value = js_sys::TypeError::new(message);
        js_sys::Reflect::set(&value, &"code".into(), &"api.invalid_input".into())
            .expect("fresh error");
        js_sys::Reflect::set(&value, &"path".into(), &"/pose".into()).expect("fresh error");
        value.into()
    }

    fn floats(value: &JsValue, what: &str) -> Result<Vec<f32>, JsValue> {
        if let Some(array) = value.dyn_ref::<js_sys::Float32Array>() {
            return Ok(array.to_vec());
        }
        if js_sys::Array::is_array(value) {
            let array: &js_sys::Array = value.unchecked_ref();
            return array.iter().enumerate().map(|(i, v)| {
                v.as_f64().map(|n| n as f32)
                    .ok_or_else(|| invalid(&format!("{what}[{i}] must be a number")))
            }).collect();
        }
        Err(invalid(&format!("{what} must be an Array or Float32Array")))
    }

    fn pose_values(value: &JsValue) -> Result<Vec<f32>, JsValue> {
        let array = js_sys::Array::is_array(value)
            || value.is_instance_of::<js_sys::Float32Array>();
        if value.is_object() && !array {
            let local = js_sys::Reflect::get(value, &"local".into())
                .map_err(|_| invalid("cannot read pose.local"))?;
            if local.is_undefined() || local.is_null() {
                return Err(invalid("pose.local is required"));
            }
            return floats(&local, "pose.local");
        }
        floats(value, "pose")
    }

    fn vec3(value: &JsValue, what: &str) -> Result<[f32; 3], JsValue> {
        let values = floats(value, what)?;
        if values.len() != 3 {
            return Err(invalid(&format!("{what} must have 3 components")));
        }
        Ok([values[0], values[1], values[2]])
    }

    fn number(value: &JsValue, what: &str, default: f64) -> Result<f64, JsValue> {
        if value.is_undefined() || value.is_null() {
            return Ok(default);
        }
        value.as_f64().ok_or_else(|| invalid(&format!("{what} must be a number")))
    }

    fn chain<'a>(rig: &'a AnimationRig, name: &str) -> Result<&'a IkChain, JsValue> {
        raa_ik::chain(rig.shared_rig(), name).map_err(error)
    }

    fn iterations(value: &JsValue, rig: &AnimationRig, name: &str) -> Result<usize, JsValue> {
        let n = number(value, "iterations", chain(rig, name)?.iterations as f64)?;
        if !n.is_finite() || n < 0.0 || n.fract() != 0.0 || n > f64::from(u32::MAX) {
            return Err(invalid("iterations must be a u32 integer"));
        }
        Ok(n as usize)
    }

    fn output_to_js(out: IkOutput) -> Result<JsValue, JsValue> {
        let object = js_sys::Object::new();
        let set = |key: &str, value: &JsValue| {
            js_sys::Reflect::set(&object, &key.into(), value).map(drop)
        };
        set("local", &js_sys::Float32Array::from(&out.local[..]).into())?;
        set("model", &js_sys::Float32Array::from(&out.model[..]).into())?;
        set("skin", &js_sys::Float32Array::from(&out.skin[..]).into())?;
        if let Some(v) = out.converged { set("converged", &v.into())?; }
        if let Some(v) = out.passes { set("passes", &(v as u32).into())?; }
        if let Some(v) = out.distance { set("distance", &v.into())?; }
        Ok(object.into())
    }

    /// Two-bone solve of the named chain; `pole` defaults to the chain's pole.
    #[wasm_bindgen(js_namespace = ik, js_name = twoBone)]
    pub fn two_bone_js(rig: &AnimationRig, pose: &JsValue, name: &str, target: &JsValue,
        pole: &JsValue, weight: &JsValue) -> Result<JsValue, JsValue> {
        let chain = chain(rig, name)?;
        let pole = if pole.is_undefined() {
            chain.pole.map(|p| p.map(|v| v as f32))
        } else if pole.is_null() {
            None
        } else {
            Some(vec3(pole, "pole")?)
        };
        let weight = number(weight, "weight", f64::from(DEFAULT_WEIGHT))? as f32;
        let out = solve_two_bone(rig.shared_rig(), name, &pose_values(pose)?,
            vec3(target, "target")?, pole, weight).map_err(error)?;
        output_to_js(out)
    }

    fn iterative_js(rig: &AnimationRig, pose: &JsValue, name: &str, target: &JsValue,
        count: &JsValue, tolerance: &JsValue, fab: bool) -> Result<JsValue, JsValue> {
        let iterations = iterations(count, rig, name)?;
        let tol = number(tolerance, "tolerance", f64::from(DEFAULT_TOLERANCE))? as f32;
        let values = pose_values(pose)?;
        let target = vec3(target, "target")?;
        let out = solve_iterative(rig.shared_rig(), name, &values, target, iterations, tol, fab)
            .map_err(error)?;
        output_to_js(out)
    }

    /// CCD solve of the named chain; omitted iterations use the S16 chain value.
    #[wasm_bindgen(js_namespace = ik, js_name = ccd)]
    pub fn ccd_js(rig: &AnimationRig, pose: &JsValue, name: &str, target: &JsValue,
        iterations: &JsValue, tolerance: &JsValue) -> Result<JsValue, JsValue> {
        iterative_js(rig, pose, name, target, iterations, tolerance, false)
    }

    /// FABRIK solve of the named chain; omitted iterations use the S16 chain value.
    #[wasm_bindgen(js_namespace = ik, js_name = fabrik)]
    pub fn fabrik_js(rig: &AnimationRig, pose: &JsValue, name: &str, target: &JsValue,
        iterations: &JsValue, tolerance: &JsValue) -> Result<JsValue, JsValue> {
        iterative_js(rig, pose, name, target, iterations, tolerance, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_anim::rig::{ChainSolver, IkChain};
    use raa_ik::{IK_BONE_UNKNOWN, IK_CHAIN, IK_TARGET_NOT_FINITE};

    const RIG: &str = r#"{"format":"raa.rig","version":1,"name":"ik","bones":[
 {"name":"root","parent":null,"rest":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}},
 {"name":"hip","parent":"root","rest":{"t":[0,1,0],"r":[0,0,0,1],"s":[1,1,1]}},
 {"name":"knee","parent":"hip","rest":{"t":[0,0.5,0],"r":[0,0,0,1],"s":[1,1,1]}},
 {"name":"foot","parent":"knee","rest":{"t":[0,0.5,0],"r":[0,0,0,1],"s":[1,1,1]}},
 {"name":"toe","parent":"foot","rest":{"t":[0,0.5,0],"r":[0,0,0,1],"s":[1,1,1]}}],
 "roles":{},"ikChains":[
 {"name":"two","solver":"twoBone","bones":["hip","knee","foot"],"pole":[0,0,1],"iterations":8},
 {"name":"cd","solver":"ccd","bones":["hip","knee","foot","toe"],"iterations":16},
 {"name":"fab","solver":"fabrik","bones":["hip","knee","foot","toe"],"iterations":16}],
 "limits":[],"attach":{}}"#;
    const TARGET: [f32; 3] = [0.7, 1.4, 0.3];

    fn rig() -> Rig { Rig::load(RIG).unwrap() }

    fn flat(rig: &Rig) -> Vec<f32> {
        rig.rest_pose().unwrap().local.iter()
            .flat_map(|p| p.t.into_iter().chain(p.r).chain(p.s)).collect()
    }

    fn code<T: core::fmt::Debug>(result: Result<T, AnimError>) -> &'static str {
        result.unwrap_err().code
    }

    #[test]
    fn converts_locals_and_flattens_every_corrected_pose() {
        let rig = rig();
        let rest = rig.rest_pose().unwrap();
        let values = flat(&rig);
        assert_eq!(local_pose(&rig, &values).unwrap(), rest.local);
        let out = solve_two_bone(&rig, "two", &values, TARGET, None, 1.0).unwrap();
        assert_eq!(out.local.len(), rig.bones.len() * 10);
        assert_eq!(out.model.len(), rig.bones.len() * 16);
        assert_eq!(out.skin.len(), rig.bones.len() * 16);
        assert_eq!(out.converged, None);
        assert!(out.model.iter().chain(&out.skin).all(|v| v.is_finite()));
        let local = local_pose(&rig, &out.local).unwrap();
        assert_eq!(local[0], rest.local[0]);
        assert_ne!(local[1], rest.local[1]);
        assert_eq!(code(local_pose(&rig, &values[..3])), "api.invalid_input");
        for (name, out) in [
            ("cd", solve_ccd(&rig, "cd", &values, TARGET, 16, 0.001)),
            ("fab", solve_fabrik(&rig, "fab", &values, TARGET, 16, 0.001)),
        ] {
            let out = out.unwrap();
            assert_eq!(out.converged, Some(true), "{name}");
            assert!(out.passes.unwrap() > 0, "{name}");
            assert!(out.distance.unwrap() < 0.002, "{name}");
            assert_eq!(out.local.len(), rig.bones.len() * 10, "{name}");
        }
    }

    #[test]
    fn propagates_ik_chain_target_and_bone_codes() {
        let rig = rig();
        let values = flat(&rig);
        assert_eq!(code(solve_two_bone(&rig, "missing", &values, TARGET, None, 1.0)), IK_CHAIN);
        assert_eq!(code(solve_two_bone(&rig, "two", &values, [f32::NAN; 3], None, 1.0)),
            IK_TARGET_NOT_FINITE);
        assert_eq!(code(solve_two_bone(&rig, "two", &values, TARGET, None, f32::NAN)),
            IK_TARGET_NOT_FINITE);
        assert_eq!(code(solve_ccd(&rig, "cd", &values, TARGET, 17, 0.001)), IK_CHAIN);
        assert_eq!(code(solve_fabrik(&rig, "fab", &values, TARGET, 16, f32::NAN)),
            IK_TARGET_NOT_FINITE);
        let mut broken = rig;
        broken.ik_chains.push(IkChain {
            name: "bad".into(), solver: ChainSolver::Ccd, bones: vec![1, 2, 99],
            pole: None, iterations: 4,
        });
        assert_eq!(code(solve_ccd(&broken, "bad", &values, TARGET, 4, 0.001)), IK_BONE_UNKNOWN);
    }
}
