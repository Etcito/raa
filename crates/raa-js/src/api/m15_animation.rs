//! m15 animation hooks: shared per-instance tint, parameters, squash and jiggle.
//! Batched extras validate completely before writing. Springs consume current node
//! world matrices before lit.prepare, in instance-handle order.

use raa_core::RaaError;
use raa_lit::InstanceExtra;

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "api.invalid_argument",
        message,
    }
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_tint(rgba: [f32; 4]) -> Result<[f32; 4], RaaError> {
    if rgba.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(invalid(
            "tint must contain four finite, nonnegative numbers".into(),
        ));
    }
    let out = [
        srgb_to_linear(rgba[0]),
        srgb_to_linear(rgba[1]),
        srgb_to_linear(rgba[2]),
        rgba[3],
    ];
    if out.iter().any(|v| !v.is_finite()) {
        return Err(invalid("linear tint exceeds the finite float range".into()));
    }
    Ok(out)
}

fn normalize_axis(axis: [f32; 3]) -> Result<[f32; 3], RaaError> {
    // f64 avoids overflow and underflow for finite f32 axes.
    let length = axis
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt();
    if !length.is_finite() || length == 0.0 {
        return Err(invalid("squash axis must be finite and nonzero".into()));
    }
    Ok(axis.map(|v| (f64::from(v) / length) as f32))
}

fn record_to_extra(data: &[f32; 32]) -> Result<InstanceExtra, RaaError> {
    if data.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "instance extras must contain only finite floats (including reserved)".into(),
        ));
    }
    let axis = normalize_axis([data[4], data[5], data[6]])?;
    Ok(InstanceExtra {
        tint: linear_tint(data[0..4].try_into().unwrap())?,
        squash: [axis[0], axis[1], axis[2], data[7]],
        jiggle: data[8..12].try_into().unwrap(),
        params: std::array::from_fn(|i| data[12 + i * 4..16 + i * 4].try_into().unwrap()),
        tint_set: 0,
    })
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::engine::{Engine, InstallCtx};
    use crate::state::EngineState;
    use js_sys::{Array, Float32Array, Reflect, Uint32Array};
    use raa_core::GpuContext;
    use raa_deform::{JiggleConfig, JiggleSprings};
    use raa_web::FrameInput;
    use wasm_bindgen::prelude::*;

    #[derive(Default)]
    struct JiggleTags(Vec<(u32, u32)>);
    // Configured height survives direct batched writes to the jiggle record.
    #[derive(Default)]
    struct JiggleHeights(std::collections::BTreeMap<u32, f32>);

    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        raa_deform::install(ctx.lit)?;
        ctx.features.insert(JiggleSprings::default());
        ctx.features.insert(JiggleTags::default());
        ctx.features.insert(JiggleHeights::default());
        Ok(())
    }

    pub(crate) fn frame(
        st: &mut EngineState,
        _gpu: &GpuContext,
        input: &FrameInput,
    ) -> Result<(), RaaError> {
        if st
            .features
            .get::<JiggleTags>()
            .is_none_or(|tags| tags.0.is_empty())
        {
            return Ok(());
        }
        st.scene.update_world();
        let mut i = 0;
        while let Some(&(handle, tag)) = st
            .features
            .get::<JiggleTags>()
            .and_then(|tags| tags.0.get(i))
        {
            let world = st
                .instance_group(handle)
                .ok()
                .filter(|group| !group.stale && group.tag == tag)
                .and_then(|group| st.scene.world(group.node).ok());
            if let Some(world) = world {
                let offset = st.features.require_mut::<JiggleSprings>("jiggle")?.step(
                    (input.dt_ms / 1000.0) as f32,
                    tag,
                    &world,
                );
                let height = st
                    .features
                    .require_mut::<JiggleHeights>("jiggle heights")?
                    .0[&tag];
                let mut extra = st.lit.instance_extra(tag);
                extra.jiggle = [offset[0], offset[1], offset[2], height];
                st.lit.set_instance_extra(tag, extra);
                i += 1;
            } else {
                st.features
                    .require_mut::<JiggleSprings>("jiggle")?
                    .set(tag, None);
                st.features
                    .require_mut::<JiggleHeights>("jiggle heights")?
                    .0
                    .remove(&tag);
                st.features
                    .require_mut::<JiggleTags>("jiggle tags")?
                    .0
                    .remove(i);
            }
        }
        Ok(())
    }

    fn get(obj: &JsValue, key: &str) -> Result<Option<JsValue>, RaaError> {
        if !obj.is_object() {
            return Err(invalid(format!("expected an object holding '{key}'")));
        }
        let value = Reflect::get(obj, &JsValue::from_str(key))
            .map_err(|_| invalid(format!("cannot read '{key}'")))?;
        Ok(if value.is_null() || value.is_undefined() {
            None
        } else {
            Some(value)
        })
    }

    fn optional_f64(obj: &JsValue, key: &str) -> Result<Option<f64>, RaaError> {
        get(obj, key)?
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| invalid(format!("'{key}' must be a number")))
            })
            .transpose()
    }

    fn numbers<const N: usize>(
        value: &JsValue,
        field: &str,
        nonnegative: bool,
    ) -> Result<[f32; N], RaaError> {
        let bad = || invalid(format!("{field} must contain exactly {N} finite numbers"));
        if !Array::is_array(value) && !value.is_instance_of::<Float32Array>() {
            return Err(bad());
        }
        let a = Array::from(value);
        if a.length() != N as u32 {
            return Err(bad());
        }
        let mut out = [0.0; N];
        for (i, slot) in out.iter_mut().enumerate() {
            let number = a.get(i as u32).as_f64().ok_or_else(bad)?;
            if nonnegative && number < 0.0 {
                return Err(invalid(format!("{field} must be nonnegative")));
            }
            *slot = number as f32;
            if !slot.is_finite() {
                return Err(bad());
            }
        }
        Ok(out)
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        #[wasm_bindgen(js_name = setInstanceTint)]
        pub fn set_instance_tint(&mut self, instance: u32, rgba: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let tag = st.instance_group(instance)?.tag;
                let tint = linear_tint(numbers(&rgba, "rgba", true)?)?;
                let mut extra = st.lit.instance_extra(tag);
                extra.tint = tint;
                // S23: graph-mode materials read the tint once it is set.
                extra.tint_set = 1;
                st.lit.set_instance_extra(tag, extra);
                Ok(())
            })
        }

        #[wasm_bindgen(js_name = setInstanceParams)]
        pub fn set_instance_params(
            &mut self,
            instance: u32,
            params: JsValue,
        ) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let tag = st.instance_group(instance)?.tag;
                let values = numbers::<16>(&params, "params", false)?;
                let mut extra = st.lit.instance_extra(tag);
                extra.params =
                    std::array::from_fn(|i| values[i * 4..i * 4 + 4].try_into().unwrap());
                st.lit.set_instance_extra(tag, extra);
                Ok(())
            })
        }

        #[wasm_bindgen(js_name = setSquash)]
        pub fn set_squash(&mut self, instance: u32, settings: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let tag = st.instance_group(instance)?.tag;
                let axis = match get(&settings, "axis")? {
                    Some(v) => numbers(&v, "axis", false)?,
                    None => [0.0, 1.0, 0.0],
                };
                let axis = normalize_axis(axis)?;
                let amount = optional_f64(&settings, "amount")?
                    .ok_or_else(|| invalid("squash amount is required".into()))?
                    as f32;
                if !amount.is_finite() {
                    return Err(invalid("squash amount must be finite".into()));
                }
                let mut extra = st.lit.instance_extra(tag);
                extra.squash = [axis[0], axis[1], axis[2], amount];
                st.lit.set_instance_extra(tag, extra);
                Ok(())
            })
        }

        #[wasm_bindgen(js_name = setJiggle)]
        pub fn set_jiggle(&mut self, instance: u32, settings: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let tag = st.instance_group(instance)?.tag;
                let config = if settings.is_null() {
                    None
                } else {
                    let mut c = JiggleConfig {
                        height_m: optional_f64(&settings, "height")?
                            .ok_or_else(|| invalid("jiggle height is required".into()))?
                            as f32,
                        ..JiggleConfig::default()
                    };
                    for (key, slot) in [
                        ("stiffness", &mut c.stiffness),
                        ("damping", &mut c.damping),
                        ("amount", &mut c.amount),
                    ] {
                        if let Some(v) = optional_f64(&settings, key)? {
                            *slot = v as f32;
                        }
                    }
                    // with_state maps RaaError through error_to_js, retaining deform.invalid.
                    c.validate()?;
                    Some(c)
                };
                st.features
                    .require_mut::<JiggleSprings>("jiggle")?
                    .set(tag, config);
                let tags = &mut st.features.require_mut::<JiggleTags>("jiggle tags")?.0;
                match (tags.binary_search_by_key(&instance, |p| p.0), config) {
                    (Err(i), Some(_)) => tags.insert(i, (instance, tag)),
                    (Ok(i), None) => {
                        tags.remove(i);
                    }
                    _ => {}
                }
                let heights = &mut st
                    .features
                    .require_mut::<JiggleHeights>("jiggle heights")?
                    .0;
                let mut extra = st.lit.instance_extra(tag);
                if let Some(c) = config {
                    heights.insert(tag, c.height_m);
                    extra.jiggle[3] = c.height_m;
                } else {
                    heights.remove(&tag);
                    extra.jiggle = [0.0; 4];
                }
                st.lit.set_instance_extra(tag, extra);
                Ok(())
            })
        }

        #[wasm_bindgen(js_name = setInstanceExtras)]
        pub fn set_instance_extras(
            &mut self,
            instances: Uint32Array,
            data: Float32Array,
        ) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let count = instances.length();
                let length = data.length();
                if u64::from(length) != 32 * u64::from(count) {
                    return Err(RaaError::FrameGraph {
                        code: "api.length_mismatch",
                        message: format!(
                            "data.length {length} must equal 32 * instances.length {count}"
                        ),
                    });
                }
                // Stage the entire call: neither a bad later handle nor a bad later
                // record can leave an earlier instance partially updated.
                let mut writes = Vec::with_capacity(count as usize);
                for i in 0..count {
                    let tag = st.instance_group(instances.get_index(i))?.tag;
                    let record = std::array::from_fn(|j| data.get_index(i * 32 + j as u32));
                    // The record's reserved lanes are not the S23 flag: keep the instance's.
                    let tint_set = st.lit.instance_extra(tag).tint_set;
                    writes.push((
                        tag,
                        InstanceExtra {
                            tint_set,
                            ..record_to_extra(&record)?
                        },
                    ));
                }
                for (tag, extra) in writes {
                    st.lit.set_instance_extra(tag, extra);
                }
                Ok(())
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> [f32; 32] {
        let mut values = [0.0; 32];
        values[..4].copy_from_slice(&[0.0, 0.04045, 0.5, 0.7]);
        values[5] = 2.0;
        values[7] = -2.0;
        for (i, value) in values.iter_mut().enumerate().skip(8) {
            *value = i as f32;
        }
        values
    }

    #[test]
    fn tint_uses_piecewise_srgb_and_preserves_alpha() {
        let tint = linear_tint([0.04045, 0.5, 1.0, 0.7]).unwrap();
        assert!((tint[0] - 0.003130805).abs() < 1e-8);
        assert!((tint[1] - 0.21404114).abs() < 1e-7);
        assert_eq!(&tint[2..], &[1.0, 0.7]);
        assert_eq!(srgb_to_linear(0.0), 0.0);
        for bad in [-0.1, f32::NAN, f32::INFINITY] {
            for i in 0..4 {
                let mut rgba = [1.0; 4];
                rgba[i] = bad;
                assert_eq!(
                    linear_tint(rgba).unwrap_err().code(),
                    "api.invalid_argument"
                );
            }
        }
    }

    #[test]
    fn axes_are_normalized_even_at_float_extremes() {
        assert_eq!(normalize_axis([0.0, 2.0, 0.0]).unwrap(), [0.0, 1.0, 0.0]);
        assert_eq!(
            normalize_axis([f32::MAX, 0.0, 0.0]).unwrap(),
            [1.0, 0.0, 0.0]
        );
        assert_eq!(
            normalize_axis([0.0, -f32::MIN_POSITIVE, 0.0]).unwrap(),
            [0.0, -1.0, 0.0]
        );
        let axis = normalize_axis([1.0, 2.0, 3.0]).unwrap();
        assert!((axis.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-6);
        for axis in [[0.0; 3], [f32::NAN, 1.0, 0.0], [0.0, f32::INFINITY, 0.0]] {
            assert_eq!(
                normalize_axis(axis).unwrap_err().code(),
                "api.invalid_argument"
            );
        }
    }

    #[test]
    fn record_layout_preserves_unclamped_amount_jiggle_and_parameters() {
        let values = record();
        let extra = record_to_extra(&values).unwrap();
        assert_eq!(
            extra.tint,
            linear_tint(values[..4].try_into().unwrap()).unwrap()
        );
        assert_eq!(extra.squash, [0.0, 1.0, 0.0, -2.0]);
        assert_eq!(extra.jiggle, [8.0, 9.0, 10.0, 11.0]);
        assert_eq!(
            extra.params,
            [
                [12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0],
                [24.0, 25.0, 26.0, 27.0]
            ]
        );
        let mut other = values;
        other[28..].fill(-123.0);
        assert_eq!(record_to_extra(&other).unwrap(), extra);
    }

    #[test]
    fn record_rejects_nonfinite_values_including_reserved() {
        for i in 0..32 {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut values = record();
                values[i] = bad;
                assert_eq!(
                    record_to_extra(&values).unwrap_err().code(),
                    "api.invalid_argument"
                );
            }
        }
        let mut values = record();
        values[5] = 0.0;
        assert!(record_to_extra(&values).is_err());
        values[5] = 1.0;
        values[3] = -1.0;
        assert!(record_to_extra(&values).is_err());
    }
}
