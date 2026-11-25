//! Installation and statistics of the DDGI probe volume (`pass.gi.ddgi`). The render graph
//! configures it; there is no standalone setter.

use js_sys::{Array, Object, Reflect};
use raa_core::RaaError;
use raa_sdf_trace::mesh::{DdgiFeature, DdgiHandle, MeshTraceHandle};
use wasm_bindgen::prelude::*;

use crate::engine::{Engine, InstallCtx};

pub(crate) fn install_for_graph(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    let mesh = ctx
        .features
        .require_mut::<MeshTraceHandle>("mesh trace")?
        .clone();
    let handle = DdgiFeature::install(&mut ctx.pipeline.borrow_mut(), mesh)?;
    ctx.features.insert(handle);
    Ok(())
}

fn put(object: &Object, key: &str, value: JsValue) {
    Reflect::set(object, &key.into(), &value).expect("fresh object");
}

#[wasm_bindgen]
impl Engine {
    /// Snapshot of the active DDGI volume and its update schedule, or null.
    #[wasm_bindgen(js_name = ddgiGiStats)]
    pub fn ddgi_gi_stats(&self) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| {
            let Some(handle) = st.features.get::<DdgiHandle>() else {
                return Ok(JsValue::NULL);
            };
            let state = handle.borrow();
            let result = Object::new();
            put(&result, "active", state.active().into());
            let s = state.settings;
            let counts = Array::new();
            for n in s.counts {
                counts.push(&n.into());
            }
            put(&result, "counts", counts.into());
            put(&result, "spacing", s.spacing.into());
            put(&result, "raysPerProbe", s.rays_per_probe.into());
            put(&result, "updateFraction", s.update_fraction.into());
            put(&result, "hysteresis", s.hysteresis.into());
            put(&result, "budgetMs", s.budget_ms.into());
            put(&result, "burstMs", s.burst_ms.into());
            if let Some(stats) = state.stats() {
                put(&result, "probes", stats.probes.into());
                put(&result, "scheduled", stats.scheduled.into());
                put(&result, "updateFrames", (stats.update_frames as f64).into());
                put(&result, "recycled", (stats.recycled as f64).into());
                let base = Array::new();
                for b in stats.base {
                    base.push(&b.into());
                }
                put(&result, "base", base.into());
                put(&result, "gpuBytes", (stats.gpu_bytes as f64).into());
                put(&result, "burst", stats.burst.into());
                put(&result, "visible", stats.visibility.visible.into());
                put(&result, "visibleValid", stats.visibility.valid.into());
                put(
                    &result,
                    "visibleConverged",
                    stats.visibility.converged.into(),
                );
                put(&result, "convergeAge", stats.converge_age.into());
                put(&result, "msPerProbe", stats.ms_per_probe.into());
                put(
                    &result,
                    "updateMs",
                    stats.update_ms.map_or(JsValue::NULL, JsValue::from),
                );
                put(&result, "screenResolve", stats.screen_resolve.into());
            }
            Ok(result.into())
        })
    }
}
