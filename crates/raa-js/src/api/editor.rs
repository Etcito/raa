//! Generic editor display controls; descriptors are checked before any state mutation.
use crate::convert::{INVALID_ARGUMENT, api_error, js};
use crate::engine::{Engine, InstallCtx};
use crate::state::EngineState;
use raa_core::{GpuContext, RaaError};
use raa_lit::WireframeMode;
use raa_web::FrameInput;
use wasm_bindgen::prelude::*;

pub(crate) fn install(_: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    Ok(())
}
pub(crate) fn frame(_: &mut EngineState, _: &GpuContext, _: &FrameInput) -> Result<(), RaaError> {
    Ok(())
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Global default when instance is omitted. "inherit" clears an instance override.
    #[wasm_bindgen(js_name = setWireframe)]
    pub fn set_wireframe(&mut self, mode: String, instance: Option<u32>) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            if let Some(id) = instance {
                st.instance_group(id)?.check_live()?;
            }
            if mode == "inherit" {
                let id = instance
                    .ok_or_else(|| api_error(INVALID_ARGUMENT, "inherit requires an instance"))?;
                st.lit.handle().clear_wireframe(id);
                return Ok(());
            }
            let mode = match mode.as_str() {
                "off" => WireframeMode::Off,
                "overlay" => WireframeMode::Overlay,
                "only" => WireframeMode::Only,
                _ => {
                    return Err(api_error(
                        INVALID_ARGUMENT,
                        "wireframe mode must be off, overlay, only or inherit",
                    ));
                }
            };
            st.lit.handle().set_wireframe(mode, instance);
            Ok(())
        })
    }

    /// Null disables the instance's selection outline; otherwise {color?: [r,g,b]}.
    #[wasm_bindgen(js_name = setSelectionOutline)]
    pub fn set_selection_outline(&mut self, instance: u32, desc: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            st.instance_group(instance)?.check_live()?;
            let color = if desc.is_null() {
                None
            } else {
                let color = js::optional_vec3(&desc, "color")?.unwrap_or([1.0, 0.6, 0.05]);
                if color.iter().any(|c| !(0.0..=1.0).contains(c)) {
                    return Err(api_error(
                        INVALID_ARGUMENT,
                        "outline color must be sRGB in [0,1]",
                    ));
                }
                Some(color)
            };
            st.lit.handle().set_outline(instance, color);
            Ok(())
        })
    }
}
