//! m21 views API (S25, S26): split-screen viewports (`setViewports`), per-viewport cameras
//! (`setViewportCamera`), the DPR-aware `resize` and debug overlay lines
//! (`setOverlayLines`, `clearOverlay`).
//!
//! This file holds the state only ([`ViewsState`], stored in the feature map by `install`);
//! m07_pipeline reads it to render one pass chain per viewport and to draw the overlays
//! after tonemap. Viewport 0 has no camera slot of its own: it is the scene camera, so
//! `setCamera` and `setViewportCamera(0, ..)` write the same camera. A viewport i > 0
//! without a camera renders with viewport 0's camera.
//!
//! `resize(cssWidth, cssHeight, dpr?)` replaces the v0 `resize(width, height)`: the backing
//! size is round(css * min(dpr, 2)), and an omitted `dpr` is 1, so the 2-argument call keeps
//! its v0 meaning (the arguments are backing pixels).
//!
//! Refusals: `views.count`, `views.rect`, `views.index`, `overlay.layer`, `overlay.length`;
//! malformed JS arguments, a bad camera, dpr or widthPx throw `api.invalid_argument`.
//! The pure part below compiles on the host (tests/views.rs includes this file).

// The readers (viewports, camera, overlays) are m07_pipeline's, which lands in the next task.
#![allow(dead_code)]

use raa_core::RaaError;

pub const MAX_VIEWPORTS: usize = 4;
pub const OVERLAY_LAYERS: usize = 8;
pub const DPR_CAP: f64 = 2.0;
/// Rounding slack for `x + width <= 1` and for rects that only touch.
const EPS: f64 = 1e-9;

fn error(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

fn invalid(message: String) -> RaaError {
    error("api.invalid_argument", message)
}

/// A viewport as fractions of the canvas, origin at the top left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl ViewRect {
    pub const FULL: ViewRect = ViewRect {
        x: 0.0,
        y: 0.0,
        width: 1.0,
        height: 1.0,
    };

    fn overlaps(&self, o: &ViewRect) -> bool {
        self.x < o.x + o.width - EPS
            && o.x < self.x + self.width - EPS
            && self.y < o.y + o.height - EPS
            && o.y < self.y + self.height - EPS
    }
}

/// A perspective camera of a viewport; its up vector is +Y.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewCamera {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
}

/// The overlay segments of one layer: positions xyz per vertex (2 vertices per segment),
/// linear rgba per vertex.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OverlayLayer {
    pub positions: Vec<f32>,
    pub colors: Vec<f32>,
    pub depth_test: bool,
    pub width_px: f32,
}

impl OverlayLayer {
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// `views.count` (0 or more than 4 rects), then `views.rect` (a value non-finite or outside
/// 0..1, a zero size, `x + width` or `y + height` above 1, or two rects overlapping).
pub fn validate_viewports(rects: &[ViewRect]) -> Result<(), RaaError> {
    if rects.is_empty() || rects.len() > MAX_VIEWPORTS {
        return Err(error(
            "views.count",
            format!("1..{MAX_VIEWPORTS} viewports, got {}", rects.len()),
        ));
    }
    for (i, r) in rects.iter().enumerate() {
        let values = [r.x, r.y, r.width, r.height];
        let inside = values
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v));
        if !inside || r.width <= 0.0 || r.height <= 0.0 {
            return Err(error(
                "views.rect",
                format!("rect {i} must hold finite values in 0..1 with a non-zero size, got {r:?}"),
            ));
        }
        if r.x + r.width > 1.0 + EPS || r.y + r.height > 1.0 + EPS {
            return Err(error(
                "views.rect",
                format!(
                    "rect {i} leaves the canvas (x + width and y + height must be <= 1), got {r:?}"
                ),
            ));
        }
    }
    for i in 0..rects.len() {
        for j in i + 1..rects.len() {
            if rects[i].overlaps(&rects[j]) {
                return Err(error("views.rect", format!("rects {i} and {j} overlap")));
            }
        }
    }
    Ok(())
}

/// The setCamera rules for a perspective camera with +Y up (`api.invalid_argument`).
pub fn validate_view_camera(c: &ViewCamera) -> Result<(), RaaError> {
    let scalars = [c.fov_y_deg, c.near, c.far];
    if !c
        .eye
        .iter()
        .chain(&c.target)
        .chain(&scalars)
        .all(|v| v.is_finite())
    {
        return Err(invalid(
            "the viewport camera must hold finite numbers".into(),
        ));
    }
    if !(c.fov_y_deg > 0.0 && c.fov_y_deg < 180.0) {
        return Err(invalid(format!(
            "'fovYDeg' must be in (0, 180), got {}",
            c.fov_y_deg
        )));
    }
    if !(c.near > 0.0 && c.far > c.near) {
        return Err(invalid(format!(
            "'near' must be > 0 and 'far' > 'near', got near {} far {}",
            c.near, c.far
        )));
    }
    let d = [0, 1, 2].map(|k| f64::from(c.target[k]) - f64::from(c.eye[k]));
    let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if len2 <= 1e-12 {
        return Err(invalid("'eye' and 'target' must differ".into()));
    }
    // |d x +Y|^2 = dx^2 + dz^2, relative to |d|^2.
    if (d[0] * d[0] + d[2] * d[2]) / len2 <= 1e-12 {
        return Err(invalid(
            "the view direction must not be parallel to +Y".into(),
        ));
    }
    Ok(())
}

/// The backing size of `resize`: round(css * min(dpr, DPR_CAP)), at least 1. `dpr` None is
/// 1 (the v0 2-argument call). Non-finite sizes, or a dpr that is not finite and > 0, are
/// refused (`api.invalid_argument`).
pub fn backing_size(css_w: f64, css_h: f64, dpr: Option<f64>) -> Result<(u32, u32), RaaError> {
    let dpr = dpr.unwrap_or(1.0);
    if !(dpr.is_finite() && dpr > 0.0) {
        return Err(invalid(format!("'dpr' must be finite and > 0, got {dpr}")));
    }
    if !(css_w.is_finite() && css_h.is_finite()) {
        return Err(invalid(format!(
            "resize sizes must be finite, got {css_w} x {css_h}"
        )));
    }
    let scale = dpr.min(DPR_CAP);
    let px = |css: f64| (css * scale).round().clamp(1.0, f64::from(u32::MAX)) as u32;
    Ok((px(css_w), px(css_h)))
}

/// `overlay.layer` (outside 0..7), then `overlay.length` (positions not a multiple of 6,
/// or colors not 4 per vertex).
pub fn validate_overlay(layer: f64, positions: usize, colors: usize) -> Result<usize, RaaError> {
    let index = overlay_layer(layer)?;
    if positions % 6 != 0 {
        return Err(error(
            "overlay.length",
            format!("positions must be a multiple of 6 (xyz pairs), got {positions}"),
        ));
    }
    let vertices = positions / 3;
    if colors != vertices * 4 {
        return Err(error(
            "overlay.length",
            format!(
                "{vertices} vertices need {} colors (rgba), got {colors}",
                vertices * 4
            ),
        ));
    }
    Ok(index)
}

/// The layer index of `layer` (`overlay.layer` unless an integer in 0..7).
pub fn overlay_layer(layer: f64) -> Result<usize, RaaError> {
    if layer.fract() == 0.0 && (0.0..OVERLAY_LAYERS as f64).contains(&layer) {
        Ok(layer as usize)
    } else {
        Err(error(
            "overlay.layer",
            format!("overlay layers are 0..{}, got {layer}", OVERLAY_LAYERS - 1),
        ))
    }
}

/// The views state read by m07_pipeline. The default is one full-canvas viewport (today's
/// single view) and no overlay.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewsState {
    rects: Vec<ViewRect>,
    /// Slot i is viewport i's camera; slot 0 stays None (the scene camera).
    cameras: [Option<ViewCamera>; MAX_VIEWPORTS],
    overlays: [OverlayLayer; OVERLAY_LAYERS],
}

impl Default for ViewsState {
    fn default() -> Self {
        Self {
            rects: vec![ViewRect::FULL],
            cameras: [None; MAX_VIEWPORTS],
            overlays: Default::default(),
        }
    }
}

impl ViewsState {
    /// The viewports, 1..4, in index order.
    pub fn viewports(&self) -> &[ViewRect] {
        &self.rects
    }

    /// Replaces the viewports; nothing changes on a refusal. Cameras of indices that leave
    /// the range are kept and apply again when the viewport comes back.
    pub fn set_viewports(&mut self, rects: &[ViewRect]) -> Result<(), RaaError> {
        validate_viewports(rects)?;
        self.rects.clear();
        self.rects.extend_from_slice(rects);
        Ok(())
    }

    /// `views.index` unless `index` is an integer below the viewport count.
    pub fn check_index(&self, index: f64) -> Result<usize, RaaError> {
        if index.fract() == 0.0 && index >= 0.0 && index < self.rects.len() as f64 {
            Ok(index as usize)
        } else {
            Err(error(
                "views.index",
                format!(
                    "viewport index must be in 0..{}, got {index}",
                    self.rects.len()
                ),
            ))
        }
    }

    /// Stores the camera of viewport `index` > 0 (viewport 0 is the scene camera, written
    /// by the caller). Validates the index and the camera first.
    pub fn set_camera(&mut self, index: f64, cam: ViewCamera) -> Result<usize, RaaError> {
        let i = self.check_index(index)?;
        validate_view_camera(&cam)?;
        if i > 0 {
            self.cameras[i] = Some(cam);
        }
        Ok(i)
    }

    /// The own camera of viewport `index`; None means the scene camera (always for 0).
    pub fn camera(&self, index: usize) -> Option<&ViewCamera> {
        self.cameras.get(index).and_then(Option::as_ref)
    }

    /// Replaces the segments of `layer`; nothing changes on a refusal.
    pub fn set_overlay_lines(
        &mut self,
        layer: f64,
        positions: &[f32],
        colors: &[f32],
        depth_test: bool,
        width_px: f32,
    ) -> Result<(), RaaError> {
        let i = validate_overlay(layer, positions.len(), colors.len())?;
        if !(width_px.is_finite() && width_px > 0.0) {
            return Err(invalid(format!(
                "'widthPx' must be finite and > 0, got {width_px}"
            )));
        }
        if !positions.iter().chain(colors).all(|v| v.is_finite()) {
            return Err(invalid(
                "overlay positions and colors must be finite".into(),
            ));
        }
        let o = &mut self.overlays[i];
        o.positions.clear();
        o.positions.extend_from_slice(positions);
        o.colors.clear();
        o.colors.extend_from_slice(colors);
        o.depth_test = depth_test;
        o.width_px = width_px;
        Ok(())
    }

    /// Empties `layer` (`overlay.layer`).
    pub fn clear_overlay(&mut self, layer: f64) -> Result<(), RaaError> {
        let o = &mut self.overlays[overlay_layer(layer)?];
        o.positions.clear();
        o.colors.clear();
        Ok(())
    }

    /// The 8 overlay layers in draw order.
    pub fn overlays(&self) -> &[OverlayLayer; OVERLAY_LAYERS] {
        &self.overlays
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::convert::js;
    use crate::engine::{Engine, InstallCtx};
    use crate::state::EngineState;
    use js_sys::{Array, Reflect};
    use raa_core::GpuContext;
    use raa_math::Vec3;
    use raa_web::{FrameInput, error_to_js};
    use wasm_bindgen::prelude::*;

    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        ctx.features.insert(ViewsState::default());
        Ok(())
    }

    pub(crate) fn frame(
        _st: &mut EngineState,
        _gpu: &GpuContext,
        _input: &FrameInput,
    ) -> Result<(), RaaError> {
        Ok(())
    }

    fn views(st: &mut EngineState) -> Result<&mut ViewsState, RaaError> {
        st.features.require_mut::<ViewsState>("views")
    }

    /// A rect field as a raw number (`views.rect` when missing or not a number).
    fn rect_field(i: usize, obj: &JsValue, key: &str) -> Result<f64, RaaError> {
        Reflect::get(obj, &JsValue::from_str(key))
            .ok()
            .and_then(|v| v.as_f64())
            .ok_or_else(|| error("views.rect", format!("rect {i}: '{key}' must be a number")))
    }

    fn rects_from_js(rects: &JsValue) -> Result<Vec<ViewRect>, RaaError> {
        if !Array::is_array(rects) {
            return Err(invalid("setViewports expects an array of rects".into()));
        }
        let arr = Array::from(rects);
        if arr.length() == 0 || arr.length() as usize > MAX_VIEWPORTS {
            return Err(error(
                "views.count",
                format!("1..{MAX_VIEWPORTS} viewports, got {}", arr.length()),
            ));
        }
        arr.iter()
            .enumerate()
            .map(|(i, r)| {
                if !r.is_object() {
                    return Err(error("views.rect", format!("rect {i} must be an object")));
                }
                Ok(ViewRect {
                    x: rect_field(i, &r, "x")?,
                    y: rect_field(i, &r, "y")?,
                    width: rect_field(i, &r, "width")?,
                    height: rect_field(i, &r, "height")?,
                })
            })
            .collect()
    }

    fn camera_from_js(cam: &JsValue) -> Result<ViewCamera, RaaError> {
        Ok(ViewCamera {
            eye: js::vec3(cam, "eye")?,
            target: js::vec3(cam, "target")?,
            fov_y_deg: js::required_f64(cam, "fovYDeg")? as f32,
            near: js::required_f64(cam, "near")? as f32,
            far: js::required_f64(cam, "far")? as f32,
        })
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        /// Sets 1..4 non-overlapping viewports as canvas fractions (`views.count`,
        /// `views.rect`). `[{x: 0, y: 0, width: 1, height: 1}]` is the single view.
        #[wasm_bindgen(js_name = setViewports)]
        pub fn set_viewports(&mut self, rects: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let rects = rects_from_js(&rects)?;
                views(st)?.set_viewports(&rects)
            })
        }

        /// Sets the camera of viewport `index` (`views.index`): `{eye, target, fovYDeg,
        /// near, far}`, +Y up. Viewport 0 is the scene camera (what `setCamera` sets).
        #[wasm_bindgen(js_name = setViewportCamera)]
        pub fn set_viewport_camera(&mut self, index: f64, cam: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                views(st)?.check_index(index)?;
                let c = camera_from_js(&cam)?;
                if views(st)?.set_camera(index, c)? == 0 {
                    let sc = &mut st.scene.camera;
                    sc.eye = Vec3::from_array(c.eye);
                    sc.target = Vec3::from_array(c.target);
                    sc.up = Vec3::Y;
                    sc.fov_y_rad = c.fov_y_deg.to_radians();
                    sc.ortho_height = None;
                    sc.near = c.near;
                    sc.far = c.far;
                }
                Ok(())
            })
        }

        /// Resizes the canvas backing store to round(css * min(dpr, 2)) (zero clamps to 1);
        /// the camera aspect follows. `dpr` omitted is 1: the v0 call `resize(width,
        /// height)` passes backing pixels and keeps that meaning.
        pub fn resize(
            &mut self,
            css_width: f64,
            css_height: f64,
            dpr: Option<f64>,
        ) -> Result<(), JsValue> {
            let (w, h) = backing_size(css_width, css_height, dpr).map_err(|e| error_to_js(&e))?;
            self.resize_backing(w, h)
        }

        /// Replaces the segments of overlay `layer` 0..7: positions are xyz pairs in world
        /// space, colors linear rgba per vertex; `opts` = `{depthTest, widthPx}`
        /// (`overlay.layer`, `overlay.length`).
        #[wasm_bindgen(js_name = setOverlayLines)]
        pub fn set_overlay_lines(
            &mut self,
            layer: f64,
            positions: &[f32],
            colors: &[f32],
            opts: JsValue,
        ) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                validate_overlay(layer, positions.len(), colors.len())?;
                let depth_test = js::bool(&opts, "depthTest")?;
                let width_px = js::required_f64(&opts, "widthPx")? as f32;
                views(st)?.set_overlay_lines(layer, positions, colors, depth_test, width_px)
            })
        }

        /// Empties overlay `layer` 0..7 (`overlay.layer`).
        #[wasm_bindgen(js_name = clearOverlay)]
        pub fn clear_overlay(&mut self, layer: f64) -> Result<(), JsValue> {
            self.with_state(|_, st| views(st)?.clear_overlay(layer))
        }
    }
}
