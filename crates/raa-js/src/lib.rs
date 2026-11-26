//! Frozen wasm-bindgen API v0 of RAA; no consumer is named.
//!
//! `convert` holds the pure-Rust helpers (error mapping, handle table,
//! descriptor conversion) that are tested natively.

pub mod convert;

// Whole-object billboard level and direction selection (host and wasm32).
pub mod billboard_selection;

// Generation-owned foliage assets and whole-tree representation groups (host and wasm32).
pub mod foliage_state;

// Used by the engine and the api files (wasm32); unit-tested on the host.
#[allow(dead_code)]
pub(crate) mod feature_map;

// The engine state (host and wasm32); the engine and the api files consume it.
#[allow(dead_code)]
pub(crate) mod state;

mod trace;

#[cfg(target_arch = "wasm32")]
mod engine;
#[cfg(target_arch = "wasm32")]
pub use engine::Engine;

/// The api files `src/api/*.rs`, discovered by build.rs: `install_all` and `frame_all`
/// call each file's `install` and `frame` in file-name order. The generated code names
/// `InstallCtx`, `EngineState`, `GpuContext`, `FrameInput` and `RaaError` unqualified;
/// they are imported here.
#[cfg(target_arch = "wasm32")]
pub(crate) mod api {
    use crate::engine::InstallCtx;
    use crate::state::EngineState;
    use raa_core::{GpuContext, RaaError};
    use raa_web::FrameInput;

    include!(concat!(env!("OUT_DIR"), "/api_registry.rs"));
    pub(crate) use m21_views::ViewsState;
}

/// Module init: routes every wasm panic message to `console.error`, followed by the wasm call
/// stack (function names come from the name section the `wasm-release` profile keeps), so a
/// panic names its cause and location instead of surfacing as a bare
/// `RuntimeError: unreachable` (same behaviour as `console_error_panic_hook`, without the
/// extra dependency). Runs once at instantiation.
#[cfg(target_arch = "wasm32")]
mod panic_hook {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console, js_name = error)]
        fn console_error(message: &str);
    }

    #[wasm_bindgen(start)]
    fn install_panic_hook() {
        std::panic::set_hook(Box::new(|info| {
            let error = js_sys::Error::new("raa-js wasm panic");
            let stack = js_sys::Reflect::get(&error, &JsValue::from_str("stack"))
                .ok()
                .and_then(|stack| stack.as_string())
                .unwrap_or_default();
            console_error(&format!("raa-js wasm panic: {info}\n{stack}"));
        }));
    }
}
