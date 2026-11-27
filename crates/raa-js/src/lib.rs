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
        crate::alloc_log::install();
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

/// Allocator failure and foliage stage logging (wasm32).
///
/// [`LoggingAlloc`] is the module's `#[global_allocator]`: it forwards every call unchanged
/// to the wasm allocator (`std::alloc::System`, dlmalloc on wasm32-unknown-unknown) and
/// counts the live allocated bytes. When an allocation returns null (the case that ends in
/// `std::alloc::rust_oom` and a bare `RuntimeError: unreachable`), it first writes one
/// `console.error` line naming the requested byte size and alignment, the live allocated
/// bytes, the wasm memory size in pages and bytes and the last foliage stage reported,
/// followed by the JS call stack. The stack is captured with `Error.stackTraceLimit` raised
/// to [`STACK_FRAMES`] (V8 keeps 10 frames by default, all of them allocator and `std`
/// frames here), so the raa frames above the failing allocation are included. The failure
/// path formats into a stack buffer and builds the stack text on the JS side, so it does
/// not allocate in the exhausted heap; a 256 KiB reserve taken at start is released first
/// for the bindings that may still need memory. The allocation then fails as before.
///
/// [`report_stage`] receives `raa_lit::foliage_bake::report_foliage_stage` and writes one
/// `console.error` line per foliage bake stage with the stage's requested bytes, the live
/// allocated bytes, the wasm memory size and the source triangle count.
#[cfg(target_arch = "wasm32")]
mod alloc_log {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::fmt::Write;
    use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};

    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console, js_name = error)]
        fn console_error(message: &str);
        #[wasm_bindgen(js_namespace = console, js_name = error)]
        fn console_error_value(message: &JsValue);
        #[wasm_bindgen(js_namespace = console, js_name = info)]
        fn console_info(message: &str);
    }

    /// Frames kept in the stack of a failed allocation.
    const STACK_FRAMES: f64 = 256.0;
    /// Bytes reserved at start and released when an allocation fails.
    const RESERVE_BYTES: usize = 256 * 1024;

    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static REPORTING: AtomicBool = AtomicBool::new(false);
    static RESERVE: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());
    static STAGE_PTR: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());
    static STAGE_LEN: AtomicUsize = AtomicUsize::new(0);

    fn reserve_layout() -> Layout {
        Layout::from_size_align(RESERVE_BYTES, 16).unwrap_or(Layout::new::<u8>())
    }

    /// Takes the failure reserve and installs the foliage stage reporter.
    pub(crate) fn install() {
        if RESERVE.load(Ordering::Relaxed).is_null() {
            // SAFETY: a non-zero-sized layout; the block is only ever freed by `release`. (test: system_forwarding_keeps_blocks_and_strings_intact)
            let block = unsafe { System.alloc(reserve_layout()) };
            RESERVE.store(block, Ordering::Relaxed);
        }
        raa_lit::foliage_bake::set_foliage_stage_reporter(report_stage);
    }

    fn release_reserve() {
        let block = RESERVE.swap(std::ptr::null_mut(), Ordering::Relaxed);
        if !block.is_null() {
            // SAFETY: `block` came from `System.alloc(reserve_layout())` in `install`. (test: system_forwarding_keeps_blocks_and_strings_intact)
            unsafe { System.dealloc(block, reserve_layout()) };
        }
    }

    /// `(pages, bytes)` of wasm memory 0.
    fn memory() -> (usize, usize) {
        let pages = core::arch::wasm32::memory_size(0);
        (pages, pages.saturating_mul(65_536))
    }

    fn last_stage() -> &'static str {
        let ptr = STAGE_PTR.load(Ordering::Relaxed);
        if ptr.is_null() {
            return "none";
        }
        let len = STAGE_LEN.load(Ordering::Relaxed);
        // SAFETY: `ptr`/`len` were stored together from a `&'static str` in `report_stage`. (test: system_forwarding_keeps_blocks_and_strings_intact)
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len)) }
    }

    /// A fixed-size message buffer: formatting into it never allocates.
    struct Line {
        bytes: [u8; 512],
        len: usize,
    }

    impl Write for Line {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            let room = self.bytes.len() - self.len;
            let mut take = text.len().min(room);
            while !text.is_char_boundary(take) {
                take -= 1;
            }
            self.bytes[self.len..self.len + take].copy_from_slice(&text.as_bytes()[..take]);
            self.len += take;
            Ok(())
        }
    }

    impl Line {
        fn new() -> Self {
            Line {
                bytes: [0; 512],
                len: 0,
            }
        }

        fn as_str(&self) -> &str {
            std::str::from_utf8(&self.bytes[..self.len]).unwrap_or("raa-js allocation failed")
        }
    }

    /// The current JS stack, captured with `Error.stackTraceLimit` raised to
    /// [`STACK_FRAMES`] and restored afterwards; built on the JS side only.
    fn js_stack() -> JsValue {
        let probe = js_sys::Error::new("raa-js stack");
        let constructor: JsValue = probe.constructor().into();
        let limit = JsValue::from_str("stackTraceLimit");
        let before = js_sys::Reflect::get(&constructor, &limit).unwrap_or(JsValue::UNDEFINED);
        let _ = js_sys::Reflect::set(&constructor, &limit, &JsValue::from_f64(STACK_FRAMES));
        let error = js_sys::Error::new("raa-js allocation failure stack");
        let stack =
            js_sys::Reflect::get(&error, &JsValue::from_str("stack")).unwrap_or(JsValue::UNDEFINED);
        let _ = js_sys::Reflect::set(&constructor, &limit, &before);
        stack
    }

    fn failed(requested: usize, align: usize) {
        if REPORTING.swap(true, Ordering::Relaxed) {
            return;
        }
        release_reserve();
        let (pages, bytes) = memory();
        let mut line = Line::new();
        let _ = writeln!(
            line,
            "raa-js allocation failed: requested {requested} bytes (align {align}), live \
             allocated {} bytes, wasm memory {pages} pages ({bytes} bytes), last foliage \
             stage: {}; JS stack follows",
            LIVE.load(Ordering::Relaxed),
            last_stage()
        );
        let message = js_sys::JsString::from(line.as_str()).concat(&js_stack());
        console_error_value(&message);
        REPORTING.store(false, Ordering::Relaxed);
    }

    /// Writes one foliage stage line to `console.info` and remembers the stage name for a
    /// later allocation failure.
    pub(crate) fn report_stage(stage: &raa_lit::foliage_bake::FoliageStage) {
        STAGE_PTR.store(stage.stage.as_ptr().cast_mut(), Ordering::Relaxed);
        STAGE_LEN.store(stage.stage.len(), Ordering::Relaxed);
        let (pages, bytes) = memory();
        let mut line = Line::new();
        let _ = write!(
            line,
            "raa-js foliage stage '{}': requested {} bytes, live allocated {} bytes, wasm \
             memory {pages} pages ({bytes} bytes), source triangles {}",
            stage.stage,
            stage.requested_bytes,
            LIVE.load(Ordering::Relaxed),
            stage.source_triangles
        );
        // A stage line is progress, not a failure: console.error is read as a page error by the scenarios.
        console_info(line.as_str());
    }

    /// The `#[global_allocator]` of raa-js on wasm32: `System` plus live byte counting and
    /// the failure log.
    pub(crate) struct LoggingAlloc;

    // SAFETY: every call forwards its layout and pointer unchanged to `System`; the counters and the failure log only observe sizes. (test: system_forwarding_keeps_blocks_and_strings_intact)
    unsafe impl GlobalAlloc for LoggingAlloc {
        // SAFETY: forwards the caller's arguments unchanged to `System`, whose contract the trait's own contract matches (test: system_forwarding_keeps_blocks_and_strings_intact)
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: forwards the caller's layout unchanged. (test: system_forwarding_keeps_blocks_and_strings_intact)
            let ptr = unsafe { System.alloc(layout) };
            if ptr.is_null() {
                failed(layout.size(), layout.align());
            } else {
                LIVE.fetch_add(layout.size(), Ordering::Relaxed);
            }
            ptr
        }

        // SAFETY: forwards the caller's arguments unchanged to `System`, whose contract the trait's own contract matches (test: system_forwarding_keeps_blocks_and_strings_intact)
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // SAFETY: forwards the caller's layout unchanged. (test: system_forwarding_keeps_blocks_and_strings_intact)
            let ptr = unsafe { System.alloc_zeroed(layout) };
            if ptr.is_null() {
                failed(layout.size(), layout.align());
            } else {
                LIVE.fetch_add(layout.size(), Ordering::Relaxed);
            }
            ptr
        }

        // SAFETY: forwards the caller's arguments unchanged to `System`, whose contract the trait's own contract matches (test: system_forwarding_keeps_blocks_and_strings_intact)
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: forwards the caller's pointer and layout unchanged. (test: system_forwarding_keeps_blocks_and_strings_intact)
            unsafe { System.dealloc(ptr, layout) };
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        }

        // SAFETY: forwards the caller's arguments unchanged to `System`, whose contract the trait's own contract matches (test: system_forwarding_keeps_blocks_and_strings_intact)
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // SAFETY: forwards the caller's pointer, layout and size unchanged. (test: system_forwarding_keeps_blocks_and_strings_intact)
            let grown = unsafe { System.realloc(ptr, layout, new_size) };
            if grown.is_null() {
                failed(new_size, layout.align());
            } else {
                LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
                LIVE.fetch_add(new_size, Ordering::Relaxed);
            }
            grown
        }
    }

    #[global_allocator]
    static GLOBAL: LoggingAlloc = LoggingAlloc;
}

#[cfg(test)]
mod safety_tests {
    use std::alloc::{GlobalAlloc, Layout, System};

    /// The forwarding the wasm32 `LoggingAlloc` and the stage name rely on: a block from `System`
    /// keeps its bytes through `realloc` and is released by `dealloc` with the layout it has, and a
    /// `&'static str` rebuilt from its pointer and length is the same string.
    #[test]
    fn system_forwarding_keeps_blocks_and_strings_intact() {
        let layout = Layout::from_size_align(64, 16).expect("layout");
        // SAFETY: a non-zero-sized layout, freed below with the layout realloc returned it under (test: system_forwarding_keeps_blocks_and_strings_intact)
        let block = unsafe { System.alloc_zeroed(layout) };
        assert!(!block.is_null());
        // SAFETY: `block` holds 64 initialized zero bytes (test: system_forwarding_keeps_blocks_and_strings_intact)
        unsafe { block.write_bytes(7, 64) };
        // SAFETY: `block` came from `System` under `layout`; 256 is a non-zero size (test: system_forwarding_keeps_blocks_and_strings_intact)
        let grown = unsafe { System.realloc(block, layout, 256) };
        assert!(!grown.is_null());
        // SAFETY: the first 64 bytes were copied by realloc and are initialized (test: system_forwarding_keeps_blocks_and_strings_intact)
        let kept = unsafe { std::slice::from_raw_parts(grown, 64) };
        assert!(kept.iter().all(|&b| b == 7));
        let grown_layout = Layout::from_size_align(256, 16).expect("layout");
        // SAFETY: `grown` came from `System.realloc` with size 256 and align 16 (test: system_forwarding_keeps_blocks_and_strings_intact)
        unsafe { System.dealloc(grown, grown_layout) };
        let stage: &'static str = "source snapshot";
        let (ptr, len) = (stage.as_ptr(), stage.len());
        // SAFETY: `ptr`/`len` come from the same `&'static str` (test: system_forwarding_keeps_blocks_and_strings_intact)
        let back = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len)) };
        assert_eq!(back, stage);
    }
}
