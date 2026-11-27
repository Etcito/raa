//! RAA patch: render and compute pipelines compiled off the frame path.
//!
//! Chrome compiles a pipeline made by `createRenderPipeline` on the GPU process thread that
//! also runs every later WebGPU command of the page (and the compositor): a graph switch or a
//! new material variant that builds a few dozen pipelines froze the page for seconds. With the
//! mode on ([`set_enabled`]), a pipeline with an explicit layout is made with
//! `createRenderPipelineAsync` / `createComputePipelineAsync` instead (compiled on Dawn's worker
//! threads) and the wgpu handle is returned at once, pending. Until the promise resolves a pass
//! that sets it skips its draws / dispatches (the frame falls back to "that pass did not draw").
//! The host awaits [`stats`]`().pending == 0` behind its loading notice before showing frames.
//!
//! Kept synchronous: auto-layout pipelines (their `getBindGroupLayout` needs the object now),
//! pipelines made inside an error scope (the caller checks creation errors in the scope), and
//! a pending pipeline whose `get_bind_group_layout` is asked (built synchronously then).

use alloc::{format, rc::Rc, string::String};
use core::cell::RefCell;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};

use wasm_bindgen::prelude::*;

use super::webgpu_sys;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(message: &str, error: &JsValue);
    #[wasm_bindgen(js_namespace = console, js_name = info)]
    fn console_info(message: &str);
}

/// Counters of the async pipeline mode, for the host's warm-up and diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AsyncPipelineStats {
    /// Whether new pipelines are created asynchronously.
    pub enabled: bool,
    /// Pipelines whose asynchronous creation has not finished.
    pub pending: u32,
    /// The part of `pending` created outside a frame (graph/feature install): a host waits
    /// for these before rendering, since a pass may use its pipelines only once (a bake).
    pub pending_install: u32,
    /// Pipelines created asynchronously (finished or not).
    pub created_async: u64,
    /// Pipelines created synchronously (mode off, auto layout, error scope).
    pub created_sync: u64,
    /// Pending pipelines that had to be built synchronously (`get_bind_group_layout`).
    pub forced_sync: u64,
    /// Asynchronous creations that failed (logged with `console.error`; the pipeline never draws).
    pub failed: u64,
    /// Draw calls skipped because their pipeline was pending (or failed).
    pub skipped_draws: u64,
    /// Dispatches skipped because their pipeline was pending (or failed).
    pub skipped_dispatches: u64,
    /// Longest asynchronous creation, request to resolution, ms.
    pub max_compile_ms: f64,
}

// Atomics only for `static` (the page is single-threaded; the crate is no_std).
static ENABLED: AtomicBool = AtomicBool::new(false);
static PENDING: AtomicU32 = AtomicU32::new(0);
static PENDING_INSTALL: AtomicU32 = AtomicU32::new(0);
static IN_FRAME: AtomicBool = AtomicBool::new(false);
static CREATED_ASYNC: AtomicU64 = AtomicU64::new(0);
static CREATED_SYNC: AtomicU64 = AtomicU64::new(0);
static FORCED_SYNC: AtomicU64 = AtomicU64::new(0);
static FAILED: AtomicU64 = AtomicU64::new(0);
static SKIPPED_DRAWS: AtomicU64 = AtomicU64::new(0);
static SKIPPED_DISPATCHES: AtomicU64 = AtomicU64::new(0);
/// f64 bits.
static MAX_COMPILE_MS: AtomicU64 = AtomicU64::new(0);

fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Relaxed);
}

/// Turns the asynchronous creation of new pipelines on or off (pipelines already made keep
/// their state). Off by default: every pipeline is usable as soon as it is created.
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Relaxed);
}

/// Marks the pipelines created from now on as made while a frame records (`true`) or
/// outside a frame (`false`, counted in `pending_install`).
pub fn set_in_frame(in_frame: bool) {
    IN_FRAME.store(in_frame, Relaxed);
}

/// Whether new pipelines are created asynchronously.
pub fn enabled() -> bool {
    ENABLED.load(Relaxed)
}

/// The counters (the skip counters and `max_compile_ms` since the last [`reset_counters`]).
pub fn stats() -> AsyncPipelineStats {
    AsyncPipelineStats {
        enabled: enabled(),
        pending: PENDING.load(Relaxed),
        pending_install: PENDING_INSTALL.load(Relaxed),
        created_async: CREATED_ASYNC.load(Relaxed),
        created_sync: CREATED_SYNC.load(Relaxed),
        forced_sync: FORCED_SYNC.load(Relaxed),
        failed: FAILED.load(Relaxed),
        skipped_draws: SKIPPED_DRAWS.load(Relaxed),
        skipped_dispatches: SKIPPED_DISPATCHES.load(Relaxed),
        max_compile_ms: f64::from_bits(MAX_COMPILE_MS.load(Relaxed)),
    }
}

/// Clears `skipped_draws`, `skipped_dispatches` and `max_compile_ms`.
pub fn reset_counters() {
    SKIPPED_DRAWS.store(0, Relaxed);
    SKIPPED_DISPATCHES.store(0, Relaxed);
    MAX_COMPILE_MS.store(0, Relaxed);
    if let Ok(mut labels) = SKIPPED_LABELS.0.try_borrow_mut() {
        labels.clear();
    }
}

pub(crate) fn note_sync() {
    bump(&CREATED_SYNC);
}

pub(crate) fn note_skipped_draw() {
    bump(&SKIPPED_DRAWS);
}

pub(crate) fn note_skipped_dispatch() {
    bump(&SKIPPED_DISPATCHES);
}

fn now_ms() -> f64 {
    js_sys::Date::now()
}

/// The state of one pipeline handle (shared by its clones).
pub(crate) enum SlotState<P, D> {
    Ready(P),
    Pending {
        device: webgpu_sys::GpuDevice,
        desc: D,
        /// Its passes may skip while it compiles: a render pipeline made while a frame
        /// records (a material variant, a lazy per-frame pass). Any other pending pipeline
        /// (compute, or made outside a frame: install, upload) is built synchronously when a
        /// pass sets it, so work that runs once (a bake, a mip chain, a probe init) is never lost.
        skippable: bool,
        label: String,
    },
    Failed {
        device: webgpu_sys::GpuDevice,
        desc: D,
    },
}

impl<P, D> core::fmt::Debug for SlotState<P, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            SlotState::Ready(_) => "Ready",
            SlotState::Pending { .. } => "Pending",
            SlotState::Failed { .. } => "Failed",
        })
    }
}

pub(crate) type Slot<P, D> = Rc<RefCell<SlotState<P, D>>>;

/// A pipeline type the module can build both ways.
pub(crate) trait AsyncPipeline:
    Sized + Clone + JsCast + wasm_bindgen::convert::FromWasmAbi + 'static
{
    type Desc: Clone + 'static;
    /// Render pipelines may be skipped while compiling; compute work is often run once.
    const SKIPPABLE: bool;
    fn create_sync(device: &webgpu_sys::GpuDevice, desc: &Self::Desc) -> Self;
    fn create_async(device: &webgpu_sys::GpuDevice, desc: &Self::Desc) -> js_sys::Promise<Self>;
    fn label(desc: &Self::Desc) -> String;
}

impl AsyncPipeline for webgpu_sys::GpuRenderPipeline {
    type Desc = webgpu_sys::GpuRenderPipelineDescriptor;
    const SKIPPABLE: bool = true;
    fn create_sync(device: &webgpu_sys::GpuDevice, desc: &Self::Desc) -> Self {
        device.create_render_pipeline(desc).unwrap()
    }
    fn create_async(device: &webgpu_sys::GpuDevice, desc: &Self::Desc) -> js_sys::Promise<Self> {
        device.create_render_pipeline_async(desc)
    }
    fn label(desc: &Self::Desc) -> String {
        label_of(desc.as_ref())
    }
}

impl AsyncPipeline for webgpu_sys::GpuComputePipeline {
    type Desc = webgpu_sys::GpuComputePipelineDescriptor;
    const SKIPPABLE: bool = false;
    fn create_sync(device: &webgpu_sys::GpuDevice, desc: &Self::Desc) -> Self {
        device.create_compute_pipeline(desc)
    }
    fn create_async(device: &webgpu_sys::GpuDevice, desc: &Self::Desc) -> js_sys::Promise<Self> {
        device.create_compute_pipeline_async(desc)
    }
    fn label(desc: &Self::Desc) -> String {
        label_of(desc.as_ref())
    }
}

fn label_of(desc: &JsValue) -> String {
    js_sys::Reflect::get(desc, &JsValue::from_str("label"))
        .ok()
        .and_then(|l| l.as_string())
        .unwrap_or_default()
}

/// A slot for a pipeline of `desc`: created asynchronously when the mode is on and `may_defer`
/// (explicit layout, no open error scope), else now.
pub(crate) fn create<P: AsyncPipeline>(
    device: &webgpu_sys::GpuDevice,
    desc: P::Desc,
    may_defer: bool,
) -> Slot<P, P::Desc> {
    if !(may_defer && enabled()) {
        note_sync();
        return Rc::new(RefCell::new(SlotState::Ready(P::create_sync(
            device, &desc,
        ))));
    }
    let promise = P::create_async(device, &desc);
    let label = P::label(&desc);
    let install = !IN_FRAME.load(Relaxed);
    let slot: Slot<P, P::Desc> = Rc::new(RefCell::new(SlotState::Pending {
        device: device.clone(),
        desc,
        skippable: P::SKIPPABLE && !install,
        label: label.clone(),
    }));
    bump(&CREATED_ASYNC);
    PENDING.fetch_add(1, Relaxed);
    if install {
        PENDING_INSTALL.fetch_add(1, Relaxed);
    }
    let weak = Rc::downgrade(&slot);
    let start = now_ms();
    wasm_bindgen_futures::spawn_local(async move {
        let result = wasm_bindgen_futures::JsFuture::from(promise).await;
        let elapsed = now_ms() - start;
        PENDING.fetch_sub(1, Relaxed);
        if install {
            PENDING_INSTALL.fetch_sub(1, Relaxed);
        }
        if elapsed > f64::from_bits(MAX_COMPILE_MS.load(Relaxed)) {
            MAX_COMPILE_MS.store(elapsed.to_bits(), Relaxed);
        }
        if let Err(error) = &result {
            bump(&FAILED);
            console_error(&format!("RAA async pipeline '{label}' failed:"), error);
        }
        // A dropped handle (or one forced synchronously meanwhile) discards the result.
        let Some(slot) = weak.upgrade() else { return };
        let mut state = slot.borrow_mut();
        if let SlotState::Pending { device, desc, .. } = &*state {
            let next = match result {
                Ok(pipeline) => SlotState::Ready(pipeline),
                Err(_) => SlotState::Failed {
                    device: device.clone(),
                    desc: desc.clone(),
                },
            };
            *state = next;
        }
    });
    slot
}

/// The pipeline now, building a pending one synchronously (for `get_bind_group_layout`).
/// A failed asynchronous pipeline is built again synchronously so the caller gets the error
/// where wgpu would have reported it.
pub(crate) fn force<P: AsyncPipeline>(slot: &Slot<P, P::Desc>) -> P {
    let mut state = slot.borrow_mut();
    let pipeline = match &*state {
        SlotState::Ready(p) => return p.clone(),
        SlotState::Pending { device, desc, .. } | SlotState::Failed { device, desc } => {
            P::create_sync(device, desc)
        }
    };
    bump(&FORCED_SYNC);
    *state = SlotState::Ready(pipeline.clone());
    pipeline
}

/// Calls `f` with the pipeline and returns true, building it synchronously first unless it
/// is a skippable pending pipeline: then nothing is called, its label is recorded and the
/// caller skips its draws (false).
pub(crate) fn with_ready<P: AsyncPipeline>(slot: &Slot<P, P::Desc>, f: impl FnOnce(&P)) -> bool {
    {
        let state = slot.borrow();
        match &*state {
            SlotState::Ready(p) => {
                f(p);
                return true;
            }
            SlotState::Pending {
                skippable: true,
                label,
                ..
            } => {
                note_skipped_label(label);
                return false;
            }
            _ => {}
        }
    }
    let pipeline = force(slot);
    f(&pipeline);
    true
}

/// Single-threaded page: a plain cell behind a `Sync` marker for the label list.
struct LabelCell(RefCell<alloc::vec::Vec<String>>);
// SAFETY: the WebGPU backend runs on one thread (wasm32 without shared-memory threads).
unsafe impl Sync for LabelCell {}
static SKIPPED_LABELS: LabelCell = LabelCell(RefCell::new(alloc::vec::Vec::new()));

fn note_skipped_label(label: &str) {
    if let Ok(mut labels) = SKIPPED_LABELS.0.try_borrow_mut() {
        if labels.len() < 64 && !labels.iter().any(|l| l == label) {
            labels.push(String::from(label));
            console_info(&format!(
                "RAA pipeline '{label}' still compiling: its draws are skipped"
            ));
        }
    }
}

/// Labels of the pipelines whose passes skipped work since the last [`reset_counters`].
pub fn skipped_labels() -> alloc::vec::Vec<String> {
    SKIPPED_LABELS
        .0
        .try_borrow()
        .map(|l| l.clone())
        .unwrap_or_default()
}
