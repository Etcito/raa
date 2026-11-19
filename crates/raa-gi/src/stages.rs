//! The stage API of the GI features (contract 'rg-gi-stage-api'): each GI stage is one
//! [`FeaturePass`] named `<type_id>#<instance>` that [`install_stage`] adds to a pipeline. A
//! [`GiStageHandle`] attaches it to a [`GiHandle`], writes its values and buffer sources and reads
//! what it published. The family cards implement [`StageImpl`] per kind in [`stage_impl`]; until
//! one does, every kind is refused with `gi.invalid`.

// The framework items below have no implementor until the family cards add one.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_lit::LitHandle;
use raa_pipeline::{
    FeaturePass, ImageDesc, Pipeline, PrepareCtx, RecordCtx, ResourceTable, Targets,
};

use crate::error::{GI_INVALID, invalid};
use crate::feature::GiHandle;
use crate::parts::GiShared;
use crate::parts::area_shadow::stage_mask::{AreaMaskStage, AreaSpatialStage};
use crate::parts::area_shadow::stages::{AreaTemporalStage, AreaTraceStage};
use crate::parts::screen::stage_accumulate::AccumulateStage;
use crate::parts::screen::stage_filters::{AtrousStage, ResolveStage};
use crate::parts::screen::stage_restir::{
    RestirInitialStage, RestirSpatialStage, RestirTemporalStage,
};
use crate::parts::screen::stages::{ProbesStage, PyramidStage};
use crate::{GiTunables, tunables};

/// Handoff: raa-graph exec gi_stages 1/3: device-free plan functions. `GiStageKind` is the anchor:
/// its `from_type_id` maps a plan node's type id to its kind, and the free function `stage_entry`
/// gives the pipeline entry `<type_id>#<node id>`. Both are implemented today and exported as
/// `raa_gi::stages`; the plan functions pass them only strings.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GiStageKind {
    Pyramid,
    Probes,
    RestirInitial,
    RestirTemporal,
    RestirSpatial,
    Accumulate,
    Atrous,
    Resolve,
    AreaTrace,
    AreaTemporal,
    AreaSpatial,
    AreaRestirInitial,
    AreaRestirTemporal,
    AreaRestirSpatial,
    AreaMask,
    SsgiTrace,
    SsgiResolve,
}

impl GiStageKind {
    /// Every kind, in declaration order.
    pub const ALL: [GiStageKind; 17] = [
        GiStageKind::Pyramid,
        GiStageKind::Probes,
        GiStageKind::RestirInitial,
        GiStageKind::RestirTemporal,
        GiStageKind::RestirSpatial,
        GiStageKind::Accumulate,
        GiStageKind::Atrous,
        GiStageKind::Resolve,
        GiStageKind::AreaTrace,
        GiStageKind::AreaTemporal,
        GiStageKind::AreaSpatial,
        GiStageKind::AreaRestirInitial,
        GiStageKind::AreaRestirTemporal,
        GiStageKind::AreaRestirSpatial,
        GiStageKind::AreaMask,
        GiStageKind::SsgiTrace,
        GiStageKind::SsgiResolve,
    ];

    /// The plan type id of the kind (`gi.atrous`, `area.mask`, ...).
    pub fn type_id(self) -> &'static str {
        match self {
            GiStageKind::Pyramid => "gi.pyramid",
            GiStageKind::Probes => "gi.probes",
            GiStageKind::RestirInitial => "gi.restir.initial",
            GiStageKind::RestirTemporal => "gi.restir.temporal",
            GiStageKind::RestirSpatial => "gi.restir.spatial",
            GiStageKind::Accumulate => "gi.accumulate",
            GiStageKind::Atrous => "gi.atrous",
            GiStageKind::Resolve => "gi.resolve",
            GiStageKind::AreaTrace => "area.trace",
            GiStageKind::AreaTemporal => "area.temporal",
            GiStageKind::AreaSpatial => "area.spatial",
            GiStageKind::AreaRestirInitial => "area.restir.initial",
            GiStageKind::AreaRestirTemporal => "area.restir.temporal",
            GiStageKind::AreaRestirSpatial => "area.restir.spatial",
            GiStageKind::AreaMask => "area.mask",
            GiStageKind::SsgiTrace => "ssgi.trace",
            GiStageKind::SsgiResolve => "ssgi.resolve",
        }
    }

    /// The kind of a plan type id; None for any other string.
    pub fn from_type_id(id: &str) -> Option<GiStageKind> {
        Self::ALL.iter().copied().find(|kind| kind.type_id() == id)
    }

    /// The pass order of the kind: 42 for the `gi.*` kinds, 43 for `area.*` and 44 for `ssgi.*`.
    pub fn order(self) -> u32 {
        let id = self.type_id();
        if id.starts_with("gi.") {
            42
        } else if id.starts_with("area.") {
            43
        } else {
            44
        }
    }

    /// True for the two kinds that keep their own reservoir history.
    pub fn internal_history(self) -> bool {
        matches!(
            self,
            GiStageKind::RestirTemporal | GiStageKind::AreaRestirTemporal
        )
    }
}

/// The pipeline entry of one stage instance: the kind's type id, `#`, then the instance name.
pub fn stage_entry(kind: GiStageKind, instance: &str) -> String {
    format!("{}#{instance}", kind.type_id())
}

/// One stage kind's maths and passes, built by [`stage_impl`] and driven by the stage pass.
pub(crate) trait StageImpl {
    /// The keys [`GiStageHandle::set_values`] accepts for this stage.
    fn param_keys(&self) -> &'static [&'static str];

    /// Accepts or refuses one value of `key`; the message is the reason.
    fn check_value(&self, key: &str, value: f64) -> Result<(), String>;

    /// Allocates or reallocates the outputs for `ctx.size` and publishes each under
    /// `(cx.entry, output key)`, pushing it onto `published`.
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError>;

    /// Records the dispatches of the frame; true when it recorded any.
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError>;

    /// Drops the state that depends on the frame size (`FeaturePass::resize`).
    fn resize(&mut self) {}

    #[cfg(test)]
    fn force_running_frame_for_test(&mut self, _limit: u32) {}
}

/// The inputs of one stage's frame: its entry, the shared GI state, the attached lit handle, the
/// values with their generation, the internal-history validity and the buffer sources as
/// `(port, entry, source port)`.
pub(crate) struct StageCx<'a> {
    pub(crate) entry: &'a str,
    pub(crate) shared: &'a GiShared,
    pub(crate) lit: &'a LitHandle,
    pub(crate) values: &'a [(String, f64)],
    pub(crate) values_generation: u64,
    pub(crate) history_valid: bool,
    pub(crate) buffer_sources: &'a [(&'static str, String, String)],
}

impl StageCx<'_> {
    /// The stored value of `key`, or `default` when none was set.
    pub(crate) fn value(&self, key: &str, default: f64) -> f64 {
        self.values
            .iter()
            .find(|(k, _)| k.as_str() == key)
            .map_or(default, |(_, value)| *value)
    }

    /// The buffer bound to input `port`: the buffer of its source entry and port.
    pub(crate) fn buffer(
        &self,
        resources: &ResourceTable,
        port: &str,
    ) -> Option<(wgpu::Buffer, u64)> {
        let (_, entry, source) = self.buffer_sources.iter().find(|(p, _, _)| *p == port)?;
        resources.buffer(entry, source)
    }

    /// A store holding every stage value whose key is a tunable of `raa_gi::tunables()`.
    pub(crate) fn tunables(&self) -> GiTunables {
        let mut store = GiTunables::new();
        for (key, value) in self.values {
            if tunables().iter().any(|spec| spec.key == key.as_str()) {
                // `set_values` checked every tunable value with the same store.
                let _ = store.set(key, *value);
            }
        }
        store
    }
}

/// Handoff: raa-gi stages 2/11: gi.pyramid and gi.probes stage passes. `stage_impl` returns None
/// for every kind today, so `install_stage` refuses each kind with `gi.invalid`. Card 2/11 adds
/// the match arms `GiStageKind::Pyramid` and `GiStageKind::Probes`, each returning its own
/// `Box<dyn StageImpl>`. The state it is handed is the key registry `StageImpl::param_keys`
/// (each key's f64 value arrives in `StageCx.values`) and the `StageCx` fields: entry, shared,
/// lit, values, values_generation, history_valid and buffer_sources.
pub(crate) fn stage_impl(kind: GiStageKind) -> Option<Box<dyn StageImpl>> {
    match kind {
        GiStageKind::Pyramid => Some(Box::new(PyramidStage::default())),
        GiStageKind::Probes => Some(Box::new(ProbesStage::default())),
        GiStageKind::Accumulate => Some(Box::new(AccumulateStage::default())),
        GiStageKind::Atrous => Some(Box::new(AtrousStage::default())),
        GiStageKind::Resolve => Some(Box::new(ResolveStage::default())),
        GiStageKind::RestirInitial => Some(Box::new(RestirInitialStage::default())),
        GiStageKind::RestirTemporal => Some(Box::new(RestirTemporalStage::default())),
        GiStageKind::RestirSpatial => Some(Box::new(RestirSpatialStage::default())),
        GiStageKind::AreaTrace => Some(Box::new(AreaTraceStage::default())),
        GiStageKind::AreaTemporal => Some(Box::new(AreaTemporalStage::default())),
        GiStageKind::AreaSpatial => Some(Box::new(AreaSpatialStage::default())),
        GiStageKind::AreaMask => Some(Box::new(AreaMaskStage::default())),
        _ => None,
    }
}

/// The state that the pass and its handle share.
struct StageState {
    kind: GiStageKind,
    entry: String,
    imp: Box<dyn StageImpl>,
    shared: Option<GiShared>,
    lit: Option<LitHandle>,
    values: Vec<(String, f64)>,
    values_generation: u64,
    history_valid: bool,
    buffer_sources: Vec<(&'static str, String, String)>,
    ran: bool,
    published: Vec<(&'static str, ImageDesc)>,
}

/// One installed stage as a pipeline pass: its prepare and record run the stage's own.
struct StagePass {
    name: String,
    kind: GiStageKind,
    state: Rc<RefCell<StageState>>,
}

/// The shared GI state and lit handle of an attached stage, or None while the stage must do
/// nothing: unattached, no frame buffer yet, or an empty viewport.
fn runnable(state: &StageState, size: (u32, u32)) -> Option<(GiShared, LitHandle)> {
    let (shared, lit) = (state.shared.as_ref()?, state.lit.as_ref()?);
    if size.0 == 0 || size.1 == 0 {
        return None;
    }
    shared.frame_buffer()?;
    Some((shared.clone(), lit.clone()))
}

impl FeaturePass for StagePass {
    fn name(&self) -> &str {
        &self.name
    }

    fn order(&self) -> u32 {
        self.kind.order()
    }

    fn resize(&mut self, _gpu: &GpuContext, _targets: &Targets) {
        self.state.borrow_mut().imp.resize();
    }

    /// Clears `published` and `ran`, then runs the stage's prepare while the pass is runnable.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut guard = self.state.borrow_mut();
        let state = &mut *guard;
        state.ran = false;
        state.published.clear();
        let Some((shared, lit)) = runnable(state, ctx.size) else {
            return Ok(());
        };
        let StageState {
            imp,
            entry,
            values,
            values_generation,
            history_valid,
            buffer_sources,
            published,
            ..
        } = state;
        let cx = StageCx {
            entry: entry.as_str(),
            shared: &shared,
            lit: &lit,
            values: values.as_slice(),
            values_generation: *values_generation,
            history_valid: *history_valid,
            buffer_sources: buffer_sources.as_slice(),
        };
        imp.prepare(&cx, ctx, published)
    }

    /// Clears `ran`, then records the stage's dispatches while the pass is runnable.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut guard = self.state.borrow_mut();
        let state = &mut *guard;
        state.ran = false;
        let Some((shared, lit)) = runnable(state, ctx.size) else {
            return Ok(());
        };
        let StageState {
            imp,
            entry,
            values,
            values_generation,
            history_valid,
            buffer_sources,
            ran,
            ..
        } = state;
        let cx = StageCx {
            entry: entry.as_str(),
            shared: &shared,
            lit: &lit,
            values: values.as_slice(),
            values_generation: *values_generation,
            history_valid: *history_valid,
            buffer_sources: buffer_sources.as_slice(),
        };
        *ran = imp.record(&cx, ctx)?;
        Ok(())
    }
}

/// Checks one value of `set_values`: a known key, a finite value, the stage's own check and, for
/// a tunable key, the range of the tunable store.
fn check_value(imp: &dyn StageImpl, key: &str, value: f64) -> Result<(), RaaError> {
    if !imp.param_keys().contains(&key) {
        return Err(invalid(key, "unknown key"));
    }
    if !value.is_finite() {
        return Err(invalid(key, &format!("value {value} is not finite")));
    }
    imp.check_value(key, value)
        .map_err(|reason| invalid(key, &format!("value {value}: {reason}")))?;
    if tunables().iter().any(|spec| spec.key == key) {
        GiTunables::new().set(key, value)?;
    }
    Ok(())
}

/// Installs `kind` as the pass `stage_entry(kind, instance)` of `pipeline`, through `Pipeline::add`
/// (which orders it by `kind.order()`), and returns its handle. A refusal changes nothing:
/// `gi.invalid` when [`stage_impl`] has no implementation of the kind, `pipeline.busy` while the
/// pipeline is borrowed, `pipeline.duplicate_pass` when the name is taken.
pub fn install_stage(
    _gpu: &GpuContext,
    pipeline: &Rc<RefCell<Pipeline>>,
    kind: GiStageKind,
    instance: &str,
) -> Result<GiStageHandle, RaaError> {
    let Some(imp) = stage_impl(kind) else {
        return Err(RaaError::FrameGraph {
            code: GI_INVALID,
            message: format!("stage `{}` is not implemented", kind.type_id()),
        });
    };
    let name = stage_entry(kind, instance);
    let state = Rc::new(RefCell::new(StageState {
        kind,
        entry: name.clone(),
        imp,
        shared: None,
        lit: None,
        values: Vec::new(),
        values_generation: 0,
        history_valid: false,
        buffer_sources: Vec::new(),
        ran: false,
        published: Vec::new(),
    }));
    let mut pipeline = pipeline
        .try_borrow_mut()
        .map_err(|_| RaaError::FrameGraph {
            code: "pipeline.busy",
            message: "the pipeline is borrowed (called while a frame executes)".to_string(),
        })?;
    pipeline.add(Box::new(StagePass {
        name,
        kind,
        state: Rc::clone(&state),
    }))?;
    Ok(GiStageHandle { state })
}

/// A handle on one installed stage. Clones share the state of the pass.
#[derive(Clone)]
pub struct GiStageHandle {
    state: Rc<RefCell<StageState>>,
}

impl GiStageHandle {
    /// The kind of the stage.
    pub fn kind(&self) -> GiStageKind {
        self.state.borrow().kind
    }

    /// The pipeline entry of the stage, `stage_entry(kind, instance)`.
    pub fn entry(&self) -> String {
        self.state.borrow().entry.clone()
    }

    /// Attaches the stage to `gi` and `lit`; the pass prepares and records only while attached.
    pub fn attach(&self, gi: &GiHandle, lit: LitHandle) {
        self.attach_shared(gi.shared(), lit);
    }

    /// Attaches the stage to the shared GI frame state `shared` and `lit` (what [`attach`] does
    /// with the handle's shared state); the crate-private entry the tests of the stage kinds use.
    pub(crate) fn attach_shared(&self, shared: GiShared, lit: LitHandle) {
        let mut state = self.state.borrow_mut();
        state.shared = Some(shared);
        state.lit = Some(lit);
    }

    #[cfg(test)]
    pub(crate) fn force_running_frame_for_test(&self, limit: u32) {
        self.state
            .borrow_mut()
            .imp
            .force_running_frame_for_test(limit);
    }

    /// Writes `values` merged over the stored ones. Atomic: every entry is checked first, and a
    /// refusal (`gi.invalid` naming the key and the value) keeps the previous values. Accepted
    /// values bump the values generation.
    pub fn set_values(&self, values: &[(String, f64)]) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        for (key, value) in values {
            check_value(state.imp.as_ref(), key, *value)?;
        }
        for (key, value) in values {
            match state.values.iter_mut().find(|(k, _)| k == key) {
                Some(slot) => slot.1 = *value,
                None => state.values.push((key.clone(), *value)),
            }
        }
        state.values_generation = state.values_generation.wrapping_add(1);
        Ok(())
    }

    /// Replaces the source of buffer input `port` with `source_port` of `entry`.
    pub fn set_buffer_source(&self, port: &'static str, entry: &str, source_port: &str) {
        let mut state = self.state.borrow_mut();
        state.buffer_sources.retain(|(p, _, _)| *p != port);
        state
            .buffer_sources
            .push((port, entry.to_owned(), source_port.to_owned()));
    }

    /// Removes every buffer source.
    pub fn clear_buffer_sources(&self) {
        self.state.borrow_mut().buffer_sources.clear();
    }

    /// Sets the validity of the internal history. Kinds without one ignore it.
    pub fn set_history_valid(&self, valid: bool) {
        let mut state = self.state.borrow_mut();
        if state.kind.internal_history() {
            state.history_valid = valid;
        }
    }

    /// True when the latest record recorded dispatches.
    pub fn ran(&self) -> bool {
        self.state.borrow().ran
    }

    /// The images the latest prepare published, in output order; buffers are not listed.
    pub fn published(&self) -> Vec<(&'static str, ImageDesc)> {
        self.state.borrow().published.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_core::testing::skip;

    #[test]
    fn kinds_follow_the_contract_table() {
        let ids: [&str; 17] = [
            "gi.pyramid",
            "gi.probes",
            "gi.restir.initial",
            "gi.restir.temporal",
            "gi.restir.spatial",
            "gi.accumulate",
            "gi.atrous",
            "gi.resolve",
            "area.trace",
            "area.temporal",
            "area.spatial",
            "area.restir.initial",
            "area.restir.temporal",
            "area.restir.spatial",
            "area.mask",
            "ssgi.trace",
            "ssgi.resolve",
        ];
        for (kind, id) in GiStageKind::ALL.into_iter().zip(ids) {
            assert_eq!(kind.type_id(), id);
            assert_eq!(GiStageKind::from_type_id(id), Some(kind));
        }
        let orders: Vec<u32> = GiStageKind::ALL.iter().map(|kind| kind.order()).collect();
        assert_eq!(orders.iter().filter(|&&o| o == 42).count(), 8);
        assert_eq!(orders.iter().filter(|&&o| o == 43).count(), 7);
        assert_eq!(orders.iter().filter(|&&o| o == 44).count(), 2);
        assert_eq!(GiStageKind::from_type_id("pass.gi.screen"), None);
        let history: Vec<GiStageKind> = GiStageKind::ALL
            .iter()
            .copied()
            .filter(|kind| kind.internal_history())
            .collect();
        assert_eq!(
            history,
            [GiStageKind::RestirTemporal, GiStageKind::AreaRestirTemporal]
        );
    }

    #[test]
    fn stage_entry_joins_the_type_id_and_the_instance() {
        assert_eq!(
            stage_entry(GiStageKind::Atrous, "giAtrous1"),
            "gi.atrous#giAtrous1"
        );
    }

    #[test]
    fn install_refuses_unimplemented_kinds_and_installs_the_others() {
        let test = "install_refuses_unimplemented_kinds_and_installs_the_others";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        for kind in GiStageKind::ALL {
            let entry = stage_entry(kind, "t");
            let installed = install_stage(&gpu, &pipeline, kind, "t");
            let names: Vec<String> = pipeline
                .borrow()
                .passes()
                .into_iter()
                .map(|info| info.name)
                .collect();
            if stage_impl(kind).is_none() {
                let refused = matches!(&installed, Err(error) if error.code() == GI_INVALID);
                assert!(
                    refused,
                    "{} must be refused with gi.invalid",
                    kind.type_id()
                );
                assert!(!names.contains(&entry));
            } else {
                assert!(installed.is_ok());
                assert!(names.contains(&entry));
            }
        }
    }

    #[test]
    fn shared_frame_validity_follows_its_setter_and_clear() {
        let test = "shared_frame_validity_follows_its_setter_and_clear";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let shared = GiShared::new(&gpu);
        assert!(!shared.history_valid() && !shared.camera_valid());
        shared.set_frame_validity(true, true);
        assert!(shared.history_valid() && shared.camera_valid());
        shared.clear();
        assert!(!shared.history_valid() && !shared.camera_valid());
    }
}
