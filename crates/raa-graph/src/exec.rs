//! Executor types of the render graph (S53): the feature handles and the settings snapshot the
//! executor works on, the pipeline instructions of a plan, the extension trait and the stats.
//! `GraphExecutor` lives in `exec/executor.rs`.

mod executor;
mod export;
mod settings;
mod settings_post;

pub use executor::{
    GraphExecutor, adopt_pipeline, external_changes, pipeline_differs, settings_targets,
};
pub use export::export_snapshot;
pub use settings::{
    AppliedDescs, default_values, merge_values, read_values, setting_keys, snapshot_keys,
    write_values,
};

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{PassInfo, PassRole, Pipeline};

use crate::catalog::{Catalog, FeatureId, ParamValue};
use crate::error::Issue;
use crate::plan::GraphPlan;

/// The live handle of each feature the graph configures, `None` while the feature is not
/// installed. Each field is the handle type that its api file inserts into the Engine's feature
/// map (`crates/raa-js/src/api/`: m07, m09, m10, m11, m14, m27, m29, m30, m31). Area shading has
/// no field: it goes through [`FeatureHandles::lit`]. The sky handle is read only by the
/// tunables extension, and Sky is never an owned setter (scene content, S54).
#[derive(Default)]
pub struct HandleSet {
    pub tonemap: Option<raa_tonemap::TonemapControl>,
    pub shadow: Option<raa_shadow::ShadowHandle>,
    pub gtao: Option<raa_gtao::GtaoHandle>,
    pub contact: Option<raa_contact::ContactHandle>,
    pub bloom: Option<raa_bloom::BloomHandle>,
    pub oit: Option<raa_oit::OitHandle>,
    pub ssr: Option<raa_ssr::SsrHandle>,
    pub fog: Option<raa_fog::FogHandle>,
    pub shafts: Option<raa_shafts::ShaftsHandle>,
    pub dof: Option<raa_dof::DofHandle>,
    pub grading: Option<raa_grading::GradingHandle>,
    pub cel: Option<raa_cel::CelHandle>,
    pub vg: Option<raa_vg::VgHandle>,
    pub gi: Option<raa_gi::GiHandle>,
    pub taa: Option<raa_taa::TaaHandle>,
    pub sky: Option<raa_sky::SkyHandle>,
}

/// What one executor call works with: the device, the pipeline, the handles, the Engine's view
/// and sample counts, and the lit renderer. Area shadows and area shading go through `lit`
/// because `LitHandle` exposes neither (`LitRenderer::area_shadows`, `frame_feature(39)`).
pub struct FeatureHandles<'a> {
    pub gpu: &'a GpuContext,
    pub pipeline: &'a Rc<RefCell<Pipeline>>,
    pub handles: &'a mut HandleSet,
    pub viewports: u32,
    pub msaa: u32,
    pub lit: Option<&'a mut raa_lit::LitRenderer>,
}

/// The configuration a graph overrides: the enabled entries, the pass order and every Setting
/// value by feature. The snapshot taken on first activation is what `clear_graph` restores.
#[derive(Clone, Debug, PartialEq)]
pub struct SettingsSnapshot {
    /// `(entry, enabled)` for every installed entry, in installed order.
    pub passes: Vec<(String, bool)>,
    /// The pass order, contributors excluded.
    pub order: Vec<String>,
    /// The Setting values of each feature.
    pub values: Vec<(FeatureId, Vec<(String, ParamValue)>)>,
}

/// The pipeline calls one plan needs, as computed by [`pipeline_instructions`].
pub struct PipelineInstructions {
    /// `(entry, enabled)` for every installed entry, in installed order.
    pub enable: Vec<(String, bool)>,
    /// Every installed pass exactly once, in run order. Contributors are not listed.
    pub order: Vec<String>,
}

/// The pipeline instructions of `plan` (S53 level 1). An installed entry is enabled when a plan
/// node drives it: a node drives the entry named by its catalog `pass`. `order` lists the planned
/// passes in plan order, then every installed pass the plan leaves out, each inserted before the
/// first planned pass of greater default order (`PassInfo::order`), or last when there is none.
pub fn pipeline_instructions(
    plan: &GraphPlan,
    catalog: &Catalog,
    installed: &[PassInfo],
) -> PipelineInstructions {
    let mut planned: Vec<&str> = Vec::new();
    for node in &plan.nodes {
        let Some(entry) = catalog.node(&node.type_id).and_then(|spec| spec.pass) else {
            continue;
        };
        if !planned.contains(&entry) {
            planned.push(entry);
        }
    }
    let mut enable = Vec::new();
    for info in installed.iter().filter(|info| !info.name.contains('#')) {
        if !enable.iter().any(|(name, _)| *name == info.name) {
            enable.push((info.name.clone(), planned.contains(&info.name.as_str())));
        }
    }

    let is_pass = |info: &&PassInfo| info.role == PassRole::Pass && !info.name.contains('#');
    let mut order: Vec<&PassInfo> = planned
        .iter()
        .filter_map(|entry| {
            installed
                .iter()
                .filter(is_pass)
                .find(|info| info.name == *entry)
        })
        .collect();
    for info in installed.iter().filter(is_pass) {
        if order.iter().any(|entry| entry.name == info.name) {
            continue;
        }
        let at = order
            .iter()
            .position(|p| planned.contains(&p.name.as_str()) && p.order > info.order)
            .unwrap_or(order.len());
        order.insert(at, info);
    }
    PipelineInstructions {
        enable,
        order: order.into_iter().map(|info| info.name.clone()).collect(),
    }
}

/// One executor extension. Each `exec_ext/*.rs` file registers one through `extension()`, and
/// the executor applies every extension after the built-in features, in registry order.
pub trait ExecExtension {
    /// The extension's name, for diagnostics.
    fn name(&self) -> &'static str;
    /// Applies `plan`; `previous` is the plan it replaces, if any. A refusal returns its issues.
    fn apply(
        &mut self,
        plan: &GraphPlan,
        previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>>;
    /// Clears what `apply` set.
    fn clear(&mut self, h: &mut FeatureHandles<'_>);
    /// Restores extension-specific pass order after an external pipeline edit.
    fn reorder(&mut self, _plan: &GraphPlan, _h: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        Ok(())
    }
    /// Runs once per frame while a graph is active.
    fn frame(&mut self, _h: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        Ok(())
    }
    /// Appends the statistics of the nodes this extension runs.
    fn stats(&self, _out: &mut Vec<NodeStats>) {}
}

/// The outcome of a graph change. `ok` false means nothing changed; `issues` says why.
pub struct SetResult {
    pub ok: bool,
    pub issues: Vec<Issue>,
    /// Changed parameters whose catalog `rebuild` is not `none`.
    pub rebuilt: u32,
}

/// The statistics of one node of the active graph's latest frame.
pub struct NodeStats {
    pub node: String,
    pub type_id: String,
    /// Whether the node's pipeline entry ran this frame.
    pub ran: bool,
    pub resources: Vec<ResourceStat>,
}

/// The size, format and memory of one resource a node produced.
pub struct ResourceStat {
    pub port: String,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub mips: u32,
    pub format: String,
    pub bytes: u64,
}
