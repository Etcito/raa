//! Executor extension `post_stages` (S58, module raa-graph-post-stages): the AO, SSR and bloom
//! stage passes installed from plan nodes. The device-free helpers live in
//! `post_stages/plan.rs`; the frame that binds their inputs is card 2/2's work.
#![allow(dead_code)] // the fields and `frame_added` are read by the extension 2/2 share

use crate::catalog::{Catalog, NodeTypeSpec, ParamBinding, ParamValue};
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles, NodeStats, ResourceStat};
use crate::plan::{GraphPlan, InputSource, PlanNode};
use raa_bloom::stages::{BloomStageHandle, BloomStageKind};
use raa_core::{GpuContext, RaaError};
use raa_gtao::stages::{AoStageHandle, AoStageKind};
use raa_pipeline::{FeaturePass, ImageDesc, Pipeline};
use raa_ssr::stages::{SsrStageHandle, SsrStageKind};
use std::{cell::RefCell, rc::Rc};

#[path = "post_stages/plan.rs"]
mod plan;
// The wiring extension declares `wiring/plan_fns.rs` once; loading it again here would be a
// second module for one file (clippy::duplicate_mod). Reach its items through that declaration.
use crate::registry::exec_wiring::plan_fns;

/// Installed stage passes, and the wiring computed from the applied plan.
/// Handoff: card 'post_stages extension 2/2: graph frame binding inputs, bloom composite chain and
/// view aliases' clones `PostStagesState` through `PostStagesExtension.state`, reads `bindings`,
/// `composites`, `aliases` and the stage handles' `StageHandle::published`, and registers its graph
/// frame while `PostStagesExtension.frame_added` is false. Today `apply` fills these fields and
/// manages the passes; no frame is added by this card.
#[derive(Default)]
pub(crate) struct PostStagesState {
    /// `(node id, entry, handle)` in plan order.
    pub(crate) stages: Vec<(String, String, StageHandle)>,
    /// `(consumer entry, consumer port, source entry, source port)` from `plan::stage_bindings`.
    pub(crate) bindings: Vec<(String, &'static str, String, String)>,
    /// `(bloom.up entry, colour source entry, colour source port)` from `plan::bloom_composites`.
    pub(crate) composites: Vec<(String, String, String)>,
    /// `(entry, type id)` from `plan::view_aliases`.
    pub(crate) aliases: Vec<(String, String)>,
}

/// One installed stage pass, by feature.
#[derive(Clone)]
pub(crate) enum StageHandle {
    Ao(AoStageHandle),
    Ssr(SsrStageHandle),
    Bloom(BloomStageHandle),
}

impl StageHandle {
    fn set_values(&self, values: &[(String, f64)]) -> Result<(), RaaError> {
        match self {
            Self::Ao(handle) => handle.set_values(values),
            Self::Ssr(handle) => handle.set_values(values),
            Self::Bloom(handle) => handle.set_values(values),
        }
    }
    fn ran(&self) -> bool {
        match self {
            Self::Ao(handle) => handle.ran(),
            Self::Ssr(handle) => handle.ran(),
            Self::Bloom(handle) => handle.ran(),
        }
    }
    fn published(&self) -> Vec<(&'static str, ImageDesc)> {
        match self {
            Self::Ao(handle) => handle.published(),
            Self::Ssr(handle) => handle.published(),
            Self::Bloom(handle) => handle.published(),
        }
    }
    fn entry(&self) -> String {
        match self {
            Self::Ao(handle) => handle.entry(),
            Self::Ssr(handle) => handle.entry(),
            Self::Bloom(handle) => handle.entry(),
        }
    }
}

/// The extension, discovered by build.rs as `exec_post_stages`.
pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(PostStagesExtension {
        state: Rc::new(RefCell::new(PostStagesState::default())),
        frame_added: false,
    })
}

struct PostStagesExtension {
    state: Rc<RefCell<PostStagesState>>,
    frame_added: bool,
}

impl ExecExtension for PostStagesExtension {
    fn name(&self) -> &'static str {
        "post_stages"
    }

    fn apply(
        &mut self,
        plan: &GraphPlan,
        _previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let nodes = plan::stage_nodes(plan);
        if let Some((id, _, _)) = nodes.iter().find(|(_, ty, _)| ty.starts_with("ao."))
            && h.handles.gtao.is_none()
        {
            return Err(vec![issue(
                "graph.unavailable",
                Some(id.clone()),
                None,
                "the GTAO feature is not installed",
                "Install the GTAO feature.",
            )]);
        }
        if let Some((id, _, _)) = nodes.iter().find(|(_, ty, _)| ty.starts_with("ssr."))
            && h.handles.ssr.is_none()
        {
            return Err(vec![issue(
                "graph.unavailable",
                Some(id.clone()),
                None,
                "the SSR feature is not installed",
                "Install the SSR feature.",
            )]);
        }
        if nodes.is_empty() && self.state.borrow().stages.is_empty() {
            return Ok(());
        }
        let mut state = std::mem::take(&mut *self.state.borrow_mut());
        let old = std::mem::take(&mut state.stages);
        let mut kept: Vec<(String, Box<dyn FeaturePass>, StageHandle)> = Vec::new();
        for (_, entry, handle) in old {
            if let Ok(pass) = h.pipeline.borrow_mut().remove(&entry) {
                kept.push((entry, pass, handle));
            }
        }
        let catalog = Catalog::build();
        let entries = plan_fns::node_entries(plan, &catalog);
        let mut installed: Vec<(String, StageHandle)> = Vec::new();
        let result: Result<(), Vec<Issue>> = (|| {
            for (id, ty, entry) in &nodes {
                let handle = if let Some(i) = kept
                    .iter()
                    .position(|(kept_entry, _, _)| kept_entry == entry)
                {
                    let (_, pass, handle) = kept.swap_remove(i);
                    h.pipeline
                        .borrow_mut()
                        .add(pass)
                        .map_err(|error| vec![apply_issue(id, &error)])?;
                    handle
                } else {
                    install(h.gpu, h.pipeline, ty, id)
                        .map_err(|error| vec![apply_issue(id, &error)])?
                };
                installed.push((id.clone(), handle));
            }
            let composites = plan::bloom_composites(plan, &entries);
            let karis = plan::karis_nodes(plan);
            for (id, handle) in &installed {
                let node = plan
                    .nodes
                    .iter()
                    .find(|node| node.id == *id)
                    .expect("stage node exists");
                let spec = catalog.node(&node.type_id).expect("stage type exists");
                let mut values = plan::stage_values(node, spec);
                if node.type_id == "bloom.down" {
                    let on = if karis.contains(id) { 1.0 } else { 0.0 };
                    values.push(("karis".to_owned(), on));
                }
                handle
                    .set_values(&values)
                    .map_err(|error| vec![apply_issue(id, &error)])?;
                if node.type_id == "bloom.up"
                    && let StageHandle::Bloom(bloom) = handle
                {
                    let on = composites
                        .iter()
                        .any(|(entry, _, _)| *entry == handle.entry());
                    bloom.set_composite(on);
                }
            }
            Ok(())
        })();
        if let Err(issues) = result {
            for (_, handle) in &installed {
                let _ = h.pipeline.borrow_mut().remove(&handle.entry());
            }
            *self.state.borrow_mut() = PostStagesState::default();
            return Err(issues);
        }
        let stages = installed
            .into_iter()
            .map(|(id, handle)| {
                let entry = handle.entry();
                (id, entry, handle)
            })
            .collect();
        *self.state.borrow_mut() = PostStagesState {
            stages,
            bindings: plan::stage_bindings(plan, &catalog, &entries),
            composites: plan::bloom_composites(plan, &entries),
            aliases: plan::view_aliases(plan),
        };
        Ok(())
    }

    fn frame(&mut self, h: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        let state = self.state.borrow();
        for (_, _, handle) in &state.stages {
            match handle {
                StageHandle::Ao(stage) => {
                    if let Some(camera) = h.handles.gtao.as_ref() {
                        stage.copy_camera_from(camera);
                    }
                }
                StageHandle::Ssr(stage) => {
                    if let Some(camera) = h.handles.ssr.as_ref() {
                        stage.copy_camera_from(camera);
                    }
                }
                StageHandle::Bloom(_) => {}
            }
        }
        Ok(())
    }

    fn clear(&mut self, h: &mut FeatureHandles<'_>) {
        let stages = std::mem::take(&mut self.state.borrow_mut().stages);
        for (_, entry, _) in stages {
            let _ = h.pipeline.borrow_mut().remove(&entry);
        }
        *self.state.borrow_mut() = PostStagesState::default();
    }

    fn stats(&self, out: &mut Vec<NodeStats>) {
        for (id, _, handle) in &self.state.borrow().stages {
            if let Some(stat) = out.iter_mut().find(|stat| stat.node == *id) {
                stat.ran = handle.ran();
                stat.resources = handle
                    .published()
                    .iter()
                    .map(|(port, desc)| image_stat(port, desc))
                    .collect();
            }
        }
    }
}

fn install(
    gpu: &GpuContext,
    pipeline: &Rc<RefCell<Pipeline>>,
    type_id: &str,
    instance: &str,
) -> Result<StageHandle, RaaError> {
    if let Some(kind) = AoStageKind::from_type_id(type_id) {
        Ok(StageHandle::Ao(raa_gtao::stages::install_stage(
            gpu, pipeline, kind, instance,
        )?))
    } else if let Some(kind) = SsrStageKind::from_type_id(type_id) {
        Ok(StageHandle::Ssr(raa_ssr::stages::install_stage(
            gpu, pipeline, kind, instance,
        )?))
    } else if let Some(kind) = BloomStageKind::from_type_id(type_id) {
        Ok(StageHandle::Bloom(raa_bloom::stages::install_stage(
            gpu, pipeline, kind, instance,
        )?))
    } else {
        Err(RaaError::FrameGraph {
            code: "graph.apply",
            message: format!("unknown post stage `{type_id}`"),
        })
    }
}

fn issue(
    code: &'static str,
    node: Option<String>,
    param: Option<String>,
    message: &str,
    fix: &str,
) -> Issue {
    Issue {
        code,
        severity: Severity::Error,
        stage: Stage::Apply,
        node,
        port: None,
        param,
        link: None,
        message: message.into(),
        fix: fix.into(),
    }
}

fn apply_issue(node: &str, error: &RaaError) -> Issue {
    let message = error.to_string();
    let param = message.split(['`', '\'']).nth(1).map(str::to_owned);
    issue(
        "graph.apply",
        Some(node.to_owned()),
        param,
        &message,
        "Use a value accepted by this post stage.",
    )
}

fn image_stat(port: &str, desc: &ImageDesc) -> ResourceStat {
    let aspect = desc
        .format
        .has_depth_aspect()
        .then_some(wgpu::TextureAspect::DepthOnly);
    let texel = desc.format.block_copy_size(aspect).unwrap_or(0);
    let bytes = (0..desc.mips)
        .map(|mip| {
            u64::from(desc.width.checked_shr(mip).unwrap_or(0).max(1))
                * u64::from(desc.height.checked_shr(mip).unwrap_or(0).max(1))
                * u64::from(desc.layers)
                * u64::from(texel)
        })
        .sum();
    ResourceStat {
        port: port.to_owned(),
        width: desc.width,
        height: desc.height,
        layers: desc.layers,
        mips: desc.mips,
        format: format!("{:?}", desc.format).to_lowercase(),
        bytes,
    }
}
