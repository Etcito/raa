//! Executor extension `wiring` (S57, module raa-graph-wiring). This share holds the device-free
//! pooling and history functions of `wiring/plan_fns.rs` and the extension skeleton, which applies
//! nothing yet: the graph's behaviour is unchanged by this file.

use crate as raa_graph;
use crate::catalog::Catalog;
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles};
use crate::plan::InputSource;
use crate::plan::{GraphPlan, PlanHistory, PlanNode, PlanPortRef, PlanResource};
use raa_core::RaaError;
use raa_pipeline::{GraphFrame, ImageDesc, ImageDim, PrepareCtx, RecordCtx, ResourceTable};
use std::{cell::RefCell, rc::Rc};

#[allow(dead_code)] // the wiring cards call these; the tests compile the same file for themselves
#[path = "wiring/plan_fns.rs"]
mod plan_fns;

// Re-exported for the wiring cards; nothing in this share calls them yet.
#[allow(unused_imports)]
pub use plan_fns::{history_resets, pool_resources};

/// The extension, discovered by build.rs (stem `wiring`, registered after `views`).
pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(WiringExtension::new())
}

/// Handoff: `WiringExtension` serves 'Wiring 4/5: multi-instance post passes'; today it
/// applies colour bindings and link order using the shared frame state. The upstream skeleton
/// handoff is at `exec_ext/wiring.rs` `WiringExtension` (Wiring 2/5).
pub(crate) struct WiringExtension {
    catalog: Catalog,
    state: Rc<RefCell<State>>,
}

impl WiringExtension {
    /// The extension with its own catalog.
    fn new() -> WiringExtension {
        WiringExtension {
            catalog: Catalog::build(),
            state: Rc::new(RefCell::new(State::default())),
        }
    }
}

impl ExecExtension for WiringExtension {
    fn name(&self) -> &'static str {
        "wiring"
    }

    fn apply(
        &mut self,
        plan: &GraphPlan,
        _previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let resources = plan_fns::chain_resources(plan, &self.catalog);
        let entries = plan_fns::node_entries(plan, &self.catalog);
        let slots = pool_resources(&resources);
        let mut bindings = Vec::new();
        for node in &plan.nodes {
            if !plan_fns::chain_type(&node.type_id) && node.type_id != "pass.tonemap" {
                continue;
            }
            let Some((_, entry)) = entries.iter().find(|(id, _)| *id == node.id) else {
                continue;
            };
            let source =
                node.inputs
                    .iter()
                    .find(|i| i.port == "color")
                    .and_then(|i| match &i.source {
                        InputSource::Link { node, port, .. }
                        | InputSource::Default { node, port } => Some((node, port)),
                        InputSource::Absent => None,
                    });
            if node.type_id != "pass.forward"
                && node.type_id != "pass.tonemap"
                && !resources.iter().any(|r| {
                    source.is_some_and(|(id, port)| {
                        *id == r.producer.node && *port == r.producer.port
                    })
                })
            {
                return Err(vec![issue(
                    "graph.link_type",
                    Some(&node.id),
                    Some("color"),
                    "The colour source must be a chain colour output.".into(),
                )]);
            }
            let source = source.and_then(|(id, port)| {
                entries
                    .iter()
                    .find(|(node, _)| node == id)
                    .map(|(_, entry)| (entry.clone(), port.clone()))
            });
            let dst = resources
                .iter()
                .position(|r| r.producer.node == node.id)
                .map(|i| slots[i] as usize);
            if dst.is_none() && node.type_id != "pass.tonemap" {
                continue;
            }
            bindings.push(Binding {
                entry: entry.clone(),
                source,
                dst,
                forward: node.type_id == "pass.forward",
                tonemap: node.type_id == "pass.tonemap",
            });
        }
        if plan_fns::bound_view_count(plan, &self.catalog) > 32 {
            let node = plan
                .nodes
                .iter()
                .find(|n| n.type_id == "pass.tonemap")
                .map(|n| n.id.as_str());
            return Err(vec![issue(
                "graph.budget",
                node,
                Some("color"),
                "The HDR chain exceeds 32 bound views.".into(),
            )]);
        }
        let mut pipeline = h.pipeline.borrow_mut();
        let order = plan_fns::link_order(plan, &entries, &pipeline.passes());
        let names: Vec<&str> = order.iter().map(String::as_str).collect();
        pipeline
            .set_order_links(&names)
            .map_err(|e| vec![issue("graph.order", None, None, e.message().to_owned())])?;
        let old = pipeline.remove_graph_frame("wiring");
        if let Err(e) = pipeline.add_graph_frame(
            "wiring",
            Box::new(WiringFrame {
                state: self.state.clone(),
                prepared: false,
            }),
        ) {
            if let Some(frame) = old {
                let _ = pipeline.add_graph_frame("wiring", frame);
            }
            return Err(vec![issue(
                "graph.order",
                None,
                None,
                e.message().to_owned(),
            )]);
        }
        let mut state = self.state.borrow_mut();
        state.bindings = bindings;
        state.slot_count = slots.iter().max().map_or(0, |s| *s as usize + 1);
        Ok(())
    }

    fn clear(&mut self, h: &mut FeatureHandles<'_>) {
        h.pipeline.borrow_mut().remove_graph_frame("wiring");
        *self.state.borrow_mut() = State::default();
    }

    fn reorder(&mut self, plan: &GraphPlan, h: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        let entries = plan_fns::node_entries(plan, &self.catalog);
        let mut pipeline = h.pipeline.borrow_mut();
        let order = plan_fns::link_order(plan, &entries, &pipeline.passes());
        let names: Vec<&str> = order.iter().map(String::as_str).collect();
        pipeline.set_order_links(&names)
    }
}

fn issue(code: &'static str, node: Option<&str>, port: Option<&str>, message: String) -> Issue {
    Issue {
        code,
        severity: Severity::Error,
        stage: Stage::Apply,
        node: node.map(str::to_owned),
        port: port.map(str::to_owned),
        param: None,
        link: None,
        message,
        fix: "Choose a compatible colour source and pass order.".into(),
    }
}

struct Binding {
    entry: String,
    source: Option<(String, String)>,
    dst: Option<usize>,
    forward: bool,
    tonemap: bool,
}

#[derive(Default)]
struct State {
    bindings: Vec<Binding>,
    slot_count: usize,
    size: (u32, u32),
    pool: Vec<(wgpu::Texture, wgpu::TextureView)>,
}

struct WiringFrame {
    state: Rc<RefCell<State>>,
    prepared: bool,
}

impl GraphFrame for WiringFrame {
    fn begin(&mut self, _table: &ResourceTable) {
        self.prepared = false;
    }

    fn before_prepare(&mut self, _pass: &str, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        if self.prepared {
            return Ok(());
        }
        self.prepared = true;
        let mut state = self.state.borrow_mut();
        if state.size != ctx.size {
            state.pool.clear();
            state.size = ctx.size;
        }
        while state.pool.len() < state.slot_count {
            let texture = ctx.gpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("raa.wiring.pool"),
                size: wgpu::Extent3d {
                    width: ctx.size.0,
                    height: ctx.size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            state.pool.push((texture, view));
        }
        Ok(())
    }

    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let state = self.state.borrow();
        let Some(binding) = state.bindings.iter().find(|b| b.entry == pass) else {
            return Ok(());
        };
        let dst = binding
            .dst
            .and_then(|i| state.pool.get(i))
            .map(|(_, view)| view);
        let source = binding
            .source
            .as_ref()
            .and_then(|(entry, port)| ctx.resources.get(entry, port));
        let src = if binding.forward {
            dst
        } else {
            source.as_ref().map(|image| &image.view)
        };
        let src = src.ok_or_else(|| {
            RaaError::Resource(format!(
                "{pass}: source {:?} was not published",
                binding.source
            ))
        })?;
        let dst = if binding.tonemap {
            src
        } else {
            dst.ok_or_else(|| {
                RaaError::Resource(format!(
                    "{pass}: no destination for source {:?}",
                    binding.source
                ))
            })?
        };
        let hdr = ctx.targets.hdr();
        hdr.bind(src, dst);
        if !hdr.is_bound() {
            return Err(RaaError::Resource(format!(
                "{pass}: unable to bind source {:?}",
                binding.source
            )));
        }
        Ok(())
    }

    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let state = self.state.borrow();
        let Some(binding) = state.bindings.iter().find(|b| b.entry == pass) else {
            return Ok(());
        };
        ctx.targets.hdr().unbind();
        if !binding.tonemap
            && let Some((_, view)) = binding.dst.and_then(|i| state.pool.get(i))
        {
            ctx.resources.publish(
                pass,
                "color",
                view,
                ImageDesc::new(
                    ImageDim::D2,
                    wgpu::TextureFormat::Rgba16Float,
                    ctx.size.0,
                    ctx.size.1,
                    1,
                    1,
                ),
            );
        }
        Ok(())
    }
}
