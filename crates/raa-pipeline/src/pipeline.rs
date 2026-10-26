//! The configurable pass pipeline (S12): pass and contributor entries, enable/disable and
//! order overrides with their refusals, and the one `raa.pipeline` frame graph pass that
//! prepares and records the enabled feature passes in order.

use std::cell::{Cell, RefCell, RefMut};
use std::rc::Rc;

use raa_core::{
    BACKBUFFER, FrameGraph, GpuContext, PassBuilder, PrepareContext, RaaError, RecordContext,
    RenderPass,
};

use crate::pass::{FeaturePass, PassInfo, PassRole, PrepareCtx, RecordCtx};
use crate::{LightingResources, Targets};

/// Name of the frame graph pass `Pipeline::install` adds.
const PASS_NAME: &str = "raa.pipeline";
/// Host of every contributor entry: contributors draw inside the forward render pass (S12).
const HOST: &str = "forward";
/// Passes that keep this relative order; the last of them also runs last.
const REQUIRED_ORDER: [&str; 3] = ["prepass", "forward", "tonemap"];

fn refusal(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

/// Refuses a sequence where `prepass`, `forward` and `tonemap` (those present) are out of
/// order or an HDR pass runs after `tonemap` (`pipeline.required_order`).
fn check_required_order(names: &[&str]) -> Result<(), RaaError> {
    let mut previous: Option<(&str, usize)> = None;
    for required in REQUIRED_ORDER {
        let Some(at) = names.iter().position(|n| *n == required) else {
            continue;
        };
        if let Some((before, before_at)) = previous
            && at < before_at
        {
            return Err(refusal(
                "pipeline.required_order",
                format!(
                    "`{required}` must run after `{before}` (order: {})",
                    names.join(", ")
                ),
            ));
        }
        previous = Some((required, at));
    }
    let last = REQUIRED_ORDER[REQUIRED_ORDER.len() - 1];
    if let Some(at) = names.iter().position(|n| *n == last)
        && let Some(after) = names.get(at + 1..)
        && !after.is_empty()
    {
        return Err(refusal(
            "pipeline.required_order",
            format!(
                "`{last}` must run last, but {} would run after it",
                after.join(", ")
            ),
        ));
    }
    Ok(())
}

/// Tone mapping ends the HDR sequence; display overlays follow that sequence.
fn check_overlay_order(names: &[&str], overlays: &[&str]) -> Result<(), RaaError> {
    let regular: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| !overlays.contains(n))
        .collect();
    check_required_order(&regular)?;
    let first_overlay = names.iter().position(|n| overlays.contains(n));
    if let Some(at) = first_overlay
        && names[at..].iter().any(|n| !overlays.contains(n))
    {
        return Err(refusal(
            "pipeline.required_order",
            "display overlays must follow all HDR passes and tonemap".into(),
        ));
    }
    Ok(())
}

/// A feature pass and its configuration.
struct PassEntry {
    name: String,
    order: u32,
    required: bool,
    enabled: bool,
    pass: Box<dyn FeaturePass>,
    /// Targets generation the pass was last resized for (0 = never resized).
    sized_for: u64,
}

impl PassEntry {
    fn info(&self) -> PassInfo {
        PassInfo {
            name: self.name.clone(),
            role: PassRole::Pass,
            enabled: self.enabled,
            order: self.order,
            required: self.required,
        }
    }
}

/// A contributor drawn by the forward pass; the pipeline lists it and toggles its flag.
struct ContributorEntry {
    name: String,
    order: u32,
    enabled: Rc<Cell<bool>>,
}

impl ContributorEntry {
    fn info(&self) -> PassInfo {
        PassInfo {
            name: self.name.clone(),
            role: PassRole::Contributor,
            enabled: self.enabled.get(),
            order: self.order,
            required: false,
        }
    }
}

/// Device-free configuration core of [`Pipeline`]: the entries, their enabled flags and the
/// execution order, with every configuration refusal. `Pipeline` delegates to it and adds
/// the GPU side (lighting resources, targets, the frame graph pass); it never calls the
/// passes' GPU methods, so it is usable (and tested) without a device.
#[derive(Default)]
pub struct PipelineConfig {
    /// Feature passes in execution order.
    passes: Vec<PassEntry>,
    /// Contributor entries sorted by order (stable for equal orders).
    contributors: Vec<ContributorEntry>,
}

impl PipelineConfig {
    /// An empty configuration.
    pub fn new() -> PipelineConfig {
        PipelineConfig::default()
    }

    fn contains(&self, name: &str) -> bool {
        self.passes.iter().any(|p| p.name == name)
            || self.contributors.iter().any(|c| c.name == name)
    }

    fn entry_names(&self) -> String {
        let names: Vec<&str> = self
            .passes
            .iter()
            .map(|p| p.name.as_str())
            .chain(self.contributors.iter().map(|c| c.name.as_str()))
            .collect();
        if names.is_empty() {
            "none".to_string()
        } else {
            names.join(", ")
        }
    }

    fn duplicate(name: &str) -> RaaError {
        refusal(
            "pipeline.duplicate_pass",
            format!("an entry named `{name}` is already in the pipeline"),
        )
    }

    /// Adds an enabled pass before the first pass of greater order in the current execution
    /// order (after equal orders). Refuses a name already used by a pass or a contributor
    /// (`pipeline.duplicate_pass`) and a position breaking prepass < forward < tonemap with
    /// tonemap last among HDR passes (`pipeline.required_order`); output overlays follow it.
    pub fn add(&mut self, pass: Box<dyn FeaturePass>) -> Result<(), RaaError> {
        let name = pass.name().to_owned();
        if self.contains(&name) {
            return Err(Self::duplicate(&name));
        }
        let order = pass.order();
        let at = self
            .passes
            .iter()
            .position(|p| p.order > order)
            .unwrap_or(self.passes.len());
        let mut names: Vec<&str> = self.passes.iter().map(|p| p.name.as_str()).collect();
        names.insert(at, &name);
        let mut overlays: Vec<&str> = self
            .passes
            .iter()
            .filter(|p| p.pass.output_overlay())
            .map(|p| p.name.as_str())
            .collect();
        if pass.output_overlay() {
            overlays.push(&name);
        }
        check_overlay_order(&names, &overlays)?;
        let required = pass.required();
        self.passes.insert(
            at,
            PassEntry {
                name,
                order,
                required,
                enabled: true,
                pass,
                sized_for: 0,
            },
        );
        Ok(())
    }

    /// Adds a contributor entry of the forward pass at `order` inside it; `enabled` is the
    /// flag the contributor's owner reads (shared, so `passes` reports its current value).
    /// Refuses a name already used (`pipeline.duplicate_pass`).
    pub fn add_contributor_entry(
        &mut self,
        name: &str,
        order: u32,
        enabled: Rc<Cell<bool>>,
    ) -> Result<(), RaaError> {
        if self.contains(name) {
            return Err(Self::duplicate(name));
        }
        let at = self
            .contributors
            .iter()
            .position(|c| c.order > order)
            .unwrap_or(self.contributors.len());
        self.contributors.insert(
            at,
            ContributorEntry {
                name: name.to_owned(),
                order,
                enabled,
            },
        );
        Ok(())
    }

    /// Every entry in execution order: the passes, with the contributors (by order) right
    /// after their host `forward`, or at the end while no `forward` pass is added.
    pub fn passes(&self) -> Vec<PassInfo> {
        let mut out = Vec::with_capacity(self.passes.len() + self.contributors.len());
        let mut listed = false;
        for pass in &self.passes {
            out.push(pass.info());
            if pass.name == HOST {
                out.extend(self.contributors.iter().map(ContributorEntry::info));
                listed = true;
            }
        }
        if !listed {
            out.extend(self.contributors.iter().map(ContributorEntry::info));
        }
        out
    }

    /// Enables or disables the entry `name`. For a pass, returns it so the caller hands it
    /// the lighting resources (`FeaturePass::set_enabled`); for a contributor, sets its flag
    /// and returns `None`. Refuses an unknown name (`pipeline.unknown_pass`) and disabling a
    /// required pass (`pipeline.required_pass`).
    pub fn set_enabled(
        &mut self,
        name: &str,
        enabled: bool,
    ) -> Result<Option<&mut dyn FeaturePass>, RaaError> {
        if let Some(contributor) = self.contributors.iter().find(|c| c.name == name) {
            contributor.enabled.set(enabled);
            return Ok(None);
        }
        let Some(at) = self.passes.iter().position(|p| p.name == name) else {
            let known = self.entry_names();
            return Err(refusal(
                "pipeline.unknown_pass",
                format!("no pass or contributor named `{name}` (entries: {known})"),
            ));
        };
        // invariant: `at` comes from position over self.passes
        let entry = &mut self.passes[at];
        if entry.required && !enabled {
            return Err(refusal(
                "pipeline.required_pass",
                format!("`{name}` is a required pass and cannot be disabled"),
            ));
        }
        entry.enabled = enabled;
        Ok(Some(entry.pass.as_mut()))
    }

    /// Replaces the execution order of the passes. `names` lists every pass (not the
    /// contributors, which keep their order inside their host) exactly once. Refusals, in
    /// order: an unknown or contributor name (`pipeline.unknown_pass`), a name listed twice
    /// (`pipeline.duplicate_pass`), a missing pass (`pipeline.order_incomplete`), prepass,
    /// forward and tonemap out of order, or an HDR pass after tonemap (`pipeline.required_order`).
    /// A refused order leaves the current one unchanged.
    pub fn set_order(&mut self, names: &[&str]) -> Result<(), RaaError> {
        for (i, name) in names.iter().enumerate() {
            if self.contributors.iter().any(|c| c.name == *name) {
                return Err(refusal(
                    "pipeline.unknown_pass",
                    format!(
                        "`{name}` is a contributor of `{HOST}`, not a pass: it keeps its order \
                         inside its host"
                    ),
                ));
            }
            if !self.passes.iter().any(|p| p.name == *name) {
                let known = self.entry_names();
                return Err(refusal(
                    "pipeline.unknown_pass",
                    format!("no pass named `{name}` (entries: {known})"),
                ));
            }
            if names.iter().take(i).any(|n| n == name) {
                return Err(refusal(
                    "pipeline.duplicate_pass",
                    format!("`{name}` is listed more than once in the order"),
                ));
            }
        }
        if names.len() != self.passes.len() {
            let missing: Vec<&str> = self
                .passes
                .iter()
                .map(|p| p.name.as_str())
                .filter(|p| !names.contains(p))
                .collect();
            return Err(refusal(
                "pipeline.order_incomplete",
                format!(
                    "the order must list every pass exactly once; missing: {}",
                    missing.join(", ")
                ),
            ));
        }
        let overlays: Vec<&str> = self
            .passes
            .iter()
            .filter(|p| p.pass.output_overlay())
            .map(|p| p.name.as_str())
            .collect();
        check_overlay_order(names, &overlays)?;
        // every pass name is in `names` exactly once (checked above)
        self.passes
            .sort_unstable_by_key(|p| names.iter().position(|n| *n == p.name));
        Ok(())
    }
}

/// Frame time from `set_time`: the latest `now_ms` and what is derived from it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Clock {
    last_ms: Option<f64>,
    time_s: f32,
    dt_s: f32,
}

impl Clock {
    /// Stores `now_ms`; `dt_s` is the time since the previous call (0 on the first call and
    /// when time goes backwards). A non-finite `now_ms` is ignored.
    fn set(&mut self, now_ms: f64) {
        if !now_ms.is_finite() {
            return;
        }
        let dt_ms = self.last_ms.map_or(0.0, |last| (now_ms - last).max(0.0));
        self.last_ms = Some(now_ms);
        self.time_s = (now_ms / 1000.0) as f32;
        self.dt_s = (dt_ms / 1000.0) as f32;
    }
}

/// The lit renderer's pass pipeline: feature passes and contributor entries (configured
/// through [`PipelineConfig`]), the group-1 lighting resources, the surface-sized targets and
/// the frame time. `install` runs it as the one `raa.pipeline` frame graph pass.
pub struct Pipeline {
    config: PipelineConfig,
    lighting: LightingResources,
    /// Created at the first prepare, recreated when the size changes.
    targets: Option<Targets>,
    /// Bumped each time the targets are (re)created; passes resized for an older generation
    /// get `resize` before their next prepare.
    targets_generation: u64,
    surface_format: wgpu::TextureFormat,
    surface_is_srgb: bool,
    clock: Clock,
}

impl Pipeline {
    /// An empty pipeline for a surface of `surface_format` (`surface_is_srgb` when the format
    /// is *Srgb), with dummy lighting resources. The targets are created at the first prepare.
    pub fn new(
        gpu: &GpuContext,
        surface_format: wgpu::TextureFormat,
        surface_is_srgb: bool,
    ) -> Pipeline {
        Pipeline {
            config: PipelineConfig::new(),
            lighting: LightingResources::new(gpu),
            targets: None,
            targets_generation: 0,
            surface_format,
            surface_is_srgb,
            clock: Clock::default(),
        }
    }

    /// Adds an enabled feature pass by its order (see [`PipelineConfig::add`]).
    pub fn add(&mut self, pass: Box<dyn FeaturePass>) -> Result<(), RaaError> {
        self.config.add(pass)
    }

    /// Adds a contributor entry of the forward pass (see
    /// [`PipelineConfig::add_contributor_entry`]).
    pub fn add_contributor_entry(
        &mut self,
        name: &str,
        order: u32,
        enabled: Rc<Cell<bool>>,
    ) -> Result<(), RaaError> {
        self.config.add_contributor_entry(name, order, enabled)
    }

    /// Every pass and contributor entry in execution order.
    pub fn passes(&self) -> Vec<PassInfo> {
        self.config.passes()
    }

    /// Enables or disables an entry; a pass also gets `set_enabled(enabled, lighting)` so a
    /// disabled feature returns its group-1 resources to the dummies. Refusals:
    /// `pipeline.unknown_pass`, `pipeline.required_pass`.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> Result<(), RaaError> {
        if let Some(pass) = self.config.set_enabled(name, enabled)? {
            pass.set_enabled(enabled, &mut self.lighting);
        }
        Ok(())
    }

    /// Overrides the execution order of the passes (see [`PipelineConfig::set_order`]).
    pub fn set_order(&mut self, names: &[&str]) -> Result<(), RaaError> {
        self.config.set_order(names)
    }

    /// Sets the frame time in milliseconds: `time_s = now_ms / 1000`, `dt_s` = seconds since
    /// the previous call (0 on the first call and when time goes backwards). A non-finite
    /// value is ignored.
    pub fn set_time(&mut self, now_ms: f64) {
        self.clock.set(now_ms);
    }

    /// The group-1 lighting resources.
    pub fn lighting(&self) -> &LightingResources {
        &self.lighting
    }

    /// Mutable lighting resources (uploaded at the next prepare).
    pub fn lighting_mut(&mut self) -> &mut LightingResources {
        &mut self.lighting
    }

    /// The surface-sized targets; `None` before the first prepare.
    pub fn targets(&self) -> Option<&Targets> {
        self.targets.as_ref()
    }

    /// The surface format given to `new`.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }

    /// Whether the surface format is *Srgb (given to `new`).
    pub fn surface_is_srgb(&self) -> bool {
        self.surface_is_srgb
    }

    /// Adds the RenderPass `raa.pipeline` to `graph`: it writes BACKBUFFER; its prepare
    /// ensures the targets, resets the HDR chain, uploads the lighting resources and prepares
    /// every enabled pass in order; its record records every enabled pass in order with the
    /// backbuffer view and format.
    pub fn install(this: &Rc<RefCell<Pipeline>>, graph: &mut FrameGraph) {
        graph.add_pass(Box::new(PipelinePass {
            pipeline: Rc::clone(this),
        }));
    }

    fn prepare_frame(&mut self, ctx: &PrepareContext<'_>) -> Result<(), RaaError> {
        let gpu = ctx.gpu;
        let targets = match &mut self.targets {
            Some(targets) => {
                if targets.ensure(gpu, ctx.size) {
                    self.targets_generation += 1;
                }
                &*targets
            }
            None => {
                self.targets_generation += 1;
                &*self.targets.insert(Targets::new(gpu, ctx.size))
            }
        };
        targets.hdr().reset();
        self.lighting.upload(gpu);

        let generation = self.targets_generation;
        let mut prepare = PrepareCtx {
            gpu,
            size: ctx.size,
            frame: ctx.frame,
            time_s: self.clock.time_s,
            dt_s: self.clock.dt_s,
            targets,
            lighting: &mut self.lighting,
        };
        for entry in self.config.passes.iter_mut().filter(|e| e.enabled) {
            if entry.sized_for != generation {
                entry.pass.resize(gpu, targets);
                entry.sized_for = generation;
            }
            entry.pass.prepare(&mut prepare)?;
        }
        // resources a pass set in its prepare are bound in this frame's records
        self.lighting.upload(gpu);
        Ok(())
    }

    fn record_frame(&mut self, ctx: &mut RecordContext<'_>) -> Result<(), RaaError> {
        let backbuffer = ctx.view(BACKBUFFER)?;
        let backbuffer_format = ctx.format(BACKBUFFER);
        let Some(targets) = self.targets.as_ref() else {
            return Err(RaaError::Resource(format!(
                "{PASS_NAME}: record ran before a prepare created the targets"
            )));
        };
        let mut record = RecordCtx {
            gpu: ctx.gpu,
            encoder: &mut *ctx.encoder,
            size: ctx.size,
            frame: ctx.frame,
            time_s: self.clock.time_s,
            targets,
            lighting: &self.lighting,
            backbuffer,
            backbuffer_format,
            surface_is_srgb: self.surface_is_srgb,
        };
        for entry in self.config.passes.iter_mut().filter(|e| e.enabled) {
            entry.pass.record(&mut record)?;
        }
        Ok(())
    }
}

/// The frame graph pass `Pipeline::install` adds.
struct PipelinePass {
    pipeline: Rc<RefCell<Pipeline>>,
}

impl PipelinePass {
    fn pipeline(&self) -> Result<RefMut<'_, Pipeline>, RaaError> {
        self.pipeline.try_borrow_mut().map_err(|_| {
            RaaError::Resource(format!(
                "{PASS_NAME}: the pipeline is borrowed elsewhere while the frame runs"
            ))
        })
    }
}

impl RenderPass for PipelinePass {
    fn name(&self) -> &str {
        PASS_NAME
    }

    fn declare(&self, builder: &mut PassBuilder) {
        builder.write(BACKBUFFER);
    }

    fn prepare(&mut self, ctx: &PrepareContext<'_>) -> Result<(), RaaError> {
        self.pipeline()?.prepare_frame(ctx)
    }

    fn record(&mut self, ctx: &mut RecordContext<'_>) -> Result<(), RaaError> {
        self.pipeline()?.record_frame(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_time_has_zero_delta() {
        let mut clock = Clock::default();
        clock.set(1500.0);
        assert_eq!((clock.time_s, clock.dt_s), (1.5, 0.0));
    }

    #[test]
    fn delta_is_the_time_since_the_previous_call() {
        let mut clock = Clock::default();
        clock.set(1000.0);
        clock.set(1016.0);
        assert_eq!(clock.time_s, 1.016);
        assert!((clock.dt_s - 0.016).abs() < 1e-6, "dt {}", clock.dt_s);
        clock.set(1016.0);
        assert_eq!(clock.dt_s, 0.0, "a repeated fixed time has no delta");
    }

    #[test]
    fn time_going_backwards_has_zero_delta() {
        let mut clock = Clock::default();
        clock.set(2000.0);
        clock.set(1000.0);
        assert_eq!((clock.time_s, clock.dt_s), (1.0, 0.0));
    }

    #[test]
    fn non_finite_time_is_ignored() {
        let mut clock = Clock::default();
        clock.set(500.0);
        let before = clock;
        clock.set(f64::NAN);
        clock.set(f64::INFINITY);
        assert_eq!(clock, before);
    }

    #[test]
    fn required_order_accepts_partial_sets_and_refuses_inversions() {
        assert!(check_required_order(&[]).is_ok());
        assert!(check_required_order(&["a", "prepass", "b", "forward", "tonemap"]).is_ok());
        assert!(check_required_order(&["forward", "x"]).is_ok());
        for names in [
            &["forward", "prepass"][..],
            &["tonemap", "forward"][..],
            &["prepass", "tonemap", "bloom"][..],
        ] {
            let err = check_required_order(names).expect_err("refused");
            assert_eq!(err.code(), "pipeline.required_order", "{names:?}");
            assert!(!err.message().is_empty());
        }
    }
    #[test]
    fn display_overlays_follow_tonemap_but_hdr_passes_cannot() {
        assert!(
            check_overlay_order(
                &["prepass", "forward", "tonemap", "selection"],
                &["selection"]
            )
            .is_ok()
        );
        assert!(
            check_overlay_order(
                &["prepass", "forward", "selection", "tonemap"],
                &["selection"]
            )
            .is_err()
        );
        assert!(check_overlay_order(&["prepass", "forward", "tonemap", "bloom"], &[]).is_err());
        assert!(check_overlay_order(&["forward", "selection", "bloom"], &["selection"]).is_err());
    }
}
