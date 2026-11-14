//! The GI feature handle (S36, S37): the activity rule [`inactive_reason`], the crate-private
//! state [`GiCore`] and the cloneable [`GiHandle`] that shares it with the Engine.
//!
//! This share holds the state and its accessors. `GiHandle::prepare_frame`, the passes and the
//! install are added by the next share, so until a frame runs the handle reports the initial
//! statistics of `stats`.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_math::Mat4;
use raa_pipeline::Pipeline;
use raa_scene::{Camera, DirectionalLight};
use raa_vg::scene::ItemScratch;

use crate::clipmap::{RebuildRounds, RelightCursor};
use crate::debug::DebugRenderer;
use crate::membership::{InstanceGi, MembershipReport};
use crate::parts::{GiPart, GiShared, VoxelDriver};
use crate::settings::{
    FINEST_VOXEL_METRES, GiQuality, GiQualityPatch, GiSettings, GiSettingsPatch,
};
use crate::stats::GiStats;
use crate::volume::{GiVolume, VolumeWork};

/// The S36 rank table of `inactive_reason`: the first match wins.
pub fn inactive_reason(
    enabled: bool,
    volume_pass_enabled: bool,
    submitted_views: u8,
    members: u32,
) -> Option<&'static str> {
    if !enabled {
        Some("disabled")
    } else if !volume_pass_enabled {
        Some("passDisabled")
    } else if submitted_views != 1 {
        Some("multiView")
    } else if members == 0 {
        Some("noInstance")
    } else {
        None
    }
}

/// The statistics before any frame: inactive for `disabled`, at the given quality.
fn initial_stats(quality: GiQuality) -> GiStats {
    GiStats {
        active: false,
        inactive_reason: Some("disabled"),
        quality,
        finest_voxel_metres: FINEST_VOXEL_METRES,
        ..GiStats::default()
    }
}

/// Handoff: `GiCore` is the state that card "raa-gi: feature.rs share 2, passes, install and
/// prepare_frame" takes over. Today `GiCore::new` builds it with the defaults of contract
/// 'gi-core-state' and has no caller, and `GiHandle` reads and writes its settings, quality,
/// requests, view, sun, statistics and recorded passes; no frame runs yet. That card constructs
/// `GiCore` through `GiCore::new` (the install context it reuses is `parts.rs:50 PartsCtx::new`),
/// adds `GiFeature::install` and `GiHandle::prepare_frame`, and adds the two passes over the
/// fields `volume`, `rounds`, `cursor`, `work`, `prepared`, `published` and `pending`.
#[allow(dead_code)]
pub(crate) struct GiCore {
    settings: GiSettings,
    quality: GiQuality,
    allocated_quality: Option<GiQuality>,
    requests: BTreeMap<u32, bool>,
    camera: Camera,
    previous_view_proj: Option<Mat4>,
    viewport: (u32, u32),
    submitted_views: u8,
    sun_direction: [f32; 3],
    sun_color: [f32; 3],
    local_light_count: u32,
    history_valid: bool,
    /// Frames prepared so far; 0 before the first frame.
    frame_index: u64,
    shared: GiShared,
    volume: Option<GiVolume>,
    rounds: RebuildRounds,
    cursor: RelightCursor,
    work: Option<VolumeWork>,
    drivers: Vec<Rc<dyn VoxelDriver>>,
    parts: Vec<Rc<dyn GiPart>>,
    debug: DebugRenderer,
    scratch: ItemScratch,
    report: MembershipReport,
    signature: Option<u64>,
    frames_since_rebuild: u32,
    prepared: GiStats,
    published: Option<GiStats>,
    pending: bool,
    recorded: Vec<&'static str>,
    pipeline: Rc<RefCell<Pipeline>>,
}

#[allow(dead_code)]
impl GiCore {
    /// The state at construction: default settings and quality, no volume and no frame, one
    /// submitted view, the sun straight up, and `prepared` holding the initial statistics.
    pub(crate) fn new(
        _gpu: &GpuContext,
        pipeline: Rc<RefCell<Pipeline>>,
        shared: GiShared,
        drivers: Vec<Rc<dyn VoxelDriver>>,
        parts: Vec<Rc<dyn GiPart>>,
        debug: DebugRenderer,
    ) -> GiCore {
        GiCore {
            settings: GiSettings::default(),
            quality: GiQuality::default(),
            allocated_quality: None,
            requests: BTreeMap::new(),
            camera: Camera::default(),
            previous_view_proj: None,
            viewport: (0, 0),
            submitted_views: 1,
            sun_direction: [0.0, 1.0, 0.0],
            sun_color: [0.0; 3],
            local_light_count: 0,
            history_valid: false,
            frame_index: 0,
            shared,
            volume: None,
            rounds: RebuildRounds::new(5),
            cursor: RelightCursor::default(),
            work: None,
            drivers,
            parts,
            debug,
            scratch: ItemScratch::default(),
            report: MembershipReport::default(),
            signature: None,
            frames_since_rebuild: 0,
            prepared: initial_stats(GiQuality::default()),
            published: None,
            pending: false,
            recorded: Vec::new(),
            pipeline,
        }
    }
}

/// The cloneable handle of the GI feature: settings, quality, instance requests, the view and
/// sun, and the published statistics. Clones share one [`GiCore`].
#[derive(Clone)]
pub struct GiHandle {
    core: Rc<RefCell<GiCore>>,
}

impl GiHandle {
    /// Applies `patch` to the settings and publishes them to the shared frame state. A refusal
    /// changes nothing.
    pub fn set(&self, patch: GiSettingsPatch) -> Result<(), RaaError> {
        let mut core = self.core.borrow_mut();
        let next = patch.apply(&core.settings)?;
        core.settings = next;
        core.shared.set_settings(next);
        Ok(())
    }

    /// The current settings.
    pub fn get(&self) -> GiSettings {
        self.core.borrow().settings
    }

    /// Applies `patch` to the quality and publishes it. A change takes effect at the next active
    /// frame, which reallocates the volume. A refusal changes nothing.
    pub fn set_quality(&self, patch: GiQualityPatch) -> Result<(), RaaError> {
        let mut core = self.core.borrow_mut();
        let next = patch.apply(&core.quality)?;
        core.quality = next;
        core.shared.set_quality(next);
        Ok(())
    }

    /// The current quality.
    pub fn quality(&self) -> GiQuality {
        self.core.borrow().quality
    }

    /// Stores the request of `tag`; `None` removes it, so the group takes the default membership.
    pub fn set_instance(&self, tag: u32, request: Option<bool>) {
        let mut core = self.core.borrow_mut();
        match request {
            Some(value) => {
                core.requests.insert(tag, value);
            }
            None => {
                core.requests.remove(&tag);
            }
        }
    }

    /// Removes the request and the latest membership entry of `tag`.
    pub fn forget_instance(&self, tag: u32) {
        let mut core = self.core.borrow_mut();
        core.requests.remove(&tag);
        core.report.per_tag.remove(&tag);
    }

    /// The request of `tag` and its membership in the latest prepared frame. A tag with no entry,
    /// including every tag before a frame, is not voxelized and has no driver, code or message.
    pub fn instance(&self, tag: u32) -> InstanceGi {
        let core = self.core.borrow();
        let latest = core.report.per_tag.get(&tag).cloned().unwrap_or_default();
        InstanceGi {
            requested: core.requests.get(&tag).copied(),
            voxelized: latest.voxelized,
            driver: latest.driver,
            code: latest.code,
            message: latest.message,
        }
    }

    /// Stores the camera with its aspect set from `viewport`, the viewport and the number of
    /// submitted views.
    pub fn set_view(&self, camera: &Camera, viewport: (u32, u32), submitted_views: u8) {
        let mut core = self.core.borrow_mut();
        core.camera = *camera;
        core.camera.set_viewport(viewport.0, viewport.1);
        core.viewport = viewport;
        core.submitted_views = submitted_views;
    }

    /// Stores the sun: the direction toward it is `-light.direction` normalised, and the colour
    /// is the linear colour times the intensity.
    pub fn set_sun(&self, light: &DirectionalLight) {
        let mut core = self.core.borrow_mut();
        let toward = (-light.direction).normalize_or_zero();
        core.sun_direction = [toward.x, toward.y, toward.z];
        core.sun_color = light.color.map(|channel| channel * light.intensity);
    }

    /// Stores the number of active local lights.
    pub fn set_local_light_count(&self, count: u32) {
        self.core.borrow_mut().local_light_count = count;
    }

    /// Marks the frame history invalid and drops the history of every part.
    pub fn invalidate_history(&self) {
        let mut core = self.core.borrow_mut();
        core.history_valid = false;
        for part in &core.parts {
            part.invalidate_history();
        }
    }

    /// Records the outcome of submission `invocation`: nothing is pending any more, and a
    /// complete submission publishes the prepared statistics with that frame index.
    pub fn observe_submission(&self, invocation: u64, complete: bool) {
        let mut core = self.core.borrow_mut();
        core.pending = false;
        if complete {
            let mut published = core.prepared.clone();
            published.stats_frame_index = Some(invocation);
            core.published = Some(published);
        }
    }

    /// The published statistics, or the prepared ones before the first complete submission, with
    /// `pending` from the core. Before any frame, the initial statistics at the current quality.
    pub fn stats(&self) -> GiStats {
        let core = self.core.borrow();
        let mut stats = match &core.published {
            Some(published) => published.clone(),
            None if core.frame_index == 0 => initial_stats(core.quality),
            None => GiStats {
                stats_frame_index: None,
                ..core.prepared.clone()
            },
        };
        stats.pending = core.pending;
        stats
    }

    /// The GPU bytes the feature owns: `allocated_bytes` of [`GiHandle::stats`].
    pub fn allocated_bytes(&self) -> u64 {
        self.stats().allocated_bytes
    }

    /// The pass names recorded by the latest frame, in record order.
    pub fn recorded_passes(&self) -> Vec<&'static str> {
        self.core.borrow().recorded.clone()
    }
}
