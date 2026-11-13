//! Typed sky lifecycle diagnostics (S25 / P2-SKY-OBSERVATIONS).
//!
//! [`SkyDiagnostics`] is the observation the renderer records for the last invocation: the
//! invocation identities, the weather/atmosphere counter generations, the environment
//! publication copied from `raa_ibl::IblHandle`, the per-view records and the readiness of
//! the physical-sky pipeline. The API layer serializes these exact records; nothing here
//! advances a clock, mutates a resource or becomes an authored setting.
//!
//! Ownership: an accepted invocation and its submitted generations are recorded only from
//! actual pipeline submissions. Recording or allocation alone leaves a view pending, and a
//! failure keeps its originating message until a new preparation or submission succeeds.

use raa_ibl::IblPublication;

/// The sky renderer's lifecycle observation.
#[derive(Clone, Debug, PartialEq)]
pub struct SkyDiagnostics {
    /// Invocation whose recording was accepted, if any.
    pub invocation: Option<u64>,
    /// Last invocation with at least one actual submitted view.
    pub submitted_invocation: Option<u64>,
    /// Counts accepted effective weather changes and explicit cycle restarts.
    pub weather_generation: u64,
    /// Counts acknowledged atmosphere (Milky Way / LUT) work.
    pub atmosphere_generation: u64,
    /// The IBL environment publication, copied from its handle.
    pub environment: IblPublication,
    /// Per-view records, indexed by view slot; `None` for an absent view.
    pub views: [Option<SkyViewDiagnostics>; 4],
    /// Whether the M1 sky pipeline completed a real submission.
    pub readiness: SkyReadiness,
}

impl Default for SkyDiagnostics {
    fn default() -> Self {
        SkyDiagnostics {
            invocation: None,
            submitted_invocation: None,
            weather_generation: 0,
            atmosphere_generation: 0,
            environment: IblPublication {
                requested_generation: 0,
                published_generation: None,
                pending: false,
            },
            views: [None; 4],
            readiness: SkyReadiness::default(),
        }
    }
}

/// The screen-cloud accumulation of one view, as the last recorded frame saw it. Read-only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CloudAccumulationDiagnostics {
    /// The committed accumulation count n (0 before any march is committed).
    pub count: u32,
    /// The flag the freeze decision used: the sky was animated (wind, evolution, weather or
    /// time changing).
    pub animated: bool,
    /// The last frame did not march (skipped, or only the cheap refresh pass of a frozen view).
    pub frozen: bool,
    /// Whether the last frame could reproject from the committed history.
    pub history_used: bool,
    /// Frames that ran the in-frame convergence burst of the screen march.
    pub burst_frames: u32,
    /// The world-space cloud dome (shared by every view).
    pub dome: CloudDomeDiagnostics,
}

/// The cloud dome accumulation, as the last recorded frame saw it. Read-only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CloudDomeDiagnostics {
    /// `WeatherSettings.cloud_dome` was on for the last recorded frame of the view.
    pub active: bool,
    /// The committed accumulation count of the dome (0 before any march is committed).
    pub count: u32,
    /// The last recorded frame did not march the dome (frozen).
    pub frozen: bool,
    /// Frames that ran the in-frame convergence burst of the dome march.
    pub burst_frames: u32,
}

/// One prepared view's lifecycle observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkyViewDiagnostics {
    /// The view's real index (`0..4`).
    pub view: u32,
    /// Physical viewport width in pixels.
    pub width: u32,
    /// Physical viewport height in pixels.
    pub height: u32,
    /// Invocation whose work this view recorded, if any.
    pub invocation: Option<u64>,
    /// Last invocation actually submitted for this view.
    pub submitted_invocation: Option<u64>,
    /// Counts acknowledged sky-view LUT work for this view.
    pub sky_view_generation: u64,
    /// Whether the view's cloud history may be reprojected.
    pub history_valid: bool,
    /// Invalidation bits accepted for this view.
    pub reset_reasons: CloudResetReasons,
    /// True only when the backend's screen result holds completed volume work.
    pub cloud_screen_valid: bool,
    /// True only when the backend's panorama holds completed volume work.
    pub cloud_panorama_valid: bool,
    /// The view's screen-cloud accumulation state.
    pub cloud: CloudAccumulationDiagnostics,
}

/// Whether the sky pipeline is ready, pending or failed, with its stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkyReadiness {
    /// Lifecycle status.
    pub status: SkyReadinessStatus,
    /// Stage the status refers to.
    pub stage: SkyReadinessStage,
    /// The originating failure message while `status` is `Failed`.
    pub error: Option<String>,
}

impl Default for SkyReadiness {
    fn default() -> Self {
        SkyReadiness {
            status: SkyReadinessStatus::Pending,
            stage: SkyReadinessStage::Install,
            error: None,
        }
    }
}

impl SkyReadiness {
    /// A pending readiness observation at `stage`.
    pub fn pending(stage: SkyReadinessStage) -> Self {
        SkyReadiness {
            status: SkyReadinessStatus::Pending,
            stage,
            error: None,
        }
    }

    /// A failed readiness observation retaining `message`.
    pub fn failed(stage: SkyReadinessStage, message: impl Into<String>) -> Self {
        SkyReadiness {
            status: SkyReadinessStatus::Failed,
            stage,
            error: Some(message.into()),
        }
    }
}

/// Readiness status; the JS boundary serializes `pending`/`ready`/`failed`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkyReadinessStatus {
    /// Work is recorded, allocated or waiting for submission.
    Pending,
    /// A real complete submission acknowledged all required M1 work.
    Ready,
    /// A preparation or recording failed at a named stage.
    Failed,
}

impl SkyReadinessStatus {
    /// The exact JS status string.
    pub fn as_str(self) -> &'static str {
        match self {
            SkyReadinessStatus::Pending => "pending",
            SkyReadinessStatus::Ready => "ready",
            SkyReadinessStatus::Failed => "failed",
        }
    }
}

/// The stage a readiness observation refers to; strings are camelCase at the JS boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkyReadinessStage {
    /// Feature installation and resource creation.
    Install,
    /// Atmosphere LUT and directional-transmission preparation.
    Atmosphere,
    /// The one-time Milky Way bake.
    MilkyWay,
    /// The cloud lighting (panorama/transmittance) stage.
    CloudLighting,
    /// The cloud screen-march stage.
    CloudScreen,
    /// IBL environment publication.
    Environment,
    /// Invocation submission acknowledgement.
    Submission,
}

impl SkyReadinessStage {
    /// The exact JS stage string.
    pub fn as_str(self) -> &'static str {
        match self {
            SkyReadinessStage::Install => "install",
            SkyReadinessStage::Atmosphere => "atmosphere",
            SkyReadinessStage::MilkyWay => "milkyWay",
            SkyReadinessStage::CloudLighting => "cloudLighting",
            SkyReadinessStage::CloudScreen => "cloudScreen",
            SkyReadinessStage::Environment => "environment",
            SkyReadinessStage::Submission => "submission",
        }
    }
}

/// Cloud history reset reasons (S25 `CloudInvalidation` set, ordered bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CloudResetReasons(pub u8);

impl CloudResetReasons {
    /// No reason.
    pub const NONE: CloudResetReasons = CloudResetReasons(0);
    /// The view was resized.
    pub const RESIZE: CloudResetReasons = CloudResetReasons(1);
    /// The camera jumped.
    pub const CAMERA_DISCONTINUITY: CloudResetReasons = CloudResetReasons(2);
    /// The weather changed.
    pub const WEATHER: CloudResetReasons = CloudResetReasons(4);
    /// The sky model switched.
    pub const MODEL: CloudResetReasons = CloudResetReasons(8);
    /// The cloud feature was enabled after being disabled.
    pub const ENABLE: CloudResetReasons = CloudResetReasons(16);
    /// The view was removed.
    pub const VIEW_REMOVED: CloudResetReasons = CloudResetReasons(32);
    /// Every valid bit.
    pub const ALL: u8 = 63;

    /// `Some` when `bits` holds only the six declared reasons.
    pub fn from_bits(bits: u8) -> Option<Self> {
        (bits & !Self::ALL == 0).then_some(CloudResetReasons(bits))
    }

    /// True when every bit of `other` is set.
    pub fn contains(self, other: CloudResetReasons) -> bool {
        self.0 & other.0 == other.0
    }

    /// Adds `other`'s bits.
    pub fn union(self, other: CloudResetReasons) -> Self {
        CloudResetReasons(self.0 | other.0)
    }

    /// The capitalized reason strings of the set bits, in bit order.
    pub fn names(self) -> Vec<&'static str> {
        const NAMES: [(&str, u8); 6] = [
            ("Resize", 1),
            ("CameraDiscontinuity", 2),
            ("Weather", 4),
            ("Model", 8),
            ("Enable", 16),
            ("ViewRemoved", 32),
        ];
        NAMES
            .into_iter()
            .filter(|(_, bit)| self.0 & bit != 0)
            .map(|(name, _)| name)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_pending_install_with_no_views() {
        let d = SkyDiagnostics::default();
        assert_eq!(d.invocation, None);
        assert_eq!(d.submitted_invocation, None);
        assert_eq!((d.weather_generation, d.atmosphere_generation), (0, 0));
        assert_eq!(
            d.environment,
            IblPublication {
                requested_generation: 0,
                published_generation: None,
                pending: false,
            }
        );
        assert_eq!(d.views, [None; 4]);
        assert_eq!(d.readiness.status, SkyReadinessStatus::Pending);
        assert_eq!(d.readiness.stage, SkyReadinessStage::Install);
        assert_eq!(d.readiness.error, None);
    }

    #[test]
    fn status_and_stage_strings_are_the_js_boundary() {
        let statuses = [
            (SkyReadinessStatus::Pending, "pending"),
            (SkyReadinessStatus::Ready, "ready"),
            (SkyReadinessStatus::Failed, "failed"),
        ];
        for (status, text) in statuses {
            assert_eq!(status.as_str(), text);
        }
        let stages = [
            (SkyReadinessStage::Install, "install"),
            (SkyReadinessStage::Atmosphere, "atmosphere"),
            (SkyReadinessStage::MilkyWay, "milkyWay"),
            (SkyReadinessStage::CloudLighting, "cloudLighting"),
            (SkyReadinessStage::CloudScreen, "cloudScreen"),
            (SkyReadinessStage::Environment, "environment"),
            (SkyReadinessStage::Submission, "submission"),
        ];
        for (stage, text) in stages {
            assert_eq!(stage.as_str(), text);
        }
    }

    #[test]
    fn reset_reasons_serialize_in_bit_order_and_refuse_unknown_bits() {
        assert_eq!(CloudResetReasons::NONE.names(), Vec::<&str>::new());
        let all = CloudResetReasons(CloudResetReasons::ALL);
        assert_eq!(
            all.names(),
            vec![
                "Resize",
                "CameraDiscontinuity",
                "Weather",
                "Model",
                "Enable",
                "ViewRemoved",
            ]
        );
        let mixed = CloudResetReasons::VIEW_REMOVED
            .union(CloudResetReasons::RESIZE)
            .union(CloudResetReasons::WEATHER);
        assert_eq!(mixed.names(), vec!["Resize", "Weather", "ViewRemoved"]);
        assert!(CloudResetReasons::from_bits(64).is_none());
        assert!(CloudResetReasons::from_bits(63).is_some());
        assert!(all.contains(CloudResetReasons::ENABLE));
        assert!(!CloudResetReasons::RESIZE.contains(CloudResetReasons::MODEL));
    }

    #[test]
    fn failed_readiness_retains_its_message() {
        let failed = SkyReadiness::failed(SkyReadinessStage::CloudScreen, "gpu.device_lost");
        assert_eq!(failed.status, SkyReadinessStatus::Failed);
        assert_eq!(failed.stage, SkyReadinessStage::CloudScreen);
        assert_eq!(failed.error.as_deref(), Some("gpu.device_lost"));
        assert_eq!(
            SkyReadiness::pending(SkyReadinessStage::Atmosphere).status,
            SkyReadinessStatus::Pending
        );
    }
}
