//! The milestone-2 part `slots` of the GI volume (S39 slot providers and the raa-ibl diffuse
//! scale): the `Indirect`, `Ao` and `LocalVisibility` providers of provider `gi`, installed by
//! the generated registry in `parts.rs`, and the part that scales the image-based lighting
//! while GI is active. `slot_indirect.wgsl` is adapted from solstice@7e465b5
//! `crates/solstice-render/shaders/lighting/deferred.wgsl:173-175` and keeps the MIT notice.

use std::borrow::Cow;
use std::cell::Cell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_ibl::IblHandle;
use raa_shader::{GI_VOLUME_WGSL, SlotId, SlotProvider};

use super::{GiPart, PartFrame, PartsCtx};

/// The provider name of the three S39 providers: `gi`.
pub const GI_PROVIDER: &str = "gi";

const SLOT_INDIRECT_WGSL: &str = include_str!("../wgsl/slots/slot_indirect.wgsl");
const SLOT_AO_WGSL: &str = include_str!("../wgsl/slots/slot_ao.wgsl");
const SLOT_LOCAL_VISIBILITY_WGSL: &str = include_str!("../wgsl/slots/slot_local_visibility.wgsl");

/// The `Indirect` slot provider `gi` (priority 10): `raa_slot_indirect` adds the screen-space GI
/// irradiance `(1 - s.metallic) * s.base_color / PI * g.rgb * raa_gi_volume_params.intensity *
/// min(s.ao, c.ao)`, with `g` the clamped `textureLoad` of `raa_gi_screen` at `c.frag`.
///
/// Handoff: `indirect_provider` is one of the three providers that card 'raa-gi screen:
/// screen_variants.rs (modules, layouts, gi compositions)' composes with [`ao_provider`] and
/// [`local_visibility_provider`] into the Forward, Transparent, FurShell and vg.shade variants
/// and validates. Today the three providers return the S39 slot texts with slots
/// `Indirect`/`Ao`/`LocalVisibility`, priorities 10/20/10 and provider `gi`, and `install` adds
/// them to the lit renderer plus [`SlotsPart`], which drives `IblHandle::set_diffuse_scale`
/// (crates/raa-ibl/src/feature.rs:205, whose own Handoff names this card) with
/// `environment_diffuse` while GI is active and 1 otherwise, only on change.
pub fn indirect_provider() -> SlotProvider {
    SlotProvider {
        slot: SlotId::Indirect,
        priority: 10,
        provider: GI_PROVIDER,
        wgsl: Cow::Borrowed(SLOT_INDIRECT_WGSL),
    }
}

/// The `Ao` slot provider `gi` (priority 20): `raa_slot_ao` returns the gtao texture times the
/// GI screen occlusion, so it equals gtao while GI is inactive (the 1x1 dummy alpha is 1).
pub fn ao_provider() -> SlotProvider {
    SlotProvider {
        slot: SlotId::Ao,
        priority: 20,
        provider: GI_PROVIDER,
        wgsl: Cow::Borrowed(SLOT_AO_WGSL),
    }
}

/// The `LocalVisibility` slot provider `gi` (priority 10): `raa_slot_local_visibility` traces
/// the GI volume toward the light while feature 30 is enabled and returns visibility in [0, 1].
/// Its text is the only one with `GI_VOLUME_WGSL`, the `fn raa_gi_trace` definition included.
pub fn local_visibility_provider() -> SlotProvider {
    SlotProvider {
        slot: SlotId::LocalVisibility,
        priority: 10,
        provider: GI_PROVIDER,
        wgsl: Cow::Owned(format!("{GI_VOLUME_WGSL}{SLOT_LOCAL_VISIBILITY_WGSL}")),
    }
}

/// Installs the three providers and the [`SlotsPart`] into the GI feature (`S39`). The refusals
/// of `LitRenderer::add_slot` propagate unchanged (`shader.slot_conflict` on an equal priority).
/// `LitRenderer` has no slot removal, so a refusal on the second or third provider leaves the
/// earlier ones installed; a failed install thus keeps its first providers.
pub(crate) fn install(ctx: &mut PartsCtx<'_>) -> Result<(), RaaError> {
    ctx.lit.add_slot(indirect_provider())?;
    ctx.lit.add_slot(ao_provider())?;
    ctx.lit.add_slot(local_visibility_provider())?;
    let part = SlotsPart {
        ibl: ctx.ibl.clone(),
        applied: Cell::new(None),
    };
    ctx.add_part(Rc::new(part));
    Ok(())
}

/// The GI part `slots`: applies the S39 raa-ibl diffuse scale while GI is active.
struct SlotsPart {
    ibl: Option<IblHandle>,
    applied: Cell<Option<f32>>,
}

impl GiPart for SlotsPart {
    fn name(&self) -> &'static str {
        "slots"
    }

    /// Sets the IBL diffuse scale to `environment_diffuse` on an active frame and to 1 otherwise,
    /// only when the target changed. Without an `IblHandle` it does nothing. `IblHandle`'s
    /// `apply_pending` runs in `sky.env` during pipeline execution, after
    /// `GiHandle::prepare_frame`, so the scale applies in the same frame.
    fn prepare_frame(&self, _gpu: &GpuContext, frame: &PartFrame<'_>) {
        let Some(ibl) = &self.ibl else {
            return;
        };
        let target = if frame.active {
            frame.settings.environment_diffuse
        } else {
            1.0
        };
        if self.applied.get() != Some(target) {
            ibl.set_diffuse_scale(target);
            self.applied.set(Some(target));
        }
    }

    /// The part owns no GPU bytes.
    fn allocated_bytes(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn providers_carry_their_slot_priority_and_name() {
        let indirect = indirect_provider();
        assert_eq!(indirect.slot, SlotId::Indirect);
        assert_eq!(indirect.priority, 10);
        assert_eq!(indirect.provider, GI_PROVIDER);
        let ao = ao_provider();
        assert_eq!(ao.slot, SlotId::Ao);
        assert_eq!(ao.priority, 20);
        assert_eq!(ao.provider, GI_PROVIDER);
        let visibility = local_visibility_provider();
        assert_eq!(visibility.slot, SlotId::LocalVisibility);
        assert_eq!(visibility.priority, 10);
        assert_eq!(visibility.provider, GI_PROVIDER);
    }

    #[test]
    fn only_local_visibility_ships_the_volume_trace() {
        assert!(!SLOT_INDIRECT_WGSL.contains("fn raa_gi_trace"));
        assert!(!SLOT_AO_WGSL.contains("fn raa_gi_trace"));
        assert!(local_visibility_provider().wgsl.contains("fn raa_gi_trace"));
    }

    #[test]
    fn local_visibility_uses_no_implicit_derivatives() {
        let text = local_visibility_provider().wgsl;
        for token in [
            "textureSample(",
            "textureSampleBias(",
            "textureSampleCompare(",
            "dpdx",
            "dpdy",
            "fwidth",
        ] {
            assert!(
                !text.contains(token),
                "LocalVisibility text contains {token}"
            );
        }
    }
}
