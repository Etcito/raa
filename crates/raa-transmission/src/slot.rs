//! The `screen` provider of the `Transmission` slot (S11): `wgsl/transmission.wgsl`.

use std::borrow::Cow;

use raa_shader::{SlotId, SlotProvider};

/// Provider name of the screen-space transmission slot.
pub const SCREEN_PROVIDER: &str = "screen";

/// Priority of the screen transmission provider (above the raa-lit defaults at 0).
pub(crate) const SCREEN_PRIORITY: i32 = 10;

/// `raa_slot_transmission` of provider `screen` and its private `raa_screen_` helpers.
pub(crate) const SCREEN_WGSL: &str = include_str!("wgsl/transmission.wgsl");

/// The `Transmission` slot provider `screen`, priority 10: the refracted view ray reads the
/// opaque colour copy at a roughness-driven mip, absorbed by Beer-Lambert and weighted by
/// (1 - Fresnel) and the material transmission.
#[allow(dead_code, reason = "installed by TransmissionFeature::install")]
pub(crate) fn screen_provider() -> SlotProvider {
    SlotProvider {
        slot: SlotId::Transmission,
        priority: SCREEN_PRIORITY,
        provider: SCREEN_PROVIDER,
        wgsl: Cow::Borrowed(SCREEN_WGSL),
    }
}

// The raa-lit default providers are crate-private; the test compiles the same source file
// (as crates/raa-lit/tests/variants.rs does).
#[cfg(test)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

#[cfg(test)]
mod tests {
    use super::*;
    use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, VariantKey};

    fn composer() -> Composer {
        let mut composer = Composer::new();
        for p in lit_slots::default_slot_providers()
            .into_iter()
            .chain([screen_provider()])
        {
            if let Err(e) = composer.add_slot(p) {
                panic!("{e}");
            }
        }
        composer
    }

    #[test]
    fn screen_provider_defines_only_its_slot_and_prefixed_helpers() {
        let fns: Vec<&str> = SCREEN_WGSL
            .lines()
            .filter_map(|l| l.strip_prefix("fn "))
            .filter_map(|l| l.split('(').next())
            .collect();
        assert_eq!(
            fns.iter()
                .filter(|f| **f == SlotId::Transmission.fn_name())
                .count(),
            1
        );
        for f in fns {
            assert!(
                f == SlotId::Transmission.fn_name() || f.starts_with("raa_screen_"),
                "helper `{f}` is not prefixed raa_screen_"
            );
        }
    }

    #[test]
    fn screen_provider_wins_the_transmission_slot() {
        let composer = composer();
        let w = composer.winner(SlotId::Transmission).expect("winner");
        assert_eq!((w.provider, w.priority), (SCREEN_PROVIDER, SCREEN_PRIORITY));
    }

    #[test]
    fn standard_transparent_with_screen_transmission_validates() {
        let reg = MaterialFnRegistry::new();
        let composer = composer();
        for msaa in [1, 4] {
            let key = VariantKey {
                material_fn: MaterialFnId::STANDARD,
                pass: PassKind::Transparent,
                msaa,
                flags: 0,
            };
            let wgsl = composer
                .compose(&reg, &key)
                .unwrap_or_else(|e| panic!("{key:?}: {e}"));
            assert!(wgsl.contains("fn raa_screen_refract("), "{key:?}");
            if let Err(e) = Composer::validate(&wgsl) {
                panic!("{key:?}: {e}");
            }
        }
    }
}
