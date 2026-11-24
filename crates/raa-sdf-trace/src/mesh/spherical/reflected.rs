use std::{cmp::Ordering, collections::BTreeMap};

use super::grid::{Surface, dot, sub};
use super::{CellKey, SphericalGrid};

pub const MAX_REFLECTED_RECEIVERS: usize = 256;

/// World-space position and shading normal from a reflected mesh hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectedReceiver {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub priority: u32,
}

#[derive(Clone, Copy, PartialEq)]
pub(super) struct Candidate {
    pub key: CellKey,
    receiver: ReflectedReceiver,
    radius: f64,
}

impl Candidate {
    pub fn supports(self, surface: Surface) -> bool {
        let normal = surface.normal.map(f64::from);
        let delta = sub(
            self.receiver.position.map(f64::from),
            surface.position.map(f64::from),
        );
        dot(normal, self.receiver.normal.map(f64::from)) > 0.4
            && dot(delta, normal).abs() < 0.65 * self.radius
            && dot(delta, delta) < (2.5 * self.radius).powi(2)
    }
}

pub(super) fn candidates(grid: &SphericalGrid, receivers: &[ReflectedReceiver]) -> Vec<Candidate> {
    let mut ranked = BTreeMap::new();
    for &receiver in receivers.iter().take(MAX_REFLECTED_RECEIVERS) {
        let Some(key) = grid.key_at(receiver.position) else {
            continue;
        };
        let normal = receiver.normal.map(f64::from);
        let length = dot(normal, normal).sqrt();
        if receiver.priority == 0 || !length.is_finite() || length == 0.0 {
            continue;
        }
        let receiver = ReflectedReceiver {
            normal: normal.map(|n| (n / length) as f32),
            ..receiver
        };
        for (key, weight) in grid.lookup_ring(key) {
            let contribution = u64::from(receiver.priority) * u64::from(weight);
            let shell = grid.radial_shell(key.shell).expect("valid lookup shell");
            let radius = (shell.outer - shell.inner)
                .max(2.0 * shell.center / f32::from(grid.angular_resolution()));
            let (score, strongest, best) = ranked.entry(key).or_insert((
                0u64,
                0u64,
                Candidate {
                    key,
                    receiver,
                    radius: f64::from(radius),
                },
            ));
            *score += contribution;
            if contribution > *strongest
                || (contribution == *strongest && receiver_order(receiver, best.receiver).is_lt())
            {
                *strongest = contribution;
                best.receiver = receiver;
            }
        }
    }
    let mut ranked: Vec<_> = ranked.into_values().collect();
    ranked.sort_by_key(|(score, _, candidate)| (std::cmp::Reverse(*score), candidate.key));
    ranked
        .into_iter()
        .map(|(_, _, candidate)| candidate)
        .collect()
}

// GPU append order may vary between otherwise identical captures.
fn receiver_order(a: ReflectedReceiver, b: ReflectedReceiver) -> Ordering {
    a.position
        .into_iter()
        .chain(a.normal)
        .zip(b.position.into_iter().chain(b.normal))
        .map(|(a, b)| a.total_cmp(&b))
        .find(|order| !order.is_eq())
        .unwrap_or(Ordering::Equal)
}
