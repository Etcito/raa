//! The virtualized-instance set (P3-LIT-SEAM): scene instances that leave the lit renderer's
//! draw lists (prepass, forward, cascade casters) while their persistent instance, extras
//! and previous-instance records are still written and committed exactly as before.
//!
//! Membership is kept per scene slot (`InstanceId::to_bits() as u32`): one bit says the slot
//! holds a member, and the generation of that member is stored beside it. So `contains`
//! answers for the exact handle, a stale id of a reused slot is not a member, and no query
//! allocates. Bits and generations grow only when a higher slot is first added.

use raa_scene::{InstanceId, Scene};

/// The set of virtualized scene instances; see the module docs. An empty set (the default)
/// changes nothing in the renderer.
#[derive(Clone, Debug, Default)]
pub struct VirtualSet {
    /// One bit per scene slot, set where the slot holds a member.
    bits: Vec<u64>,
    /// Generation of the member of each slot; meaningful where its bit is set.
    generations: Vec<u32>,
    len: usize,
    /// Advances on every membership change (`set` that changes it, a non-empty `clear`).
    epoch: u64,
}

/// The scene slot and generation of `instance`, from its packed bits (generation high,
/// index low).
fn split(instance: InstanceId) -> (usize, u32) {
    let bits = instance.to_bits();
    (bits as u32 as usize, (bits >> 32) as u32)
}

/// Whether bit `slot` of a slot bitset is set.
pub(crate) fn slot_member(bits: &[u64], slot: usize) -> bool {
    bits.get(slot / 64)
        .is_some_and(|word| word & (1u64 << (slot % 64)) != 0)
}

impl VirtualSet {
    /// An empty set (generation 0).
    pub fn new() -> VirtualSet {
        VirtualSet::default()
    }

    /// Adds `instance` (`on`) or removes it. Adding a member that is already present, or
    /// removing one that is not (a stale id included), changes nothing. Adding over a
    /// stale member of the same slot replaces it and counts as a change.
    pub fn set(&mut self, instance: InstanceId, on: bool) {
        if self.contains(instance) == on {
            return;
        }
        let (slot, generation) = split(instance);
        if on {
            self.reserve(slot);
            if !slot_member(&self.bits, slot) {
                self.len += 1;
            }
            if let Some(word) = self.bits.get_mut(slot / 64) {
                *word |= 1u64 << (slot % 64);
            }
            if let Some(stored) = self.generations.get_mut(slot) {
                *stored = generation;
            }
        } else {
            if let Some(word) = self.bits.get_mut(slot / 64) {
                *word &= !(1u64 << (slot % 64));
            }
            self.len = self.len.saturating_sub(1);
        }
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Whether exactly `instance` (slot and generation) is a member.
    pub fn contains(&self, instance: InstanceId) -> bool {
        let (slot, generation) = split(instance);
        slot_member(&self.bits, slot) && self.generations.get(slot).copied() == Some(generation)
    }

    /// Removes every member; a non-empty set advances the generation, an empty one does not.
    pub fn clear(&mut self) {
        if self.len == 0 {
            return;
        }
        self.bits.fill(0);
        self.len = 0;
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// The number of members.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the set has no member.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The membership generation: 0 for a new set, +1 per change of membership.
    pub fn generation(&self) -> u64 {
        self.epoch
    }

    /// The (slot, generation) of every member, in slot order.
    pub(crate) fn members(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.bits.iter().enumerate().flat_map(move |(word, &bits)| {
            (0..64usize)
                .filter(move |&bit| bits & (1u64 << bit) != 0)
                .map(move |bit| {
                    let slot = word * 64 + bit;
                    (
                        slot as u32,
                        self.generations.get(slot).copied().unwrap_or(0),
                    )
                })
        })
    }

    /// Writes the members that are live in `scene` (their slot holds the same generation)
    /// into the slot bitset `out`, of at least `words` words. `out` is reused: it is cleared
    /// and resized in place, so a warm frame allocates nothing.
    pub(crate) fn mirror(&self, scene: &Scene, words: usize, out: &mut Vec<u64>) {
        out.clear();
        out.resize(words.max(self.bits.len()), 0);
        for (slot, generation) in self.members() {
            let live = scene
                .instance_at_slot(slot)
                .is_some_and(|id| id.to_bits() >> 32 == u64::from(generation));
            if let Some(word) = out.get_mut(slot as usize / 64).filter(|_| live) {
                *word |= 1u64 << (slot % 64);
            }
        }
    }

    /// Grows the bitset and generation table so that `slot` has a bit.
    fn reserve(&mut self, slot: usize) {
        let words = slot / 64 + 1;
        if self.bits.len() < words {
            self.bits.resize(words, 0);
            self.generations.resize(words * 64, 0);
        }
    }
}
