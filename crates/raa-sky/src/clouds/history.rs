//! Device-free per-view lifecycle of the M2 cloud backend (contract
//! `P2-CLOUD-HISTORY`): panorama/screen validity and the committed screen
//! history of view slots `0..MAX_VIEWS`. No GPU type, no allocation.

use raa_math::Mat4;

use super::super::{CloudInvalidation, MAX_VIEWS};

/// One view slot's lifecycle state.
#[derive(Clone, Copy, Debug, Default)]
struct Slot {
    /// Size and invocation of the last `prepare`; `None` when never prepared or removed.
    prepared: Option<((u32, u32), u64)>,
    marchable: bool,
    panorama_valid: bool,
    screen_valid: bool,
    /// Screen-march note of the prepared invocation: `Some(None)` = recorded nothing.
    note: Option<Option<Mat4>>,
    /// Committed screen march: (size, camera view-projection).
    committed: Option<((u32, u32), Mat4)>,
}

impl Slot {
    fn clear_validity(&mut self) {
        self.panorama_valid = false;
        self.screen_valid = false;
        self.note = None;
        self.committed = None;
    }
}

/// The per-view cloud lifecycle of contract `P2-CLOUD-VALIDITY`.
#[derive(Clone, Debug)]
pub(super) struct CloudHistory {
    slots: [Slot; MAX_VIEWS],
}

impl CloudHistory {
    /// No slot prepared, every flag false.
    pub(super) fn new() -> CloudHistory {
        CloudHistory {
            slots: [Slot::default(); MAX_VIEWS],
        }
    }

    /// Stores the validated frame's size, invocation and marchability.
    pub(super) fn prepare(
        &mut self,
        view: u32,
        size: (u32, u32),
        invocation: u64,
        marchable: bool,
        panorama: bool,
    ) {
        let Some(slot) = self.slots.get_mut(view as usize) else {
            return;
        };
        if let Some((old_size, old_invocation)) = slot.prepared {
            if old_invocation != invocation {
                slot.note = None;
            }
            if old_size != size {
                slot.screen_valid = false;
                slot.note = None;
                slot.committed = None;
            }
        }
        slot.prepared = Some((size, invocation));
        slot.marchable = marchable;
        if marchable {
            slot.panorama_valid = panorama;
        } else {
            slot.clear_validity();
        }
    }

    /// True only for the prepared invocation of a marchable `prepare`.
    pub(super) fn marchable(&self, view: u32, invocation: u64) -> bool {
        self.slots.get(view as usize).is_some_and(|slot| {
            slot.marchable && slot.prepared.is_some_and(|(_, i)| i == invocation)
        })
    }

    /// Notes whether the prepared invocation recorded a screen march.
    pub(super) fn record_screen(&mut self, view: u32, invocation: u64, marched: Option<Mat4>) {
        if let Some(slot) = self.slots.get_mut(view as usize)
            && slot.prepared.is_some_and(|(_, i)| i == invocation)
        {
            slot.note = Some(marched);
        }
    }

    /// Commits the note of an acknowledged submission of the prepared invocation.
    pub(super) fn submitted(&mut self, view: u32, invocation: u64) {
        let Some(slot) = self.slots.get_mut(view as usize) else {
            return;
        };
        let Some((size, prepared)) = slot.prepared else {
            return;
        };
        if prepared != invocation {
            return;
        }
        match slot.note.take().flatten() {
            Some(matrix) => {
                slot.screen_valid = true;
                slot.committed = Some((size, matrix));
            }
            None => {
                slot.screen_valid = false;
                slot.committed = None;
            }
        }
    }

    /// Clears validity and history of one view or (`None`) every view.
    pub(super) fn invalidate(&mut self, view: Option<u32>, reason: CloudInvalidation) {
        let range = match view {
            Some(view) if (view as usize) < MAX_VIEWS => view as usize..view as usize + 1,
            Some(_) => return,
            None => 0..MAX_VIEWS,
        };
        for slot in &mut self.slots[range] {
            slot.clear_validity();
            if reason == CloudInvalidation::ViewRemoved {
                *slot = Slot::default();
            }
        }
    }

    /// `(panorama_valid, screen_valid)`; `(false, false)` for unknown views.
    pub(super) fn flags(&self, view: u32) -> (bool, bool) {
        self.slots
            .get(view as usize)
            .map_or((false, false), |slot| {
                (slot.panorama_valid, slot.screen_valid)
            })
    }

    /// The committed matrix when the frame allows history and the size still matches.
    pub(super) fn history(&self, view: u32, frame_history_valid: bool) -> Option<Mat4> {
        if !frame_history_valid {
            return None;
        }
        let slot = self.slots.get(view as usize)?;
        let (size, _) = slot.prepared?;
        slot.committed
            .filter(|(committed, _)| *committed == size)
            .map(|(_, matrix)| matrix)
    }
}
