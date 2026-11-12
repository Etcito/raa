//! Device-free per-view lifecycle of the M2 cloud backend (contract
//! `P2-CLOUD-HISTORY`): panorama/screen validity and the committed screen
//! history of view slots `0..MAX_VIEWS`. No GPU type, no allocation.

use raa_math::Mat4;

use super::super::{CloudInvalidation, MAX_VIEWS};

/// Frames of one unchanged view after which a non-animated view stops marching.
pub(super) const FREEZE_FRAMES: u32 = 64;
/// A frozen view still runs a cheap refresh pass on every invocation divisible by this.
pub(super) const REFRESH_PERIOD: u64 = 8;
/// Largest accumulation count kept.
const MAX_COUNT: u32 = 1 << 16;

/// What the committed screen march was made with, besides its camera matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Marked {
    /// Frames accumulated at the march: 1 after a change, then 2, 3, ... while unchanged.
    pub(super) count: u32,
    /// `CloudFrame::signature` of the march.
    pub(super) signature: u64,
}

/// The accumulation of one frame: the committed camera to reproject from and the count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Accumulation {
    /// The committed view-projection, reprojected from.
    pub(super) previous: Mat4,
    /// Frames accumulated including this one: 1 when the camera or the animation signature
    /// changed since the committed march (the blend then stays the moving-camera EMA), else
    /// the committed count + 1.
    pub(super) count: u32,
}

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
    /// Count and signature of the noted march.
    note_marked: Option<Marked>,
    /// Committed screen march: (size, camera view-projection).
    committed: Option<((u32, u32), Mat4)>,
    /// Count and signature of the committed march.
    committed_marked: Option<Marked>,
    /// What the last recorded screen frame decided (diagnostics only).
    last_frame: Option<FrameNote>,
}

/// The freeze decision of the last recorded frame of a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FrameNote {
    pub(super) animated: bool,
    pub(super) frozen: bool,
    pub(super) history_used: bool,
}

impl Slot {
    fn clear_validity(&mut self) {
        self.panorama_valid = false;
        self.screen_valid = false;
        self.note = None;
        self.note_marked = None;
        self.committed = None;
        self.committed_marked = None;
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
                slot.note_marked = None;
            }
            if old_size != size {
                slot.screen_valid = false;
                slot.note = None;
                slot.note_marked = None;
                slot.committed = None;
                slot.committed_marked = None;
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

    /// Notes the freeze decision of the prepared invocation (observation only).
    pub(super) fn record_frame(&mut self, view: u32, invocation: u64, note: FrameNote) {
        if let Some(slot) = self.slots.get_mut(view as usize)
            && slot.prepared.is_some_and(|(_, i)| i == invocation)
        {
            slot.last_frame = Some(note);
        }
    }

    /// `(committed count, last frame note)` of a view, for the diagnostics.
    pub(super) fn observation(&self, view: u32) -> (u32, Option<FrameNote>) {
        self.slots.get(view as usize).map_or((0, None), |slot| {
            (
                slot.committed_marked.map_or(0, |m| m.count),
                slot.last_frame,
            )
        })
    }

    /// Notes the accumulation count and signature of the prepared invocation's screen march.
    pub(super) fn record_marked(&mut self, view: u32, invocation: u64, marked: Marked) {
        if let Some(slot) = self.slots.get_mut(view as usize)
            && slot.prepared.is_some_and(|(_, i)| i == invocation)
        {
            slot.note_marked = Some(marked);
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
        let marked = slot.note_marked.take();
        match slot.note.take().flatten() {
            Some(matrix) => {
                slot.screen_valid = true;
                slot.committed = Some((size, matrix));
                slot.committed_marked = marked;
            }
            None => {
                slot.screen_valid = false;
                slot.committed = None;
                slot.committed_marked = None;
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

    /// Forgets the committed screen march of `view` (its screen target was recreated).
    pub(super) fn reset_screen(&mut self, view: u32) {
        if let Some(slot) = self.slots.get_mut(view as usize) {
            slot.screen_valid = false;
            slot.note = None;
            slot.note_marked = None;
            slot.committed = None;
            slot.committed_marked = None;
        }
    }

    /// The accumulation of the frame when it allows history and the size still matches:
    /// the count grows only while `view_proj` and `signature` equal the committed march's.
    pub(super) fn accumulation(
        &self,
        view: u32,
        frame_history_valid: bool,
        view_proj: Mat4,
        signature: u64,
    ) -> Option<Accumulation> {
        let previous = self.history(view, frame_history_valid)?;
        let last = self.slots.get(view as usize)?.committed_marked;
        let unchanged = previous == view_proj && last.is_some_and(|m| m.signature == signature);
        Some(Accumulation {
            previous,
            count: match last {
                Some(m) if unchanged => (m.count + 1).min(MAX_COUNT),
                _ => 1,
            },
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
