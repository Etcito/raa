//! Virtual shadow map (VSM) of the sun, CPU side: the page cache.
//!
//! The virtual map is a clipmap of [`MAX_LEVELS`] levels around the camera. Level `l` is a
//! 16384 x 16384 texel map ([`LEVEL_PAGES`]² pages of [`PAGE`]² texels) whose texel is
//! `TEXEL0_M * 2^l` metres, in the light space of the sun (`light_view_of`, x right, virtual
//! row down, depth along the travel direction). Pages have absolute coordinates in that light
//! space, so a page stays valid while the camera moves: only the window of 128 x 128 pages
//! around the camera ("origin") slides, and the page table of a level is addressed modulo 128
//! (toroidal).
//!
//! Each frame the GPU marks the pages the visible pixels need (`vsm_mark.wgsl`, read back a
//! few frames later); the cache applies the latest request set ([`VsmCache::apply_requests`]),
//! then [`VsmCache::plan`] picks the pages to render (requested, not resident or invalidated),
//! groups them by render window (a square of `window_pages`² pages rendered into one cascade
//! layer of the shadow array), allocates physical pool slots (free list, then least recently
//! requested pages), and rebuilds the page tables of the levels that changed. The pool slots
//! `0..levels` hold the page tables themselves (one 128 x 128 table per level, the entry of a
//! page is its slot + 1, 0 = not resident).
//!
//! Pages are invalidated when the sun direction or the scene depth range changes (all pages)
//! and when a caster's bounds change (the pages its light-space footprint overlaps). Static
//! pages are cached across frames; the per-frame page budget bounds the cost of a camera cut.

use std::collections::HashMap;

use raa_math::{Aabb, Mat4, Vec3, Vec4};

use crate::cascades::{ladder_up, light_view_of, ortho};

/// Texels per page side.
pub(crate) const PAGE: u32 = 128;
/// Pages per side of a level's clipmap window (16384 virtual texels).
pub(crate) const LEVEL_PAGES: i32 = 128;
/// Clipmap levels.
pub(crate) const MAX_LEVELS: u32 = 8;
/// Texel of level 0 in metres (level 7: 12.5 cm).
pub(crate) const TEXEL0_M: f32 = 1.0 / 1024.0;
/// Entries of one level's page table.
pub(crate) const TABLE_WORDS: usize = (LEVEL_PAGES * LEVEL_PAGES) as usize;
/// u32 words of the request bit set (one bit per table entry of every level).
pub(crate) const REQUEST_WORDS: usize = TABLE_WORDS / 32 * MAX_LEVELS as usize;
/// Render windows per frame (cascade layers 0..3; layer 3 holds the coarse scene map).
pub(crate) const MAX_WINDOWS: usize = 3;
/// Margin around the scene bounds in light space (m).
const SCENE_MARGIN_M: f32 = 0.5;
/// The scene depth range is widened to multiples of this (m), so small bounds changes keep
/// the cached pages.
const DEPTH_STEP_M: f32 = 4.0;
/// Distance of the farthest blocker the PCSS search looks for (m).
pub(crate) const BLOCKER_REACH_M: f32 = 8.0;
/// Pages per side of the dilation the marking pass applies (texels).
pub(crate) const MARK_MARGIN_TEXELS: f32 = 12.0;

/// One virtual page: level and absolute page coordinates (x right, y = virtual row down).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct PageKey {
    pub level: u8,
    pub x: i32,
    pub y: i32,
}

#[derive(Copy, Clone, Debug)]
struct PageEntry {
    slot: u32,
    /// Stamp of the latest request set that held the page.
    last_request: u64,
    /// The slot holds this page rendered for the current light and casters.
    valid: bool,
}

/// One render window of this frame: `window_pages`² pages of `level` starting at page
/// `cell * window_pages`, rendered into cascade layer `index`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Window {
    pub level: u8,
    pub cell: (i32, i32),
    pub matrix: Mat4,
    /// Pixel rect (x, y, width, height) of the pages copied from it.
    pub scissor: [u32; 4],
}

/// One page copy of this frame: window `window`'s texels at `src` into pool slot `slot`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct CopyJob {
    pub window: u32,
    pub src: (u32, u32),
    pub slot: u32,
}

/// The light space of the cache: the sun's view and the scene depth range.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct LightSpace {
    pub sun_dir: Vec3,
    pub view: Mat4,
    pub depth0: f32,
    pub depth1: f32,
}

impl LightSpace {
    /// World to (light x, light y, depth in [0, 1] over the scene depth range).
    pub(crate) fn matrix(&self) -> Mat4 {
        let (scale, bias) = self.depth_map();
        Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, -scale, 0.0),
            Vec4::new(0.0, 0.0, bias, 1.0),
        ) * self.view
    }

    fn depth_map(&self) -> (f32, f32) {
        let scale = 1.0 / (self.depth1 - self.depth0).max(1e-3);
        (scale, -self.depth0 * scale)
    }
}

/// What one frame gives the cache.
pub(crate) struct FrameInput<'a> {
    pub sun_dir: Vec3,
    pub eye: Vec3,
    pub scene: Option<Aabb>,
    /// World boxes of the casters that moved (old and new boxes) or are posed.
    pub changed: &'a [Aabb],
    /// Pages rendered at most this frame.
    pub budget: u32,
}

/// Diagnostics of the last plan (`shadowStats().vsm` in JS).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct VsmStats {
    /// Pages resident in the pool.
    pub resident: u32,
    /// Pages of the latest request set.
    pub requested: u32,
    /// Requested pages not rendered yet (budget, pool or window limits).
    pub pending: u32,
    /// Pages rendered this frame.
    pub rendered: u32,
    /// Render windows this frame.
    pub windows: u32,
    /// The coarse scene map was rendered this frame.
    pub coarse: bool,
    /// Pages invalidated by moving casters this frame.
    pub invalidated: u32,
    /// Requested pages that found no slot (pool full) this frame.
    pub starved: u32,
    /// Usable pool slots (the pool minus the page tables).
    pub capacity: u32,
    /// Frames between the marking of the applied request set and this frame.
    pub latency: u32,
    /// Page-table levels rewritten this frame.
    pub tables: u32,
}

/// The page cache.
pub(crate) struct VsmCache {
    levels: u32,
    slots: u32,
    window_pages: i32,
    pages: HashMap<PageKey, PageEntry>,
    owner: Vec<Option<PageKey>>,
    free: Vec<u32>,
    origins: [[i32; 2]; MAX_LEVELS as usize],
    tables: Vec<u32>,
    table_dirty: u32,
    wanted: Vec<PageKey>,
    wanted_stamp: u64,
    light: Option<LightSpace>,
    coarse: Option<Mat4>,
    stamp: u64,
    pub(crate) windows: Vec<Window>,
    pub(crate) copies: Vec<CopyJob>,
    /// The coarse map must be rendered this frame.
    pub(crate) coarse_dirty: bool,
    pub(crate) stats: VsmStats,
    queue: Vec<PageKey>,
    evict: Vec<(u64, PageKey)>,
    evict_built: bool,
}

impl VsmCache {
    /// A cache of `slots` pool slots (the first `MAX_LEVELS` hold the page tables) whose
    /// render windows are `resolution` texels wide.
    pub(crate) fn new(slots: u32, resolution: u32) -> VsmCache {
        let levels = MAX_LEVELS;
        let slots = slots.max(levels + 1);
        VsmCache {
            levels,
            slots,
            window_pages: (resolution / PAGE).max(1) as i32,
            pages: HashMap::new(),
            owner: vec![None; slots as usize],
            free: (levels..slots).rev().collect(),
            origins: [[0; 2]; MAX_LEVELS as usize],
            tables: vec![0; TABLE_WORDS * levels as usize],
            table_dirty: (1 << levels) - 1,
            wanted: Vec::new(),
            wanted_stamp: 0,
            light: None,
            coarse: None,
            stamp: 1,
            windows: Vec::new(),
            copies: Vec::new(),
            coarse_dirty: true,
            stats: VsmStats::default(),
            queue: Vec::new(),
            evict: Vec::new(),
            evict_built: false,
        }
    }

    pub(crate) fn levels(&self) -> u32 {
        self.levels
    }

    pub(crate) fn slots(&self) -> u32 {
        self.slots
    }

    pub(crate) fn window_pages(&self) -> i32 {
        self.window_pages
    }

    pub(crate) fn origins(&self) -> [[i32; 2]; MAX_LEVELS as usize] {
        self.origins
    }

    pub(crate) fn light(&self) -> Option<&LightSpace> {
        self.light.as_ref()
    }

    pub(crate) fn coarse_matrix(&self) -> Option<Mat4> {
        self.coarse
    }

    /// Bit `l` set: level `l`'s page table changed this frame.
    pub(crate) fn table_dirty(&self) -> u32 {
        self.table_dirty
    }

    /// The page table of `level` (`TABLE_WORDS` entries, slot + 1 or 0).
    pub(crate) fn table(&self, level: u32) -> &[u32] {
        let start = level as usize * TABLE_WORDS;
        &self.tables[start..start + TABLE_WORDS]
    }

    /// Texel size of `level` (m).
    pub(crate) fn texel_m(level: u32) -> f32 {
        TEXEL0_M * (1u32 << level) as f32
    }

    /// Drops every page (new light space, pool or window size).
    fn reset_pages(&mut self) {
        self.pages.clear();
        self.owner.iter_mut().for_each(|o| *o = None);
        self.free = (self.levels..self.slots).rev().collect();
        self.table_dirty = (1 << self.levels) - 1;
        self.wanted.clear();
    }

    fn free_page(&mut self, key: PageKey) {
        if let Some(entry) = self.pages.remove(&key) {
            if let Some(o) = self.owner.get_mut(entry.slot as usize) {
                *o = None;
            }
            self.free.push(entry.slot);
            self.table_dirty |= 1 << key.level;
        }
    }

    /// A free slot, or the slot of the least recently requested page outside the current
    /// request set; `None` when every slot holds a requested page.
    fn alloc(&mut self) -> Option<u32> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        if !self.evict_built {
            self.evict_built = true;
            let stamp = self.wanted_stamp;
            self.evict.clear();
            self.evict.extend(
                self.pages
                    .iter()
                    .filter(|(_, e)| e.last_request < stamp)
                    .map(|(k, e)| (e.last_request, *k)),
            );
            // Oldest last: popped first.
            self.evict.sort_unstable_by(|a, b| b.cmp(a));
        }
        while let Some((_, key)) = self.evict.pop() {
            if self.pages.contains_key(&key) {
                self.free_page(key);
                return self.free.pop();
            }
        }
        None
    }

    /// The light space for `sun_dir` over `scene` (depth range widened to `DEPTH_STEP_M`).
    fn light_space(sun_dir: Vec3, scene: &Aabb) -> Option<LightSpace> {
        if !(scene.min.is_finite() && scene.max.is_finite()) {
            return None;
        }
        let view = light_view_of(sun_dir);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for k in 0..8 {
            let p = corner(scene, k);
            let depth = -view.transform_point(p).z;
            lo = lo.min(depth);
            hi = hi.max(depth);
        }
        let depth0 = ((lo - SCENE_MARGIN_M) / DEPTH_STEP_M).floor() * DEPTH_STEP_M;
        let depth1 = ((hi + SCENE_MARGIN_M) / DEPTH_STEP_M).ceil() * DEPTH_STEP_M;
        Some(LightSpace {
            sun_dir: sun_dir.normalize_or_zero(),
            view,
            depth0,
            depth1,
        })
    }

    /// The scene-wide coarse map (cascade layer 3, the fallback of non-resident pages).
    fn coarse_of(light: &LightSpace, scene: &Aabb, resolution: u32) -> Mat4 {
        let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
        for k in 0..8 {
            let l = light.view.transform_point(corner(scene, k));
            lo = [lo[0].min(l.x), lo[1].min(l.y)];
            hi = [hi[0].max(l.x), hi[1].max(l.y)];
        }
        let extent = (hi[0] - lo[0]).max(hi[1] - lo[1]);
        let half = ladder_up(0.5 * extent + SCENE_MARGIN_M, 8.0);
        let texel = 2.0 * half / resolution.max(1) as f32;
        let cx = (0.5 * (lo[0] + hi[0]) / texel).round() * texel;
        let cy = (0.5 * (lo[1] + hi[1]) / texel).round() * texel;
        let (scale, bias) = light.depth_map();
        ortho(light.view, half, cx, cy, scale, bias)
    }

    /// The view-projection of the window of `window_pages`² pages of `level` at `cell`.
    pub(crate) fn window_matrix(&self, light: &LightSpace, level: u8, cell: (i32, i32)) -> Mat4 {
        let page_m = PAGE as f32 * Self::texel_m(u32::from(level));
        let n = self.window_pages as f32;
        let x0 = cell.0 as f32 * n;
        let y0 = cell.1 as f32 * n;
        let half = 0.5 * n * page_m;
        let cx = (x0 + 0.5 * n) * page_m;
        // Virtual rows grow downward: light y = -row.
        let cy = -(y0 + 0.5 * n) * page_m;
        let (scale, bias) = light.depth_map();
        ortho(light.view, half, cx, cy, scale, bias)
    }

    /// Applies a read-back request bit set (`REQUEST_WORDS` words) that was marked with the
    /// window origins `origins`; `latency` is its age in frames. The set replaces the previous
    /// one: its pages are the ones [`plan`](Self::plan) renders and keeps.
    pub(crate) fn apply_requests(
        &mut self,
        words: &[u32],
        origins: &[[i32; 2]; MAX_LEVELS as usize],
        latency: u32,
    ) {
        self.stamp += 1;
        self.wanted_stamp = self.stamp;
        self.wanted.clear();
        let per_level = TABLE_WORDS / 32;
        for level in 0..self.levels as usize {
            let [ox, oy] = origins[level];
            for (w, &bits) in words
                .iter()
                .skip(level * per_level)
                .take(per_level)
                .enumerate()
            {
                let mut bits = bits;
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let index = (w * 32 + b) as i32;
                    let (tx, ty) = (index % LEVEL_PAGES, index / LEVEL_PAGES);
                    let key = PageKey {
                        level: level as u8,
                        x: ox + (tx - ox).rem_euclid(LEVEL_PAGES),
                        y: oy + (ty - oy).rem_euclid(LEVEL_PAGES),
                    };
                    if let Some(e) = self.pages.get_mut(&key) {
                        e.last_request = self.wanted_stamp;
                    }
                    self.wanted.push(key);
                }
            }
        }
        self.stats.requested = self.wanted.len() as u32;
        self.stats.latency = latency;
    }

    fn inside(&self, key: &PageKey) -> bool {
        let [ox, oy] = self.origins[key.level as usize];
        (0..LEVEL_PAGES).contains(&(key.x - ox)) && (0..LEVEL_PAGES).contains(&(key.y - oy))
    }

    /// Plans this frame: light space, window origins, invalidation, the pages to render
    /// (`windows`, `copies`), their slots and the page tables. `resolution` is the window size.
    pub(crate) fn plan(&mut self, input: &FrameInput<'_>, resolution: u32) {
        self.windows.clear();
        self.copies.clear();
        self.table_dirty = 0;
        self.evict_built = false;
        self.coarse_dirty = false;
        let mut stats = VsmStats {
            requested: self.stats.requested,
            latency: self.stats.latency,
            capacity: self.slots - self.levels,
            ..VsmStats::default()
        };
        let Some(scene) = input.scene else {
            if self.light.take().is_some() {
                self.reset_pages();
            }
            self.coarse = None;
            self.stats = stats;
            return;
        };
        let Some(light) = Self::light_space(input.sun_dir, &scene) else {
            self.stats = stats;
            return;
        };
        if self.light.as_ref() != Some(&light) {
            self.reset_pages();
            self.light = Some(light);
            self.coarse_dirty = true;
        }
        let coarse = Self::coarse_of(&light, &scene, resolution);
        if self.coarse != Some(coarse) {
            self.coarse = Some(coarse);
            self.coarse_dirty = true;
        }

        // Window origins: 64 pages on each side of the camera's page.
        let eye = light.view.transform_point(input.eye);
        for level in 0..self.levels {
            let page_m = PAGE as f32 * Self::texel_m(level);
            let origin = [
                (eye.x / page_m).floor() as i32 - LEVEL_PAGES / 2,
                (-eye.y / page_m).floor() as i32 - LEVEL_PAGES / 2,
            ];
            if self.origins[level as usize] != origin {
                self.origins[level as usize] = origin;
                self.table_dirty |= 1 << level;
            }
        }

        // Moving casters: requested pages under their light-space footprint are re-rendered
        // (they keep showing the previous content meanwhile), the others are dropped.
        if !input.changed.is_empty() {
            self.coarse_dirty = true;
            for b in input.changed {
                stats.invalidated += self.invalidate(&light, b);
            }
        }

        // The requested pages to render, grouped by window cell.
        self.queue.clear();
        for key in &self.wanted {
            if !self.inside(key) {
                continue;
            }
            if self.pages.get(key).is_none_or(|e| !e.valid) {
                self.queue.push(*key);
            }
        }
        let n = self.window_pages;
        let cell_of = |k: &PageKey| (k.level, k.x.div_euclid(n), k.y.div_euclid(n));
        let mut cells: HashMap<(u8, i32, i32), u32> = HashMap::new();
        for key in &self.queue {
            *cells.entry(cell_of(key)).or_default() += 1;
        }
        let mut order: Vec<((u8, i32, i32), u32)> = cells.into_iter().collect();
        // Fullest windows first, coarse levels first on ties (a coarse page also serves the
        // finer pixels as a fallback).
        order.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
        self.queue.sort_unstable();
        let mut left = input.budget;
        let mut starved = false;
        'cells: for (cell, _) in order.into_iter().take(MAX_WINDOWS) {
            if left == 0 || starved {
                break;
            }
            let window = self.windows.len() as u32;
            let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            let keys: Vec<PageKey> = self
                .queue
                .iter()
                .filter(|k| cell_of(k) == cell)
                .copied()
                .collect();
            for key in keys {
                if left == 0 {
                    break;
                }
                let slot = match self.pages.get(&key) {
                    Some(e) => e.slot,
                    None => match self.alloc() {
                        Some(slot) => {
                            self.owner[slot as usize] = Some(key);
                            self.table_dirty |= 1 << key.level;
                            slot
                        }
                        None => {
                            starved = true;
                            if x0 == i32::MAX {
                                break 'cells;
                            }
                            break;
                        }
                    },
                };
                self.pages.insert(
                    key,
                    PageEntry {
                        slot,
                        last_request: self.wanted_stamp,
                        valid: true,
                    },
                );
                let local = (key.x - cell.1 * n, key.y - cell.2 * n);
                self.copies.push(CopyJob {
                    window,
                    src: (local.0 as u32 * PAGE, local.1 as u32 * PAGE),
                    slot,
                });
                x0 = x0.min(local.0);
                y0 = y0.min(local.1);
                x1 = x1.max(local.0);
                y1 = y1.max(local.1);
                left -= 1;
            }
            if x0 == i32::MAX {
                continue;
            }
            self.windows.push(Window {
                level: cell.0,
                cell: (cell.1, cell.2),
                matrix: self.window_matrix(&light, cell.0, (cell.1, cell.2)),
                scissor: [
                    x0 as u32 * PAGE,
                    y0 as u32 * PAGE,
                    (x1 - x0 + 1) as u32 * PAGE,
                    (y1 - y0 + 1) as u32 * PAGE,
                ],
            });
        }

        // Page tables of the levels that changed.
        for level in 0..self.levels {
            if self.table_dirty & (1 << level) == 0 {
                continue;
            }
            let start = level as usize * TABLE_WORDS;
            self.tables[start..start + TABLE_WORDS].fill(0);
            let [ox, oy] = self.origins[level as usize];
            for (key, e) in &self.pages {
                if u32::from(key.level) != level {
                    continue;
                }
                if !(0..LEVEL_PAGES).contains(&(key.x - ox))
                    || !(0..LEVEL_PAGES).contains(&(key.y - oy))
                {
                    continue;
                }
                let index = (key.y & (LEVEL_PAGES - 1)) * LEVEL_PAGES + (key.x & (LEVEL_PAGES - 1));
                self.tables[start + index as usize] = e.slot + 1;
            }
            stats.tables += 1;
        }

        stats.resident = self.pages.len() as u32;
        stats.rendered = self.copies.len() as u32;
        stats.windows = self.windows.len() as u32;
        stats.coarse = self.coarse_dirty;
        stats.pending = self.queue.len() as u32 - stats.rendered.min(self.queue.len() as u32);
        stats.starved = u32::from(starved) * stats.pending;
        self.stats = stats;
    }

    /// Invalidates the pages under the light-space footprint of `b` (one page of margin for
    /// the filter); returns how many.
    fn invalidate(&mut self, light: &LightSpace, b: &Aabb) -> u32 {
        let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
        for k in 0..8 {
            let l = light.view.transform_point(corner(b, k));
            lo = [lo[0].min(l.x), lo[1].min(-l.y)];
            hi = [hi[0].max(l.x), hi[1].max(-l.y)];
        }
        if !(lo[0].is_finite() && lo[1].is_finite() && hi[0].is_finite() && hi[1].is_finite()) {
            return 0;
        }
        let mut hits: Vec<PageKey> = Vec::new();
        for level in 0..self.levels {
            let page_m = PAGE as f32 * Self::texel_m(level);
            let x0 = (lo[0] / page_m).floor() as i64 - 1;
            let x1 = (hi[0] / page_m).floor() as i64 + 1;
            let y0 = (lo[1] / page_m).floor() as i64 - 1;
            let y1 = (hi[1] / page_m).floor() as i64 + 1;
            let area = (x1 - x0 + 1) * (y1 - y0 + 1);
            if area <= 4096 {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let key = PageKey {
                            level: level as u8,
                            x: x as i32,
                            y: y as i32,
                        };
                        if self.pages.contains_key(&key) {
                            hits.push(key);
                        }
                    }
                }
            } else {
                hits.extend(self.pages.keys().filter(|k| {
                    u32::from(k.level) == level
                        && (x0..=x1).contains(&i64::from(k.x))
                        && (y0..=y1).contains(&i64::from(k.y))
                }));
            }
        }
        let stamp = self.wanted_stamp;
        let mut count = 0;
        for key in hits {
            let Some(e) = self.pages.get_mut(&key) else {
                continue;
            };
            if e.last_request == stamp {
                if e.valid {
                    e.valid = false;
                    count += 1;
                }
            } else {
                self.free_page(key);
                count += 1;
            }
        }
        count
    }
}

/// A window matrix whose box is far outside any scene: its caster list selects nothing.
pub(crate) fn empty_window(light: &LightSpace) -> Mat4 {
    let (scale, bias) = light.depth_map();
    ortho(light.view, 1.0, 1.0e7, 1.0e7, scale, bias)
}

/// Corner `k` (bits x, y, z) of `b`.
fn corner(b: &Aabb, k: usize) -> Vec3 {
    Vec3::new(
        if k & 1 == 0 { b.min.x } else { b.max.x },
        if k & 2 == 0 { b.min.y } else { b.max.y },
        if k & 4 == 0 { b.min.z } else { b.max.z },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> Aabb {
        Aabb {
            min: Vec3::new(-20.0, 0.0, -10.0),
            max: Vec3::new(20.0, 15.0, 10.0),
        }
    }

    fn input(changed: &[Aabb]) -> FrameInput<'_> {
        FrameInput {
            sun_dir: Vec3::new(0.3, -1.0, 0.2),
            eye: Vec3::new(1.0, 1.7, 0.5),
            scene: Some(scene()),
            changed,
            budget: 256,
        }
    }

    /// A request set holding `keys` (window-relative toroidal bits).
    fn request(cache: &VsmCache, keys: &[PageKey]) -> Vec<u32> {
        let mut words = vec![0u32; REQUEST_WORDS];
        for k in keys {
            let index = (k.y & 127) * 128 + (k.x & 127);
            let bit = k.level as usize * TABLE_WORDS + index as usize;
            words[bit / 32] |= 1 << (bit % 32);
        }
        let _ = cache;
        words
    }

    fn near_page(cache: &VsmCache, level: u8, dx: i32, dy: i32) -> PageKey {
        let [ox, oy] = cache.origins()[level as usize];
        PageKey {
            level,
            x: ox + 64 + dx,
            y: oy + 64 + dy,
        }
    }

    #[test]
    fn requested_pages_are_rendered_once_and_cached() {
        let mut c = VsmCache::new(2048, 2048);
        c.plan(&input(&[]), 2048);
        assert!(c.coarse_dirty, "first frame renders the coarse map");
        assert!(c.copies.is_empty());
        let keys = [
            near_page(&c, 0, 0, 0),
            near_page(&c, 0, 1, 0),
            near_page(&c, 2, -3, 5),
        ];
        let words = request(&c, &keys);
        let origins = c.origins();
        c.apply_requests(&words, &origins, 2);
        assert_eq!(c.stats.requested, 3);
        c.plan(&input(&[]), 2048);
        assert_eq!(c.copies.len(), 3);
        assert_eq!(c.windows.len(), 2);
        assert!(!c.coarse_dirty);
        // Tables hold slot + 1 at the toroidal entries.
        for k in keys {
            let e = c.table(u32::from(k.level))[((k.y & 127) * 128 + (k.x & 127)) as usize];
            assert!(e > MAX_LEVELS, "{k:?} -> {e}");
        }
        // Next frame: cached, nothing rendered.
        c.plan(&input(&[]), 2048);
        assert!(c.copies.is_empty());
        assert_eq!(c.stats.resident, 3);
    }

    #[test]
    fn window_matrix_maps_pages_to_window_texels() {
        let mut c = VsmCache::new(2048, 2048);
        c.plan(&input(&[]), 2048);
        let light = *c.light().unwrap();
        let level = 3u8;
        let cell = (-2, 5);
        let m = c.window_matrix(&light, level, cell);
        let lm = light.matrix();
        let t = VsmCache::texel_m(u32::from(level));
        // A world point; its virtual texel from the light matrix must land at the same
        // window texel as the window projection gives.
        let inv = lm.inverse().unwrap();
        let u = (cell.0 * 16) as f32 * 128.0 + 300.25;
        let v = (cell.1 * 16) as f32 * 128.0 + 1000.75;
        let world = inv.transform_point(Vec3::new(u * t, -v * t, 0.5));
        let clip = m.transform_point(world);
        let px = (clip.x * 0.5 + 0.5) * 2048.0;
        let py = (0.5 - clip.y * 0.5) * 2048.0;
        assert!((px - 300.25).abs() < 0.05, "{px}");
        assert!((py - 1000.75).abs() < 0.05, "{py}");
        assert!((clip.z - 0.5).abs() < 1e-3, "{}", clip.z);
    }

    #[test]
    fn pool_evicts_least_recently_requested_pages() {
        // 8 table slots + 4 page slots.
        let mut c = VsmCache::new(12, 2048);
        c.plan(&input(&[]), 2048);
        let a: Vec<PageKey> = (0..4).map(|i| near_page(&c, 1, i, 0)).collect();
        let origins = c.origins();
        c.apply_requests(&request(&c, &a), &origins, 1);
        c.plan(&input(&[]), 2048);
        assert_eq!(c.copies.len(), 4);
        let b: Vec<PageKey> = (0..2).map(|i| near_page(&c, 1, i, 3)).collect();
        c.apply_requests(&request(&c, &b), &origins, 1);
        c.plan(&input(&[]), 2048);
        assert_eq!(c.copies.len(), 2);
        assert_eq!(c.stats.resident, 4);
        assert_eq!(c.stats.starved, 0);
        // All 6 at once do not fit: 4 rendered at most, the rest starves.
        let mut all = a.clone();
        all.extend(&b);
        c.apply_requests(&request(&c, &all), &origins, 1);
        c.plan(&input(&[]), 2048);
        assert_eq!(c.stats.resident, 4);
        assert!(c.stats.pending > 0);
    }

    #[test]
    fn moving_caster_invalidates_only_its_pages_and_sun_change_resets() {
        let mut c = VsmCache::new(2048, 2048);
        c.plan(&input(&[]), 2048);
        let keys = [near_page(&c, 0, 0, 0), near_page(&c, 0, 40, 40)];
        let origins = c.origins();
        c.apply_requests(&request(&c, &keys), &origins, 1);
        c.plan(&input(&[]), 2048);
        assert_eq!(c.copies.len(), 2);
        // A box at the first page's position.
        let light = *c.light().unwrap();
        let t = VsmCache::texel_m(0) * 128.0;
        let inv = light.view.inverse().unwrap();
        let k = keys[0];
        let p = inv.transform_point(Vec3::new(
            (k.x as f32 + 0.5) * t,
            -(k.y as f32 + 0.5) * t,
            -10.0,
        ));
        let b = Aabb {
            min: p - Vec3::new(0.01, 0.01, 0.01),
            max: p + Vec3::new(0.01, 0.01, 0.01),
        };
        c.plan(&input(&[b]), 2048);
        assert!(c.coarse_dirty);
        assert_eq!(c.copies.len(), 1, "only the page under the box");
        assert_eq!(c.stats.invalidated, 1);
        // A new sun direction drops everything.
        let mut moved = input(&[]);
        moved.sun_dir = Vec3::new(0.5, -1.0, 0.2);
        c.plan(&moved, 2048);
        assert_eq!(c.stats.resident, 0);
        assert!(c.coarse_dirty);
    }

    #[test]
    fn budget_caps_pages_per_frame() {
        let mut c = VsmCache::new(4096, 2048);
        c.plan(&input(&[]), 2048);
        let keys: Vec<PageKey> = (0..40)
            .flat_map(|y| (0..20).map(move |x| (x, y)))
            .map(|(x, y)| near_page(&c, 0, x - 10, y - 20))
            .collect();
        let origins = c.origins();
        c.apply_requests(&request(&c, &keys), &origins, 1);
        let mut low = input(&[]);
        low.budget = 100;
        c.plan(&low, 2048);
        assert_eq!(c.copies.len(), 100);
        assert!(c.windows.len() <= MAX_WINDOWS);
        let mut total = 100;
        for _ in 0..20 {
            c.plan(&low, 2048);
            total += c.copies.len();
        }
        assert_eq!(total, 800);
    }
}
