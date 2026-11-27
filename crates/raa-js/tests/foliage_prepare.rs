//! m7 foliage group preparation (`prepare_foliage_groups`): the per-view call that installs
//! whole-tree groups after the view's camera and before `lit.prepare`, the missing, pending
//! and failed rules that keep source draws visible, the split-view level records and the
//! teardown order.
//!
//! `EngineApp::update` (viewport 0) and `api::m07_pipeline::prepare_view` (split viewports
//! through `EngineState::prepare_pipeline_view`) are compiled only for wasm32 and need a GPU,
//! so their call order is pinned from the source text at the end of this file. The pure
//! preparation core and the removal plan run natively.

use std::collections::{BTreeMap, BTreeSet};

use raa_core::RaaError;
use raa_math::{Aabb, Mat4, Vec3};

mod foliage_state {
    pub use raa_js::foliage_state::*;
}

#[allow(dead_code)]
#[path = "../src/api/m38_foliage.rs"]
mod m38;

use foliage_state::{
    FOLIAGE_INVALID, FoliageGroupConfig, FoliageGroupRecord, FoliageLevel, FoliagePhase,
    FoliagePrepareRecipe, FoliageStateStore, fail_foliage_record, foliage_visibility,
    prepare_foliage_record,
};
use m38::{FoliageRemovalStep, plan_removal};

const GROUP: &str = "tree";
const ROOT: u32 = 10;
const SOURCES: [u32; 2] = [1, 2];
const MATERIAL: u32 = 41;
const MESH: u32 = 7;
const PROXY: u32 = 8;

fn roots() -> BTreeSet<u32> {
    [ROOT].into_iter().collect()
}

fn live() -> BTreeMap<u32, bool> {
    SOURCES.iter().map(|&handle| (handle, true)).collect()
}

/// Registers `GROUP` over `asset` at `level`, with both live source draws.
fn group_over(store: &mut FoliageStateStore, asset: u32, level: FoliageLevel) {
    let config = FoliageGroupConfig {
        asset,
        full: SOURCES.to_vec(),
        root: ROOT,
        switch_pixels: 100.0,
        level,
    };
    store
        .groups
        .set(GROUP, Some(config), &roots(), &live())
        .expect("a valid group");
}

/// A camera at `eye` looking at a unit box at the origin, on a 1000 pixel tall viewport: the
/// projected diameter is 1500 / distance pixels, so a distance of 15 is 100 px.
fn recipe_at(eye: Vec3) -> FoliagePrepareRecipe {
    FoliagePrepareRecipe {
        eye,
        target: Vec3::ZERO,
        up: Vec3::Y,
        fov_y_rad: 60.0_f32.to_radians(),
        near: 0.1,
        ortho_height: None,
        viewport: (1000, 1000),
        root_world: Mat4::IDENTITY,
        bounds: Aabb {
            min: Vec3::splat(-0.5),
            max: Vec3::splat(0.5),
        },
    }
}

fn group_record(store: &mut FoliageStateStore) -> &mut FoliageGroupRecord {
    store
        .groups
        .get_mut(GROUP)
        .expect("the group is registered")
}

/// One preparation of `GROUP` for `view`, in the order the wasm loop uses: the asset is read
/// first (`None` when it was never published or has been removed), then the group record is
/// prepared against it. Returns the level the view selected.
fn prepare(store: &mut FoliageStateStore, view: u8, eye: Vec3) -> Result<u8, RaaError> {
    let asset = store.groups.get(GROUP).map(|record| record.config.asset);
    let entry = asset.and_then(|id| store.assets.get(id).cloned());
    prepare_foliage_record(group_record(store), view, &recipe_at(eye), entry.as_ref())
}

/// Whether the group's state draws its source and its proxy instances, as `(source, proxy)`.
fn drawn(store: &FoliageStateStore) -> (bool, bool) {
    let state = store.state(GROUP).expect("the group has a state");
    let visible = foliage_visibility(state.phase, state.level);
    (visible.source, visible.proxy)
}

/// A published, ready asset with mesh `MESH` and proxy `PROXY`; returns its id.
fn ready_asset(store: &mut FoliageStateStore) -> u32 {
    let asset = store.assets.begin(1, 0, 16, MATERIAL);
    assert!(store.assets.complete(asset, 1, MESH, PROXY, 512));
    asset
}

#[test]
fn a_missing_asset_keeps_every_source_visible() {
    for level in [FoliageLevel::Auto, FoliageLevel::Full, FoliageLevel::Proxy] {
        let mut store = FoliageStateStore::new();
        group_over(&mut store, 99, level);
        let selected = prepare(&mut store, 0, Vec3::new(0.0, 0.0, 100.0)).expect("view 0");
        assert_eq!(
            selected, 0,
            "{level:?}: a missing asset never selects the proxy"
        );
        let state = store.state(GROUP).expect("the group has a state");
        assert_eq!(state.phase, FoliagePhase::Failed);
        assert_eq!(state.asset, None);
        assert_eq!(
            state.error.as_deref(),
            Some("foliage asset 99 is not published")
        );
        assert_eq!(drawn(&store), (true, false));
    }
}

#[test]
fn a_pending_or_failed_asset_keeps_the_source_even_when_the_proxy_is_forced() {
    for failed in [false, true] {
        let mut store = FoliageStateStore::new();
        let asset = store.assets.begin(1, 0, 16, MATERIAL);
        if failed {
            assert!(store.assets.fail(asset, 1, "bake failed"));
        }
        group_over(&mut store, asset, FoliageLevel::Proxy);
        let selected = prepare(&mut store, 0, Vec3::new(0.0, 0.0, 100.0)).expect("view 0");
        assert_eq!(
            selected, 0,
            "a generation that is not ready never selects the proxy"
        );
        assert_eq!(drawn(&store), (true, false));
        let expected = if failed {
            FoliagePhase::Failed
        } else {
            FoliagePhase::Pending
        };
        assert_eq!(store.state(GROUP).map(|state| state.phase), Some(expected));
    }
}

#[test]
fn only_a_ready_asset_at_the_proxy_level_swaps_the_source_out() {
    let cases = [
        (FoliageLevel::Auto, 100.0, 1),
        (FoliageLevel::Auto, 5.0, 0),
        (FoliageLevel::Proxy, 5.0, 1),
        (FoliageLevel::Full, 100.0, 0),
    ];
    for (level, distance, expected) in cases {
        let mut store = FoliageStateStore::new();
        let asset = ready_asset(&mut store);
        group_over(&mut store, asset, level);
        let selected = prepare(&mut store, 0, Vec3::new(0.0, 0.0, distance)).expect("view 0");
        assert_eq!(selected, expected, "{level:?} at distance {distance}");
        let (source, proxy) = drawn(&store);
        assert_eq!((source, proxy), (expected == 0, expected == 1));
        assert!(
            !(source && proxy),
            "the source and the proxy are never drawn together"
        );
    }
}

#[test]
fn split_views_keep_their_own_level_record() {
    let mut store = FoliageStateStore::new();
    let asset = ready_asset(&mut store);
    group_over(&mut store, asset, FoliageLevel::Auto);
    // View 0 is far, so it selects the proxy. At 100 px it keeps the proxy it drew (hysteresis
    // holds it up to 110 px), while view 1, which never drew the proxy, switches to it only
    // below 90 px and so stays on the source.
    assert_eq!(prepare(&mut store, 0, Vec3::new(0.0, 0.0, 100.0)), Ok(1));
    assert_eq!(prepare(&mut store, 0, Vec3::new(0.0, 0.0, 15.0)), Ok(1));
    assert_eq!(prepare(&mut store, 1, Vec3::new(0.0, 0.0, 15.0)), Ok(0));
    let views = group_record(&mut store).views.clone();
    assert_eq!(views, BTreeMap::from([(0, 1), (1, 0)]));
}

#[test]
fn an_out_of_range_view_is_refused_without_mutation() {
    let mut store = FoliageStateStore::new();
    let asset = ready_asset(&mut store);
    group_over(&mut store, asset, FoliageLevel::Auto);
    let before = store
        .groups
        .get(GROUP)
        .cloned()
        .expect("the group is registered");
    let refused = prepare(&mut store, 4, Vec3::new(0.0, 0.0, 100.0)).expect_err("view 4");
    assert_eq!(refused.code(), FOLIAGE_INVALID);
    let failed = fail_foliage_record(group_record(&mut store), 4, "unused").expect_err("view 4");
    assert_eq!(failed.code(), FOLIAGE_INVALID);
    assert_eq!(
        store.groups.get(GROUP),
        Some(&before),
        "a refused view changes nothing"
    );
}

#[test]
fn a_failed_resolution_keeps_every_source_draw_and_its_diagnostic() {
    let mut store = FoliageStateStore::new();
    let asset = ready_asset(&mut store);
    group_over(&mut store, asset, FoliageLevel::Proxy);
    // A removed root or source instance is recorded this way by the wasm loop.
    fail_foliage_record(group_record(&mut store), 0, "source instance 2 is not live")
        .expect("view 0 is valid");
    assert_eq!(drawn(&store), (true, false));
    assert_eq!(
        store.state(GROUP).and_then(|state| state.error.as_deref()),
        Some("source instance 2 is not live")
    );
}

#[test]
fn the_level_follows_distance_and_not_the_camera_direction() {
    for (distance, expected) in [(100.0_f32, 1_u8), (5.0, 0)] {
        let mut levels = Vec::new();
        for direction in [Vec3::Z, Vec3::X, -Vec3::Z] {
            let mut store = FoliageStateStore::new();
            let asset = ready_asset(&mut store);
            group_over(&mut store, asset, FoliageLevel::Auto);
            levels.push(prepare(&mut store, 0, direction * distance).expect("view 0"));
        }
        assert_eq!(levels, vec![expected; 3], "distance {distance}");
    }
}

#[test]
fn teardown_restores_the_sources_before_the_proxy_and_a_removed_asset_hides_nothing() {
    let mut store = FoliageStateStore::new();
    let asset = ready_asset(&mut store);
    group_over(&mut store, asset, FoliageLevel::Proxy);
    assert_eq!(prepare(&mut store, 0, Vec3::new(0.0, 0.0, 100.0)), Ok(1));
    assert_eq!(drawn(&store), (false, true));

    let removal = plan_removal(&mut store, asset).expect("the published asset is removed");
    // The dependent group detaches first. The caller restores the saved source draws before
    // the proxy goes; the material association and the owned mesh follow.
    assert_eq!(removal.restored, vec![(1, true), (2, true)]);
    assert_eq!(
        removal.steps(),
        [
            FoliageRemovalStep::DetachGroups,
            FoliageRemovalStep::RemoveProxy(PROXY),
            FoliageRemovalStep::DetachAssociation(MATERIAL),
            FoliageRemovalStep::ReleaseMesh(MESH),
        ]
    );
    assert!(
        store.groups.get(GROUP).is_none(),
        "no group keeps the released asset"
    );

    // A group registered again over the removed id finds no asset: its sources stay drawn.
    group_over(&mut store, asset, FoliageLevel::Proxy);
    assert_eq!(prepare(&mut store, 0, Vec3::new(0.0, 0.0, 100.0)), Ok(0));
    assert_eq!(drawn(&store), (true, false));
}

/// The engine and pipeline sources, read at compile time for the call-order checks.
const ENGINE: &str = include_str!("../src/engine.rs");
const PIPELINE: &str = include_str!("../src/api/m07_pipeline.rs");
const STATE: &str = include_str!("../src/state.rs");

/// The text of the function whose header contains `header`, from the header through the
/// brace that closes its body.
fn body<'a>(source: &'a str, header: &str) -> &'a str {
    let start = source
        .find(header)
        .unwrap_or_else(|| panic!("`{header}` is missing"));
    let open = start + source[start..].find('{').expect("a function body");
    let mut depth = 0_usize;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("`{header}` is not closed")
}

/// Asserts that every step occurs in `text`, each one after the step before it.
fn assert_in_order(text: &str, steps: &[&str]) {
    let mut last = 0;
    for step in steps {
        let found = text
            .find(step)
            .unwrap_or_else(|| panic!("`{step}` is missing"));
        assert!(found > last, "`{step}` must come after the steps before it");
        last = found;
    }
}

#[test]
fn update_prepares_viewport_zero_foliage_beside_billboards_before_lit() {
    let update = body(
        ENGINE,
        "fn update(&mut self, gpu: &GpuContext, input: &FrameInput)",
    );
    assert_in_order(
        update,
        &[
            "s.rebuild_dirty_groups()",
            "prepare_sky_view(s, 0, size)",
            "s.prepare_billboard_groups(0)",
            "s.prepare_foliage_groups(0)",
            "crate::trace::prepare_view(s, 0",
            "s.scene_invocation.is_discontinuous(0)",
            ".prepare(gpu, &mut s.scene",
        ],
    );
    assert_eq!(update.matches("prepare_foliage_groups(").count(), 1);
}

#[test]
fn split_views_prepare_foliage_after_their_camera_and_before_lit() {
    let view = body(PIPELINE, "pub(crate) fn prepare_view(");
    assert_in_order(
        view,
        &[
            "st.scene.camera.set_viewport(size.0, size.1)",
            "let result = (|| {",
            "super::m36_billboard_groups::prepare_billboard_groups(",
            "st.prepare_foliage_groups(u8::try_from(index).unwrap_or(u8::MAX))",
            "st.scene_invocation.is_discontinuous(index)",
            "st.lit.prepare(",
            "st.scene.camera = original;",
        ],
    );
    assert_eq!(view.matches("prepare_foliage_groups(").count(), 1);
}

#[test]
fn the_split_callback_selects_nothing_before_prepare_view_installs_the_camera() {
    let init = body(ENGINE, "fn init(");
    let start = init.find("set_view_prepare(").expect("the split callback");
    let end = start
        + init[start..]
            .find("Pipeline::install(")
            .expect("the install");
    let callback = &init[start..end];
    assert!(callback.contains("st.prepare_pipeline_view(gpu, index, size)"));
    assert!(!callback.contains("prepare_foliage_groups"));
    let pipeline_view = body(PIPELINE, "fn prepare_pipeline_view(");
    assert!(pipeline_view.contains("prepare_view(self, gpu, index, size)"));
}

#[test]
fn the_engine_hook_forwards_to_the_foliage_preparation() {
    let hook = body(
        STATE,
        "pub(crate) fn prepare_foliage_groups(&mut self, view: u8)",
    );
    assert!(hook.contains("crate::foliage_state::prepare_foliage_groups(self, view)"));
}
