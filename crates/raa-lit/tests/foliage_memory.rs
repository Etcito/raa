//! Host-heap measurement of the foliage preparation path at the size of the 41-part sample
//! tree (card 'Stop the out-of-memory trap in raa-js foliage:tree:prepare').
//!
//! The source is synthetic but sized from a measured evaluation of the canonical sample:
//! `procgen-eval 3` of MAMA@d6e36fc on the procedural graph of sdfm
//! `samples/m6-leaf-tree.sdfm.json` gives one oak whose complete realization holds 1546
//! leaf members of one 346-triangle leaf mesh (534 916 leaf triangles) and 995 bark cone
//! segments. The cone tessellation is not part of that output; this test gives each segment
//! 128 triangles. The triangles are split into 41 parts (one wood part, 40 leaf parts), each
//! with its own source tag, like the 41 source instances of the preparation.
//!
//! A counting global allocator records, for every CPU stage of the bake path, the peak
//! heap growth above the stage start and the bytes the stage's result keeps. The numbers are
//! printed as measured facts (`cargo test -p raa-lit --test foliage_memory -- --nocapture`);
//! the assertions check structure only (one record per run, windowed streams), never a byte
//! threshold. GPU stages are not executed here: the bins readback is printed as computed
//! sizes. Everything runs in one test so no other test thread disturbs the counters.

#![cfg(not(target_arch = "wasm32"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use raa_lit::foliage_appearance::pack_samples;
use raa_lit::foliage_bins::{bin_view_table, tiles_per_axis};
use raa_lit::foliage_optimize::assemble_optimized_foliage;
use raa_lit::foliage_poisson::initialize_foliage;
use raa_lit::foliage_reference::{
    REFERENCE_STREAM_CHUNK_TRIANGLES, REFERENCE_STREAM0_CORNER_BYTES,
    REFERENCE_STREAM1_CORNER_BYTES, encode_reference_streams, reference_draw_plan,
};
use raa_lit::foliage_resources::{FoliageSource, SourceTriangle};
use raa_lit::foliage_sampling::sample_foliage;
use raa_lit::foliage_stochastic::build_stochastic_foliage;
use raa_lit::foliage_volume::FoliageVolumeAccumulator;
use raa_math::Mat4;

/// Live and peak heap bytes of this test binary.
struct PeakAlloc;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(bytes: usize) {
    let now = CURRENT.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every call forwards its layout and pointer unchanged to `System`; the counters only observe sizes. (test: sample_tree_bake_stages_fit_the_heap_one_stage_at_a_time)
unsafe impl GlobalAlloc for PeakAlloc {
    // SAFETY: forwards the caller's arguments unchanged to `System`, whose contract the trait's own contract matches (test: sample_tree_bake_stages_fit_the_heap_one_stage_at_a_time)
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwards the caller layout unchanged. (test: sample_tree_bake_stages_fit_the_heap_one_stage_at_a_time)
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    // SAFETY: forwards the caller's arguments unchanged to `System`, whose contract the trait's own contract matches (test: sample_tree_bake_stages_fit_the_heap_one_stage_at_a_time)
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwards the caller layout unchanged. (test: sample_tree_bake_stages_fit_the_heap_one_stage_at_a_time)
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    // SAFETY: forwards the caller's arguments unchanged to `System`, whose contract the trait's own contract matches (test: sample_tree_bake_stages_fit_the_heap_one_stage_at_a_time)
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwards the caller arguments unchanged. (test: sample_tree_bake_stages_fit_the_heap_one_stage_at_a_time)
        let out = unsafe { System.realloc(ptr, layout, new_size) };
        if !out.is_null() {
            // A moving realloc briefly holds both blocks; count that as the peak.
            grew(new_size);
            CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        out
    }

    // SAFETY: forwards the caller's arguments unchanged to `System`, whose contract the trait's own contract matches (test: sample_tree_bake_stages_fit_the_heap_one_stage_at_a_time)
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: forwards the caller arguments unchanged. (test: sample_tree_bake_stages_fit_the_heap_one_stage_at_a_time)
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc;

const LEAVES: usize = 1546;
const LEAF_TRIANGLES: usize = 346;
const WOOD_SEGMENTS: usize = 995;
const WOOD_SEGMENT_TRIANGLES: usize = 128;
const PARTS: u32 = 41;
const LEAF_MATERIAL: u32 = 7;
const BARK_MATERIAL: u32 = 3;
/// MAMA `FOLIAGE_DEFAULTS.foliage_triangles` and `foliage_views`.
const COUNT: u32 = 4096;
const VIEWS: u32 = 96;
/// The native volume extent and raa-js `FOLIAGE_TEXTURE_RESOLUTION`.
const VOLUME: u32 = 32;
const TEXTURE: u32 = 128;

/// Runs `f` and returns its result with the peak heap growth above the stage start and the
/// heap bytes still held after it (its result).
fn stage<T>(name: &str, f: impl FnOnce() -> T) -> (T, usize, usize) {
    let start = CURRENT.load(Ordering::Relaxed);
    PEAK.store(start, Ordering::Relaxed);
    let out = f();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(start);
    let kept = CURRENT.load(Ordering::Relaxed).saturating_sub(start);
    println!("foliage memory (measured): {name:<34} peak {peak:>11} B  kept {kept:>11} B");
    (out, peak, kept)
}

/// A deterministic hash stream for the synthetic placement.
fn next(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*state >> 8) as f32 / (1u32 << 24) as f32
}

fn triangle(positions: [[f32; 3]; 3], material: u32, tag: u32, foliage: bool) -> SourceTriangle {
    SourceTriangle {
        positions,
        normals: [[0.0, 0.0, 1.0]; 3],
        uv: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        material,
        source_tag: tag,
        foliage,
    }
}

/// The 41-part tree: part 0 is the wood, parts 1..=40 the leaves, in part order like the
/// ordered draws of `LitHandle::foliage_source`. The list is reserved once at its exact
/// size, fallibly, as there.
fn sample_tree() -> FoliageSource {
    let mut state = 0x5eed_u32;
    let mut triangles = Vec::new();
    triangles
        .try_reserve_exact(WOOD_SEGMENTS * WOOD_SEGMENT_TRIANGLES + LEAVES * LEAF_TRIANGLES)
        .expect("the snapshot reserves");
    for segment in 0..WOOD_SEGMENTS {
        let height = 12.0 * segment as f32 / WOOD_SEGMENTS as f32;
        let (x, z) = (next(&mut state) * 6.0 - 3.0, next(&mut state) * 6.0 - 3.0);
        for t in 0..WOOD_SEGMENT_TRIANGLES {
            let a = t as f32 / WOOD_SEGMENT_TRIANGLES as f32 * std::f32::consts::TAU;
            let b = a + std::f32::consts::TAU / WOOD_SEGMENT_TRIANGLES as f32;
            let p =
                |angle: f32, dy: f32| [x + 0.1 * angle.cos(), height + dy, z + 0.1 * angle.sin()];
            triangles.push(triangle(
                [p(a, 0.0), p(b, 0.0), p(a, 0.3)],
                BARK_MATERIAL,
                0,
                false,
            ));
        }
    }
    for leaf in 0..LEAVES {
        let tag = 1 + (leaf * (PARTS as usize - 1) / LEAVES) as u32;
        let centre = [
            next(&mut state) * 8.0 - 4.0,
            4.0 + next(&mut state) * 9.0,
            next(&mut state) * 8.0 - 4.0,
        ];
        for t in 0..LEAF_TRIANGLES {
            let a = t as f32 / LEAF_TRIANGLES as f32 * std::f32::consts::TAU;
            let r = 0.05 + 0.1 * next(&mut state);
            let corner = |angle: f32| {
                [
                    centre[0] + r * angle.cos(),
                    centre[1] + r * angle.sin(),
                    centre[2],
                ]
            };
            triangles.push(triangle(
                [centre, corner(a), corner(a + 0.02)],
                LEAF_MATERIAL,
                tag,
                true,
            ));
        }
    }
    FoliageSource {
        triangles,
        min: [-4.2, 0.0, -4.2],
        max: [4.2, 13.4, 4.2],
        root_world: Mat4::IDENTITY,
        generation: 1,
    }
}

#[test]
fn sample_tree_bake_stages_fit_the_heap_one_stage_at_a_time() {
    let (source, source_peak, source_kept) = stage("source snapshot (41 parts)", sample_tree);
    let triangles = source.triangles.len();
    assert_eq!(
        triangles,
        WOOD_SEGMENTS * WOOD_SEGMENT_TRIANGLES + LEAVES * LEAF_TRIANGLES
    );
    let triangle_bytes = std::mem::size_of::<SourceTriangle>();
    assert_eq!(
        source.triangles.capacity(),
        triangles,
        "the snapshot is reserved at its exact size"
    );
    println!(
        "foliage memory (computed): source triangles {triangles}, {triangle_bytes} B each; \
         exact snapshot {} B, base push-grown snapshot peak below {} B \
         (measured: peak {source_peak} B, kept {source_kept} B)",
        triangles * triangle_bytes,
        triangles * 3 * triangle_bytes
    );

    let (samples, _, _) = stage("sampling (sample_foliage)", || {
        sample_foliage(&source, COUNT, 11).expect("sampling succeeds")
    });
    assert_eq!(samples.len(), COUNT as usize);

    let (seed, _, _) = stage("Poisson init (initialize_foliage)", || {
        initialize_foliage(&source, COUNT, 11).expect("Poisson init succeeds")
    });
    assert_eq!(seed.len(), COUNT as usize);

    let (inputs, _, _) = stage("appearance inputs (pack_samples)", || {
        pack_samples(&source, &samples).expect("packing succeeds")
    });
    drop(inputs);
    // The unlit appearance pass needs the GPU; a uniform albedo stands in for its output.
    let albedo = vec![[0.3f32, 0.5, 0.2]; samples.len()];

    let (volume, _, _) = stage("volume (accumulate + finish)", || {
        let mut volume =
            FoliageVolumeAccumulator::new(source.min, source.max, VOLUME).expect("volume box");
        for (sample, albedo) in samples.iter().zip(&albedo) {
            volume
                .add(sample.position, *albedo, sample.represented_area)
                .expect("volume sample");
        }
        volume.finish()
    });
    assert_eq!(volume.len(), 4 * (VOLUME as usize).pow(3));
    drop(volume);

    let (bake, _, _) = stage("stochastic assembly", || {
        build_stochastic_foliage(&source, &samples, &albedo, VOLUME).expect("assembly succeeds")
    });
    assert_eq!(
        bake.mesh.positions.capacity(),
        3 * (WOOD_SEGMENTS * WOOD_SEGMENT_TRIANGLES + samples.len()),
        "the mesh is reserved once at its exact size"
    );
    drop(bake);

    let (plan, _, _) = stage("reference draw plan (runs)", || {
        reference_draw_plan(&source).expect("plan succeeds")
    });
    assert_eq!(plan.material_ids, vec![BARK_MATERIAL, LEAF_MATERIAL]);
    assert_eq!(
        plan.runs.len(),
        PARTS as usize,
        "one run per part and material"
    );
    assert_eq!(
        plan.instances.len(),
        plan.runs.len() * raa_lit::INSTANCE_GPU_SIZE
    );
    assert_eq!(
        plan.runs
            .iter()
            .map(|run| run.count as usize)
            .sum::<usize>(),
        triangles
    );
    let (streamed, _, _) = stage("reference vertex streams (windows)", || {
        let (mut window0, mut window1) = (Vec::new(), Vec::new());
        let mut total = 0usize;
        let mut first = 0usize;
        while first < triangles {
            let count = REFERENCE_STREAM_CHUNK_TRIANGLES.min(triangles - first);
            encode_reference_streams(&source, first, count, &mut window0, &mut window1)
                .expect("window encodes");
            total += window0.len() + window1.len();
            first += count;
        }
        total
    });
    let corner_bytes = REFERENCE_STREAM0_CORNER_BYTES + REFERENCE_STREAM1_CORNER_BYTES;
    assert_eq!(streamed, triangles * 3 * corner_bytes);
    println!(
        "foliage memory (computed): reference host bytes, base per-triangle layout {} B (streams, \
         instance, extra, id, previous per triangle); now plan {} B + one window of {} B",
        triangles
            * (3 * corner_bytes
                + raa_lit::INSTANCE_GPU_SIZE
                + raa_lit::INSTANCE_EXTRA_GPU_SIZE
                + 4
                + 64),
        plan.heap_bytes(),
        REFERENCE_STREAM_CHUNK_TRIANGLES * 3 * corner_bytes
    );
    drop(plan);

    let views: Vec<([f32; 16], [f32; 2])> = (0..VIEWS)
        .map(|_| (Mat4::IDENTITY.to_cols_array(), [0.1, 40.0]))
        .collect();
    let (table, _, _) = stage("bins view table", || {
        bin_view_table(&views).expect("table succeeds")
    });
    assert_eq!(table.len(), 128 * 80);
    let side = tiles_per_axis(TEXTURE) as usize;
    let tiles = side * side * VIEWS as usize;
    for capacity in [32usize, 1024, COUNT as usize] {
        println!(
            "foliage memory (computed): bins readback at capacity {capacity}: full records {} B (base), \
             now count words {} B through windows of at most 1048576 B (computed, GPU not run)",
            tiles * (capacity + 1) * 4,
            tiles * 4
        );
    }

    let (optimized, _, _) = stage("optimization assembly", || {
        assemble_optimized_foliage(&source, &seed, &samples, &albedo, VOLUME)
            .expect("assembly succeeds")
    });
    assert_eq!(optimized.foliage_triangles, COUNT);
    let records = seed.len() * raa_lit::foliage_update::TRIANGLE_RECORD_STRIDE;
    println!(
        "foliage memory (computed): optimization triangle records {records} B (reserved once)"
    );
    let impostor = raa_lit::foliage_bake::check_impostor_allocation(triangles)
        .expect("the impostor pre-check accepts the sample tree");
    println!(
        "foliage memory (computed): impostor hull working set {impostor} B, base pre-check \
         layout {} B",
        triangles
            * (3 * corner_bytes
                + raa_lit::INSTANCE_GPU_SIZE
                + raa_lit::INSTANCE_EXTRA_GPU_SIZE
                + 4
                + 64
                + 8)
    );
}
