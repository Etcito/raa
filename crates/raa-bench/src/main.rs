//! Allocation benchmark over the CPU frame path.
//!
//! `frame-allocs` builds a GPU-free scene of `instances` instances under
//! `ceil(instances / 100)` parent nodes, runs `warmup` frames, resets the counters of
//! [`raa_alloc_count::CountingAlloc`], runs `frames` measured frames and prints one JSON
//! line with the allocation count, the allocated bytes and the median CPU time per frame.

use std::process::ExitCode;
use std::time::Instant;

use raa_core::{MaterialId, MeshId};
use raa_math::{Quat, Vec3};
use raa_render::FramePrep;
use raa_scene::{NodeId, RenderList, Scene, Transform};

#[global_allocator]
static A: raa_alloc_count::CountingAlloc = raa_alloc_count::CountingAlloc;

const USAGE: &str = "usage: frame-allocs [--instances N] [--warmup N] [--frames N]\n  \
--instances  instances in the scene (default 1000)\n  \
--warmup     frames run before the counters are reset (default 120)\n  \
--frames     measured frames, at least 1 (default 600)";

/// Children per parent node.
const CHILDREN_PER_PARENT: usize = 100;
/// Side of the square grid of children under one parent.
const CHILD_GRID: usize = 10;
/// Distance between neighbouring children, in metres.
const CHILD_SPACING: f32 = 1.0;
/// Distance between neighbouring parents, in metres.
const PARENT_SPACING: f32 = 12.0;

struct Args {
    instances: usize,
    warmup: usize,
    frames: usize,
}

fn usage_error(msg: &str) -> ExitCode {
    eprintln!("frame-allocs: {msg}\n{USAGE}");
    ExitCode::from(2)
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        instances: 1000,
        warmup: 120,
        frames: 600,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let slot = match flag.as_str() {
            "--instances" => &mut args.instances,
            "--warmup" => &mut args.warmup,
            "--frames" => &mut args.frames,
            other => return Err(format!("unknown argument `{other}`")),
        };
        let value = it
            .next()
            .ok_or_else(|| format!("missing value for `{flag}`"))?;
        *slot = value
            .parse::<usize>()
            .map_err(|_| format!("invalid value `{value}` for `{flag}`"))?;
    }
    if args.frames == 0 {
        return Err("`--frames` must be at least 1".to_string());
    }
    if u32::try_from(args.instances).is_err() {
        return Err(format!("`--instances` must be at most {}", u32::MAX));
    }
    Ok(args)
}

/// Builds the scene and returns it with each parent's id and translation.
fn build_scene(instances: usize) -> Result<(Scene, Vec<(NodeId, Vec3)>), String> {
    let parents = instances.div_ceil(CHILDREN_PER_PARENT);
    let mut scene = Scene::with_capacity(parents + instances, instances);
    let mut parent_nodes = Vec::with_capacity(parents);
    // Parents on a square grid centred on the origin.
    let side = (1..).find(|s| s * s >= parents).unwrap_or(1);
    let parent_offset = (side as f32 - 1.0) * 0.5;
    let child_offset = (CHILD_GRID as f32 - 1.0) * 0.5;
    for p in 0..parents {
        let translation = Vec3::new(
            ((p % side) as f32 - parent_offset) * PARENT_SPACING,
            0.0,
            ((p / side) as f32 - parent_offset) * PARENT_SPACING,
        );
        let local = Transform::from_trs(translation, Quat::IDENTITY, Vec3::ONE);
        let parent = scene.add_node(None, local).map_err(|e| e.to_string())?;
        parent_nodes.push((parent, translation));
        let first = p * CHILDREN_PER_PARENT;
        let last = (first + CHILDREN_PER_PARENT).min(instances);
        for i in first..last {
            let c = i - first;
            let translation = Vec3::new(
                ((c % CHILD_GRID) as f32 - child_offset) * CHILD_SPACING,
                0.0,
                ((c / CHILD_GRID) as f32 - child_offset) * CHILD_SPACING,
            );
            let local = Transform::from_trs(translation, Quat::IDENTITY, Vec3::splat(0.4));
            let child = scene
                .add_node(Some(parent), local)
                .map_err(|e| e.to_string())?;
            scene
                .add_instance(child, MeshId((i % 4) as u32), MaterialId((i % 8) as u32))
                .map_err(|e| e.to_string())?;
        }
    }
    Ok((scene, parent_nodes))
}

/// One frame of the CPU frame path. Errors carry the message to print.
fn frame(
    f: usize,
    scene: &mut Scene,
    parents: &[(NodeId, Vec3)],
    list: &mut RenderList,
    prep: &mut FramePrep,
) -> Result<(), String> {
    let rotation = Quat::from_rotation_y(f as f32 * 0.01);
    for &(id, translation) in parents {
        scene
            .set_local(id, Transform::from_trs(translation, rotation, Vec3::ONE))
            .map_err(|e| e.to_string())?;
    }
    scene.update_world();
    scene.build_render_list(list);
    prep.prepare_cpu(scene, list, false)
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => return usage_error(&msg),
    };
    let (mut scene, parents) = match build_scene(args.instances) {
        Ok(s) => s,
        Err(msg) => {
            eprintln!("frame-allocs: {msg}");
            return ExitCode::from(1);
        }
    };
    let mut list = RenderList::with_capacity(args.instances);
    // parse_args guarantees instances fits in u32.
    let mut prep = FramePrep::new(args.instances.max(1) as u32);
    let mut timings: Vec<f64> = vec![0.0; args.frames];

    for f in 0..args.warmup {
        if let Err(msg) = frame(f, &mut scene, &parents, &mut list, &mut prep) {
            eprintln!("frame-allocs: {msg}");
            return ExitCode::from(1);
        }
    }
    raa_alloc_count::reset();
    for (k, slot) in timings.iter_mut().enumerate() {
        let start = Instant::now();
        let result = frame(args.warmup + k, &mut scene, &parents, &mut list, &mut prep);
        *slot = start.elapsed().as_secs_f64() * 1000.0;
        if let Err(msg) = result {
            eprintln!("frame-allocs: {msg}");
            return ExitCode::from(1);
        }
    }
    let allocations = raa_alloc_count::allocations();
    let bytes = raa_alloc_count::bytes();

    timings.sort_unstable_by(|a, b| a.total_cmp(b));
    let mid = timings.len() / 2;
    let median = if timings.len().is_multiple_of(2) {
        (timings[mid - 1] + timings[mid]) * 0.5
    } else {
        timings[mid]
    };
    println!(
        "{{\"instances\":{},\"warmup\":{},\"frames\":{},\"allocations\":{allocations},\"bytes\":{bytes},\"prepCpuMsMedian\":{median}}}",
        args.instances, args.warmup, args.frames
    );
    ExitCode::SUCCESS
}
