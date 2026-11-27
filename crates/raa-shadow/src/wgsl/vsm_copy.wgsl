// VSM page writes (crates/raa-shadow/src/vsm_gpu.rs): one instanced quad per job into a pool
// layer of the shadow array. Kind 0 copies a 128 x 128 page rendered in a window (cascade
// layer src_layer at texel src) to the job's slot rect at dst; kind 1 writes the page table
// of level `level` (entry / 2^24 as depth, exact in Depth32Float).

struct VsmJob {
    src: vec2<u32>,
    dst: vec2<u32>,
    src_layer: u32,
    kind: u32,
    level: u32,
    reserved: u32,
}

@group(0) @binding(0) var scratch: texture_depth_2d_array;
@group(0) @binding(1) var<storage, read> jobs: array<VsmJob>;
@group(0) @binding(2) var<storage, read> tables: array<u32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) job: u32,
}

@vertex
fn vs_page(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let res = f32(textureDimensions(scratch).x);
    let px = vec2<f32>(jobs[ii].dst) + corners[vi] * 128.0;
    var out: VsOut;
    out.pos = vec4<f32>(px.x / res * 2.0 - 1.0, 1.0 - px.y / res * 2.0, 0.0, 1.0);
    out.job = ii;
    return out;
}

@fragment
fn fs_page(in: VsOut) -> @builtin(frag_depth) f32 {
    let job = jobs[in.job];
    let local = min(vec2<u32>(in.pos.xy) - job.dst, vec2<u32>(127u));
    if (job.kind == 0u) {
        return textureLoad(scratch, vec2<i32>(job.src + local), i32(job.src_layer), 0);
    }
    let entry = tables[job.level * 16384u + local.y * 128u + local.x];
    return f32(entry) / 16777216.0;
}
