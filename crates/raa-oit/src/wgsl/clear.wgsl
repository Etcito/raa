// raa-oit head clear: every head becomes the end marker and the node counter restarts at 0.
// Dispatched with ceil(count / 64) workgroups, split over x and y when that exceeds the
// per-dimension limit; the flat index covers both.

struct ClearParams {
    count: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

@group(0) @binding(0) var<storage, read_write> heads: array<u32>;
@group(0) @binding(1) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(2) var<uniform> params: ClearParams;

@compute @workgroup_size(64)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let i = gid.y * groups.x * 64u + gid.x;
    if (i == 0u) {
        atomicStore(&counter, 0u);
    }
    if (i < params.count && i < arrayLength(&heads)) {
        heads[i] = 0xffffffffu;
    }
}
