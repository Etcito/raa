// Edge-segment generation after vertex skinning (S18). The rest topology decides
// which vertices form a segment; every endpoint is read from the posed stream 0,
// so duplicated vertices and multiple instances keep their own posed endpoints.
// Each 48-byte record holds the two endpoints as S1 vertices:
// position[3], normal[3] (24 B) for the start, then the same for the end.
@group(0) @binding(0) var<storage, read> edges: array<u32>;
@group(0) @binding(1) var<storage, read> stream0: array<f32>;
@group(0) @binding(2) var<storage, read_write> segments: array<f32>;

@compute @workgroup_size(64)
fn skin_edges(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&edges) / 2u { return; }
    let a = edges[i * 2u] * 6u;
    let b = edges[i * 2u + 1u] * 6u;
    for (var c = 0u; c < 6u; c++) {
        segments[i * 12u + c] = stream0[a + c];
        segments[i * 12u + 6u + c] = stream0[b + c];
    }
}
