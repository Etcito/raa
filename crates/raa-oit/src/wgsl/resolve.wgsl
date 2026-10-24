// raa-oit resolve: per pixel, keep the max_fragments nearest nodes of the list (insertion
// sort on depth, nearest first), then composite them farther-first over hdr.src with
// dst = c * a + dst * (1 - a) (straight colour, premultiplied here). Same algorithm as
// src/resolve_cpu.rs.

struct OitNode {
    // pack2x16float(colour.rg).
    color_rg: u32,
    // pack2x16float(vec2(colour.b, alpha)).
    color_ba: u32,
    // Fragment depth in [0, 1].
    depth: f32,
    // Next node index, 0xffffffff = end.
    next: u32,
}

struct ResolveParams {
    width: u32,
    height: u32,
    max_fragments: u32,
    reserved: u32,
}

const OIT_END: u32 = 0xffffffffu;
const OIT_MAX_LIMIT: u32 = 32u;

@group(0) @binding(0) var<storage, read> heads: array<u32>;
@group(0) @binding(1) var<storage, read> nodes: array<OitNode>;
@group(0) @binding(2) var hdr_src: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: ResolveParams;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle: (-1, -1), (3, -1), (-1, 3).
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<u32>(frag.xy);
    let src = textureLoad(hdr_src, px, 0);
    if (px.x >= params.width || px.y >= params.height) {
        return src;
    }
    let pixel = px.y * params.width + px.x;
    if (pixel >= arrayLength(&heads)) {
        return src;
    }
    let head = heads[pixel];
    if (head == OIT_END) {
        return src;
    }

    let max_kept = clamp(params.max_fragments, 1u, OIT_MAX_LIMIT);
    let node_count = arrayLength(&nodes);
    var color: array<vec3<f32>, 32>;
    var alpha: array<f32, 32>;
    var depth: array<f32, 32>;
    var kept = 0u;
    var cur = head;
    var steps = 0u;
    // Stops at the end marker, an out-of-range index, or after node_count steps (cycle guard).
    loop {
        if (cur == OIT_END || cur >= node_count || steps > node_count) {
            break;
        }
        steps += 1u;
        let n = nodes[cur];
        cur = n.next;
        let rg = unpack2x16float(n.color_rg);
        let ba = unpack2x16float(n.color_ba);
        let d = n.depth;
        if (kept == max_kept) {
            if (d < depth[kept - 1u]) {
                kept -= 1u;
            } else {
                continue;
            }
        }
        // Insert after every kept node with depth <= d.
        var at = kept;
        loop {
            if (at == 0u || depth[at - 1u] <= d) {
                break;
            }
            color[at] = color[at - 1u];
            alpha[at] = alpha[at - 1u];
            depth[at] = depth[at - 1u];
            at -= 1u;
        }
        color[at] = vec3<f32>(rg, ba.x);
        alpha[at] = ba.y;
        depth[at] = d;
        kept += 1u;
    }

    var dst = src.rgb;
    var i = kept;
    loop {
        if (i == 0u) {
            break;
        }
        i -= 1u;
        let a = clamp(alpha[i], 0.0, 1.0);
        dst = color[i] * a + dst * (1.0 - a);
    }
    return vec4<f32>(dst, 1.0);
}
