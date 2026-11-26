// raa-oit resolve: per pixel, keep the max_fragments nearest nodes of the list (insertion
// sort on depth, nearest first), then composite them farther-first over hdr.src with
// dst = c * a + dst * (1 - a) (straight colour, premultiplied here). Same algorithm as
// src/resolve_cpu.rs.
//
// Lists of up to OIT_FAST nodes (and any list when max_fragments <= OIT_FAST) are sorted in
// registers; only longer lists use the general path and its 32-entry local arrays, whose
// dynamic indexing and zero-initialization go through scratch memory. Both paths apply the
// same comparisons and the same compositing operations in the same order (bit-identical).

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
const OIT_FAST: u32 = 8u;

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
    // Fast path: up to OIT_FAST nodes held in registers (constant indices only), same insertion
    // and eviction rules as the general path below; a longer list when max_kept > OIT_FAST falls
    // back to it from the head.
    let fast_cap = min(max_kept, OIT_FAST);
    var d0 = vec4<f32>(0.0);
    var d1 = vec4<f32>(0.0);
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var c0 = vec3<f32>(0.0);
    var c1 = vec3<f32>(0.0);
    var c2 = vec3<f32>(0.0);
    var c3 = vec3<f32>(0.0);
    var c4 = vec3<f32>(0.0);
    var c5 = vec3<f32>(0.0);
    var c6 = vec3<f32>(0.0);
    var c7 = vec3<f32>(0.0);
    var kept = 0u;
    var cur = head;
    var steps = 0u;
    var overflow = false;
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
        if (kept == fast_cap) {
            if (fast_cap < max_kept) {
                overflow = true;
                break;
            }
            let last = select(select(select(select(select(select(select(d0.x, d0.y, kept == 2u), d0.z, kept == 3u), d0.w, kept == 4u), d1.x, kept == 5u), d1.y, kept == 6u), d1.z, kept == 7u), d1.w, kept == 8u);
            if (d < last) {
                kept -= 1u;
            } else {
                continue;
            }
        }
        // Insert after every kept node with depth <= d (the general path's backward scan).
        var at = 0u;
        if (0u < kept && d0.x <= d) { at = 1u; }
        if (1u < kept && d0.y <= d) { at = 2u; }
        if (2u < kept && d0.z <= d) { at = 3u; }
        if (3u < kept && d0.w <= d) { at = 4u; }
        if (4u < kept && d1.x <= d) { at = 5u; }
        if (5u < kept && d1.y <= d) { at = 6u; }
        if (6u < kept && d1.z <= d) { at = 7u; }
        let color = vec3<f32>(rg, ba.x);
        if (7u > at) { d1.w = d1.z; a1.w = a1.z; c7 = c6; }
        if (6u > at) { d1.z = d1.y; a1.z = a1.y; c6 = c5; }
        if (5u > at) { d1.y = d1.x; a1.y = a1.x; c5 = c4; }
        if (4u > at) { d1.x = d0.w; a1.x = a0.w; c4 = c3; }
        if (3u > at) { d0.w = d0.z; a0.w = a0.z; c3 = c2; }
        if (2u > at) { d0.z = d0.y; a0.z = a0.y; c2 = c1; }
        if (1u > at) { d0.y = d0.x; a0.y = a0.x; c1 = c0; }
        if (0u == at) { d0.x = d; a0.x = ba.y; c0 = color; }
        if (1u == at) { d0.y = d; a0.y = ba.y; c1 = color; }
        if (2u == at) { d0.z = d; a0.z = ba.y; c2 = color; }
        if (3u == at) { d0.w = d; a0.w = ba.y; c3 = color; }
        if (4u == at) { d1.x = d; a1.x = ba.y; c4 = color; }
        if (5u == at) { d1.y = d; a1.y = ba.y; c5 = color; }
        if (6u == at) { d1.z = d; a1.z = ba.y; c6 = color; }
        if (7u == at) { d1.w = d; a1.w = ba.y; c7 = color; }
        kept += 1u;
    }
    if (!overflow) {
        var dst = src.rgb;
        if (7u < kept) {
            let a = clamp(a1.w, 0.0, 1.0);
            dst = c7 * a + dst * (1.0 - a);
        }
        if (6u < kept) {
            let a = clamp(a1.z, 0.0, 1.0);
            dst = c6 * a + dst * (1.0 - a);
        }
        if (5u < kept) {
            let a = clamp(a1.y, 0.0, 1.0);
            dst = c5 * a + dst * (1.0 - a);
        }
        if (4u < kept) {
            let a = clamp(a1.x, 0.0, 1.0);
            dst = c4 * a + dst * (1.0 - a);
        }
        if (3u < kept) {
            let a = clamp(a0.w, 0.0, 1.0);
            dst = c3 * a + dst * (1.0 - a);
        }
        if (2u < kept) {
            let a = clamp(a0.z, 0.0, 1.0);
            dst = c2 * a + dst * (1.0 - a);
        }
        if (1u < kept) {
            let a = clamp(a0.y, 0.0, 1.0);
            dst = c1 * a + dst * (1.0 - a);
        }
        if (0u < kept) {
            let a = clamp(a0.x, 0.0, 1.0);
            dst = c0 * a + dst * (1.0 - a);
        }
        return vec4<f32>(dst, 1.0);
    }

    return resolve_general(head, max_kept, node_count, src);
}

// General path: any list, up to OIT_MAX_LIMIT kept nodes in local arrays.
fn resolve_general(head: u32, max_kept: u32, node_count: u32, src: vec4<f32>) -> vec4<f32> {
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
