// `view.grid`: up to 16 float tiles in a grid with optional labels drawn in this file's own
// 5x7 font (crates/raa-graph/src/exec_ext/views.rs `view_grid_source`).

struct GridParams {
    columns: u32,
    count: u32,
    labels: u32,
    pad0: u32,
    label_words: array<vec4<u32>, 32>,
}

@group(0) @binding(0) var<uniform> params: GridParams;
@group(0) @binding(1) var t0: texture_2d<f32>;
@group(0) @binding(2) var t1: texture_2d<f32>;
@group(0) @binding(3) var t2: texture_2d<f32>;
@group(0) @binding(4) var t3: texture_2d<f32>;
@group(0) @binding(5) var t4: texture_2d<f32>;
@group(0) @binding(6) var t5: texture_2d<f32>;
@group(0) @binding(7) var t6: texture_2d<f32>;
@group(0) @binding(8) var t7: texture_2d<f32>;
@group(0) @binding(9) var t8: texture_2d<f32>;
@group(0) @binding(10) var t9: texture_2d<f32>;
@group(0) @binding(11) var t10: texture_2d<f32>;
@group(0) @binding(12) var t11: texture_2d<f32>;
@group(0) @binding(13) var t12: texture_2d<f32>;
@group(0) @binding(14) var t13: texture_2d<f32>;
@group(0) @binding(15) var t14: texture_2d<f32>;
@group(0) @binding(16) var t15: texture_2d<f32>;

// Glyph c at index c - 32; bit (row * 5 + col), rows top to bottom, bits 32..34 in .y.
const VIEW_FONT: array<vec2<u32>, 95> = array<vec2<u32>, 95>(
    vec2<u32>(0x00000000u, 0x0u), //  
    vec2<u32>(0x00421084u, 0x1u), // !
    vec2<u32>(0x0000014au, 0x0u), // "
    vec2<u32>(0x15f52beau, 0x0u), // #
    vec2<u32>(0x1f4717c4u, 0x1u), // $
    vec2<u32>(0x32d11173u, 0x0u), // %
    vec2<u32>(0x2c9a9926u, 0x0u), // &
    vec2<u32>(0x00000084u, 0x0u), // '
    vec2<u32>(0x08210888u, 0x2u), // (
    vec2<u32>(0x88842082u, 0x0u), // )
    vec2<u32>(0x2aefbaa0u, 0x0u), // *
    vec2<u32>(0x084f9080u, 0x0u), // +
    vec2<u32>(0x88c00000u, 0x0u), // ,
    vec2<u32>(0x000f8000u, 0x0u), // -
    vec2<u32>(0x8c000000u, 0x1u), // .
    vec2<u32>(0x44222110u, 0x0u), // /
    vec2<u32>(0xa33ae62eu, 0x3u), // 0
    vec2<u32>(0x884210c4u, 0x3u), // 1
    vec2<u32>(0xc444422eu, 0x7u), // 2
    vec2<u32>(0xe107420fu, 0x3u), // 3
    vec2<u32>(0x11f4a988u, 0x2u), // 4
    vec2<u32>(0xa3083c3fu, 0x3u), // 5
    vec2<u32>(0xa317842eu, 0x3u), // 6
    vec2<u32>(0x8422221fu, 0x0u), // 7
    vec2<u32>(0xa317462eu, 0x3u), // 8
    vec2<u32>(0xa10f462eu, 0x3u), // 9
    vec2<u32>(0x0c6018c0u, 0x0u), // :
    vec2<u32>(0x886018c0u, 0x0u), // ;
    vec2<u32>(0x08208888u, 0x2u), // <
    vec2<u32>(0x01f07c00u, 0x0u), // =
    vec2<u32>(0x88882082u, 0x0u), // >
    vec2<u32>(0x0046422eu, 0x1u), // ?
    vec2<u32>(0x83daf62eu, 0x7u), // @
    vec2<u32>(0x631fc62eu, 0x4u), // A
    vec2<u32>(0xe317c62fu, 0x3u), // B
    vec2<u32>(0xa210862eu, 0x3u), // C
    vec2<u32>(0xd318c527u, 0x1u), // D
    vec2<u32>(0xc217843fu, 0x7u), // E
    vec2<u32>(0x4217843fu, 0x0u), // F
    vec2<u32>(0xa31e862eu, 0x7u), // G
    vec2<u32>(0x631fc631u, 0x4u), // H
    vec2<u32>(0x8842108eu, 0x3u), // I
    vec2<u32>(0x9284211cu, 0x1u), // J
    vec2<u32>(0x52519531u, 0x4u), // K
    vec2<u32>(0xc2108421u, 0x7u), // L
    vec2<u32>(0x631ad771u, 0x4u), // M
    vec2<u32>(0x639ace31u, 0x4u), // N
    vec2<u32>(0xa318c62eu, 0x3u), // O
    vec2<u32>(0x4217c62fu, 0x0u), // P
    vec2<u32>(0x9358c62eu, 0x5u), // Q
    vec2<u32>(0x5257c62fu, 0x4u), // R
    vec2<u32>(0xe107043eu, 0x3u), // S
    vec2<u32>(0x0842109fu, 0x1u), // T
    vec2<u32>(0xa318c631u, 0x3u), // U
    vec2<u32>(0x1518c631u, 0x1u), // V
    vec2<u32>(0xab5ac631u, 0x2u), // W
    vec2<u32>(0x62a22a31u, 0x4u), // X
    vec2<u32>(0x08422a31u, 0x1u), // Y
    vec2<u32>(0xc222221fu, 0x7u), // Z
    vec2<u32>(0x8421084eu, 0x3u), // [
    vec2<u32>(0x10820841u, 0x4u), // backslash
    vec2<u32>(0x9084210eu, 0x3u), // ]
    vec2<u32>(0x00004544u, 0x0u), // ^
    vec2<u32>(0xc0000000u, 0x7u), // _
    vec2<u32>(0x00000082u, 0x0u), // `
    vec2<u32>(0xa3e83800u, 0x7u), // a
    vec2<u32>(0xe319b421u, 0x3u), // b
    vec2<u32>(0xa210b800u, 0x3u), // c
    vec2<u32>(0xa31cda10u, 0x7u), // d
    vec2<u32>(0x83f8b800u, 0x3u), // e
    vec2<u32>(0x84238a4cu, 0x0u), // f
    vec2<u32>(0xa1e8c7c0u, 0x3u), // g
    vec2<u32>(0x6319b421u, 0x4u), // h
    vec2<u32>(0x88421804u, 0x3u), // i
    vec2<u32>(0x92843008u, 0x1u), // j
    vec2<u32>(0x4a32a421u, 0x2u), // k
    vec2<u32>(0x88421086u, 0x3u), // l
    vec2<u32>(0x635aac00u, 0x4u), // m
    vec2<u32>(0x6319b400u, 0x4u), // n
    vec2<u32>(0xa318b800u, 0x3u), // o
    vec2<u32>(0x42f8bc00u, 0x0u), // p
    vec2<u32>(0x21ecd800u, 0x4u), // q
    vec2<u32>(0x4219b400u, 0x0u), // r
    vec2<u32>(0xe0e0b800u, 0x3u), // s
    vec2<u32>(0x24211c42u, 0x3u), // t
    vec2<u32>(0xb318c400u, 0x5u), // u
    vec2<u32>(0x1518c400u, 0x1u), // v
    vec2<u32>(0xab58c400u, 0x2u), // w
    vec2<u32>(0x54454400u, 0x4u), // x
    vec2<u32>(0xa1e8c400u, 0x3u), // y
    vec2<u32>(0xc4447c00u, 0x7u), // z
    vec2<u32>(0x08411098u, 0x6u), // {
    vec2<u32>(0x08421084u, 0x1u), // |
    vec2<u32>(0xc8441083u, 0x0u), // }
    vec2<u32>(0x008a8800u, 0x0u), // ~
);

struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) index: u32) -> Vertex {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: Vertex;
    out.position = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

fn load(t: texture_2d<f32>, uv: vec2<f32>) -> vec3<f32> {
    let size = textureDimensions(t, 0);
    let px = min(vec2<u32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(size)), size - 1u);
    return textureLoad(t, vec2<i32>(px), 0).rgb;
}

fn tile(k: u32, uv: vec2<f32>) -> vec3<f32> {
    switch k {
        case 0u: { return load(t0, uv); }
        case 1u: { return load(t1, uv); }
        case 2u: { return load(t2, uv); }
        case 3u: { return load(t3, uv); }
        case 4u: { return load(t4, uv); }
        case 5u: { return load(t5, uv); }
        case 6u: { return load(t6, uv); }
        case 7u: { return load(t7, uv); }
        case 8u: { return load(t8, uv); }
        case 9u: { return load(t9, uv); }
        case 10u: { return load(t10, uv); }
        case 11u: { return load(t11, uv); }
        case 12u: { return load(t12, uv); }
        case 13u: { return load(t13, uv); }
        case 14u: { return load(t14, uv); }
        default: { return load(t15, uv); }
    }
}

// Byte i of tile k's label (0 past its end).
fn label_byte(k: u32, i: u32) -> u32 {
    if (i >= 32u) {
        return 0u;
    }
    let w = k * 8u + i / 4u;
    let word = params.label_words[w / 4u][w % 4u];
    return (word >> ((i % 4u) * 8u)) & 0xffu;
}

fn label_length(k: u32) -> u32 {
    var n = 0u;
    loop {
        if (n >= 32u || label_byte(k, n) == 0u) {
            break;
        }
        n = n + 1u;
    }
    return n;
}

fn glyph_bit(c: u32, col: u32, row: u32) -> bool {
    var index = c - 32u;
    if (c < 32u || c > 126u) {
        index = 31u;
    }
    let g = VIEW_FONT[index];
    let bit = row * 5u + col;
    if (bit < 32u) {
        return ((g.x >> bit) & 1u) == 1u;
    }
    return ((g.y >> (bit - 32u)) & 1u) == 1u;
}

@fragment
fn fs(frag: Vertex) -> @location(0) vec4<f32> {
    let grey = vec4<f32>(0.1, 0.1, 0.1, 1.0);
    let columns = max(params.columns, 1u);
    let count = min(params.count, 16u);
    let rows = max((count + columns - 1u) / columns, 1u);
    let cells = vec2<f32>(f32(columns), f32(rows));
    let uv = frag.uv;
    let cell = min(vec2<u32>(uv * cells), vec2<u32>(columns, rows) - 1u);
    let target_size = frag.position.xy / max(uv, vec2<f32>(1e-6));
    let cell_size = target_size / cells;
    let origin = vec2<f32>(cell) * cell_size;
    let local = frag.position.xy - origin;
    if ((cell.x > 0u && local.x < 1.0) || (cell.y > 0u && local.y < 1.0)) {
        return grey;
    }
    let k = cell.y * columns + cell.x;
    if (k >= count) {
        return grey;
    }
    if (params.labels == 1u && local.y < 20.0) {
        let n = label_length(k);
        let lx = u32(local.x);
        if (n > 0u && lx < n * 12u + 4u) {
            let ly = u32(local.y);
            if (lx >= 2u && ly >= 3u && ly < 17u) {
                let gx = (lx - 2u) % 12u;
                let i = (lx - 2u) / 12u;
                if (i < n && gx < 10u && glyph_bit(label_byte(k, i), gx / 2u, (ly - 3u) / 2u)) {
                    return vec4<f32>(1.0);
                }
            }
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }
    let tile_uv = local / cell_size;
    return vec4<f32>(tile(k, tile_uv), 1.0);
}
