// `view.compare`: two float images shown split, side by side or as their absolute difference
// (crates/raa-graph/src/exec_ext/views.rs `view_compare_source`).

struct CompareParams {
    mode: u32,
    position: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: CompareParams;
@group(0) @binding(1) var view_a: texture_2d<f32>;
@group(0) @binding(2) var view_b: texture_2d<f32>;

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

fn load_a(uv: vec2<f32>) -> vec3<f32> {
    let size = textureDimensions(view_a, 0);
    let px = min(vec2<u32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(size)), size - 1u);
    return textureLoad(view_a, vec2<i32>(px), 0).rgb;
}

fn load_b(uv: vec2<f32>) -> vec3<f32> {
    let size = textureDimensions(view_b, 0);
    let px = min(vec2<u32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(size)), size - 1u);
    return textureLoad(view_b, vec2<i32>(px), 0).rgb;
}

@fragment
fn fs(frag: Vertex) -> @location(0) vec4<f32> {
    let uv = frag.uv;
    var c = vec3<f32>(0.0);
    switch params.mode {
        case 1u: {
            if (uv.x < 0.5) {
                c = load_a(vec2<f32>(2.0 * uv.x, uv.y));
            } else {
                c = load_b(vec2<f32>(2.0 * uv.x - 1.0, uv.y));
            }
        }
        case 2u: {
            c = abs(load_a(uv) - load_b(uv));
        }
        default: {
            if (uv.x < params.position) {
                c = load_a(uv);
            } else {
                c = load_b(uv);
            }
            let width = frag.position.x / max(uv.x, 1e-6);
            if (abs(frag.position.x - params.position * width) <= 1.0) {
                c = vec3<f32>(1.0);
            }
        }
    }
    return vec4<f32>(c, 1.0);
}
