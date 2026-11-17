// One `view.image` variant: the placeholder line below is replaced by the variant's texture
// binding and its `view_fetch` (crates/raa-graph/src/exec_ext/views.rs).

struct ViewParams {
    channels: u32,
    linearize: u32,
    layer: u32,
    slice: u32,
    mip: u32,
    decode: u32,
    range_min: f32,
    range_max: f32,
    exposure_ev: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> view: ViewParams;
//#SOURCE
@group(0) @binding(2) var view_sampler: sampler;

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

@fragment
fn fs(frag: Vertex) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(view_source, view.mip).xy);
    let uv = frag.uv;
    let px = vec2<i32>(uv * size);
    var c = view_fetch(px, uv);
    if (view.linearize == 1u) {
        let v = log2(1.0 / max(1.0 - c.r, 1e-6)) / 16.0;
        c = vec4<f32>(v, v, v, 1.0);
    }
    c = vec4<f32>(c.rgb * exp2(view.exposure_ev), c.a);
    let span = max(view.range_max - view.range_min, 1e-6);
    c = clamp(
        (c - vec4<f32>(view.range_min)) / span,
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );
    var out = vec4<f32>(c.r, c.g, c.b, 1.0);
    switch view.channels {
        case 1u: {
            out = vec4<f32>(c.r, c.r, c.r, 1.0);
        }
        case 2u: {
            out = vec4<f32>(c.g, c.g, c.g, 1.0);
        }
        case 3u: {
            out = vec4<f32>(c.b, c.b, c.b, 1.0);
        }
        case 4u: {
            out = vec4<f32>(c.a, c.a, c.a, 1.0);
        }
        case 5u: {
            out = vec4<f32>(c.rgb * c.a, 1.0);
        }
        default: {
            out = vec4<f32>(c.r, c.g, c.b, 1.0);
        }
    }
    return out;
}
