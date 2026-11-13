/*
MIT License

Copyright (c) 2026 SamG-Coder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Adapted from solstice@7e465b599c13da165948ebaa44df824c36c3ae07
// crates/solstice-render/shaders/vg/hzb_seed.wgsl.
// Rewritten for RAA's S32 assembly (`DEPTH_WGSL` + `pass_view.wgsl` + this body) and for
// standard-Z: every texel holds the *farthest* depth of the area it covers, reduced through
// `raa_depth_farthest` from the nearest possible depth. The pyramid is a power of two and the
// screen need not be, so a texel can straddle source pixels. Every touched pixel counts, to
// stay conservative. Key 0 is far or empty, so an all-empty texel ends at 1.0.

struct HzbParams {
    source_size: vec2u,
    target_size: vec2u,
}

@group(0) @binding(0) var<uniform> params: HzbParams;
@group(0) @binding(1) var<storage, read> vis_depth: array<u32>;
@group(0) @binding(2) var output: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var source_depth: texture_depth_2d;

struct SourceRange {
    begin: vec2u,
    end: vec2u,
}

fn source_range(id: vec2u) -> SourceRange {
    let source = params.source_size;
    let begin = id * source / params.target_size;
    let end = min(
        ((id + vec2u(1u)) * source + params.target_size - vec2u(1u)) / params.target_size,
        source,
    );
    return SourceRange(begin, end);
}

@compute @workgroup_size(8, 8)
fn seed_from_keys(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.target_size.x || id.y >= params.target_size.y {
        return;
    }
    let range = source_range(id.xy);
    var farthest = 0.0;
    for (var y = range.begin.y; y < range.end.y; y++) {
        for (var x = range.begin.x; x < range.end.x; x++) {
            let key = vis_depth[y * params.source_size.x + x];
            farthest = raa_depth_farthest(farthest, raa_depth_from_key(key));
        }
    }
    textureStore(output, id.xy, vec4f(farthest, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn seed_from_depth(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.target_size.x || id.y >= params.target_size.y {
        return;
    }
    let range = source_range(id.xy);
    var farthest = 0.0;
    for (var y = range.begin.y; y < range.end.y; y++) {
        for (var x = range.begin.x; x < range.end.x; x++) {
            let depth = textureLoad(source_depth, vec2i(i32(x), i32(y)), 0);
            farthest = raa_depth_farthest(farthest, depth);
        }
    }
    textureStore(output, id.xy, vec4f(farthest, 0.0, 0.0, 0.0));
}
