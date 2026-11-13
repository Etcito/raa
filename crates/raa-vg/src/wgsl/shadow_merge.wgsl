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
// crates/solstice-render/shaders/vg/shadow_merge.wgsl:1-28.
// The include preprocessor is not ported: `shadow_merge_module_wgsl()` concatenates
// `pass_view.wgsl`, `depth_stamp.wgsl` and this file, without `raster_common.wgsl`, so the
// group-2 bindings are 0 `shadow_depth` and 1 `view`. Lays the depth the software rasterizer
// wrote for a shadow cascade over the cascade's depth map: one triangle over the whole map,
// writing each pixel's depth where this drawing left one, under the same depth test as the
// hardware rasterizer's clusters. The cull's arguments drop the triangle when no cluster went
// to software, so an empty drawing costs nothing.

@group(2) @binding(0) var<storage, read> shadow_depth: array<u32>;
@group(2) @binding(1) var<uniform> view: PassView;

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    // Three corners covering the clip square.
    let x = f32(i32(vertex & 1u) * 4 - 1);
    let y = f32(i32(vertex >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @builtin(frag_depth) f32 {
    let pixel = vec2<u32>(position.xy);
    let word = shadow_depth[pixel.y * u32(view.viewport.x) + pixel.x];
    if stamp_of(word) != view.stamp {
        discard;
    }
    return stamped_depth(word);
}
