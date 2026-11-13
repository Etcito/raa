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
// crates/solstice-render/shaders/vg/hzb_down.wgsl.
// Every level of the depth pyramid after the first: the farthest of the 2x2 texels above,
// through `raa_depth_farthest` (solstice's reversed-Z `min`) from the nearest possible depth.

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let output_size = textureDimensions(output);
    if id.x >= output_size.x || id.y >= output_size.y {
        return;
    }
    let last = vec2i(textureDimensions(source)) - vec2i(1);
    let base = vec2i(id.xy) * 2;
    var farthest = raa_depth_farthest(
        textureLoad(source, min(base, last), 0).r,
        textureLoad(source, min(base + vec2i(1, 0), last), 0).r,
    );
    farthest = raa_depth_farthest(
        farthest,
        textureLoad(source, min(base + vec2i(0, 1), last), 0).r,
    );
    farthest = raa_depth_farthest(
        farthest,
        textureLoad(source, min(base + vec2i(1, 1), last), 0).r,
    );
    textureStore(output, id.xy, vec4f(farthest, 0.0, 0.0, 0.0));
}
