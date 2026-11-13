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
// crates/solstice-render/shaders/vg/raster_sw_depth.wgsl:1-42.
// The include preprocessor is not ported: `raster_sw_depth_module_wgsl()` concatenates
// `raster_common.wgsl`, `depth_stamp.wgsl`, this file and `raster_sw_body.wgsl`. RAA is
// standard-Z, so `write_visibility` stamps the depth key directly instead of reversing it.

// Per pixel: the view's stamp in the top byte over the 24-bit depth key, so that the nearest
// sample of the current drawing has the largest value and anything an earlier drawing left
// behind loses to it. The buffer is not cleared per drawing, only once each time the stamps
// start a new round (shadow.rs); the merge ignores other stamps.
@group(2) @binding(7) var<storage, read_write> shadow_depth: array<atomic<u32>>;

// Keeps the nearest depth. The view is orthographic with depth running 0 at the light's side
// to 1 at the far side, so anything past either side is left out.
fn write_visibility(index: u32, depth: f32, id: u32) {
    if depth < 0.0 || depth >= RAA_DEPTH_FAR {
        return;
    }
    atomicMax(&shadow_depth[index], stamped(view.stamp, depth));
}

// The same slope-scaled bias as the hardware shadow pipeline's, in depth per pixel of the
// steepest direction, so that a surface does not shadow itself where a texel spans a long
// stretch of it. Twin of the `slope_scale` in shadow.rs.
const SLOPE_BIAS: f32 = 1.5;

fn depth_bias(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let ab = b - a;
    let ac = c - a;
    // Never zero: the rasterizer has already dropped triangles with no area. A triangle almost
    // edge-on to the light gets a huge bias and is pushed past the far side, as the hardware
    // does too, clamping to 1 and failing its depth test.
    let det = ab.x * ac.y - ac.x * ab.y;
    let dzdx = (ab.z * ac.y - ac.z * ab.y) / det;
    let dzdy = (ac.z * ab.x - ab.z * ac.x) / det;
    return SLOPE_BIAS * max(abs(dzdx), abs(dzdy));
}
