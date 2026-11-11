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
// shaders/sky/milky_way_bake.wgsl; output moved to the explicit private group 2.
// Works out the Milky Way once, into a map of the celestial sphere: glow in RGB, as the square
// root of its share of raa_milky_way_range, and how thick its faint stars are in A.


@group(2) @binding(0) var raa_milky_way_output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn raa_milky_way_bake(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(raa_milky_way_output);
    if any(id.xy >= size) {
        return;
    }
    let uv = (vec2f(id.xy) + vec2f(0.5)) / vec2f(size);
    let galaxy = raa_milky_way_glow(raa_milky_way_celestial_direction(uv));
    let glow = sqrt(clamp(galaxy.rgb / raa_milky_way_range, vec3f(0.0), vec3f(1.0)));
    textureStore(raa_milky_way_output, id.xy, vec4f(glow, clamp(galaxy.a, 0.0, 1.0)));
}
