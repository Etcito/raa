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
// crates/solstice-render/shaders/sky/bodies.wgsl:14,23-205.
// Requires atmosphere.wgsl and the shared milky_way.wgsl noise helpers.
// The composer supplies S10 group 2: raa_physical_sky (0),
// raa_sky_transmittance (1), raa_sky_clamp_sampler (6). No bindings here.
// Authored discs.xy replace the source sun display constants; the lunar radius
// remains the source 0.0075. Pixel angles are guarded for zero-sized footprints.
const raa_celestial_moon_radius: f32 = 0.0075;

// Light that crosses the whole atmosphere to reach the viewer along `direction`: dimmed and
// reddened near the horizon, and nothing from below it.
fn air_transmittance(direction: vec3f) -> vec3f {
    let r = max(raa_physical_sky.viewer.w, raa_atmosphere_min_radius_km);
    let ratio = clamp(raa_atmosphere_ground_km(raa_physical_sky) / r, 0.0, 1.0);
    let dip = -sqrt(max(1.0 - ratio * ratio, 0.0));
    let t = textureSampleLevel(raa_sky_transmittance, raa_sky_clamp_sampler, raa_atmosphere_transmittance_uv(r, direction.y, raa_physical_sky), 0.0).rgb;
    return t * smoothstep(dip - 0.006, dip + 0.006, direction.y);
}

// How much of a pixel `pixel_angle` across lies inside a disc of `radius`, `angle` from its
// centre: a one-pixel ramp, so the edge is crisp at any resolution.
fn disc_coverage(angle: f32, radius: f32, pixel_angle: f32) -> f32 {
    return clamp((radius - angle) / max(pixel_angle, 1.0e-8) + 0.5, 0.0, 1.0);
}

// ---- The sun ----

fn sun_body(direction: vec3f, pixel_angle: f32) -> vec3f {
    if raa_physical_sky.discs.x <= 0.0 || raa_physical_sky.discs.y <= 0.0 {
        return vec3f(0.0);
    }
    let toward = raa_physical_sky.sun.xyz;
    // The chord between unit vectors: the angle, for small angles, without acos's lost
    // precision near 1.
    let angle = length(direction - toward);
    let cover = disc_coverage(angle, raa_physical_sky.discs.x, pixel_angle);
    if cover <= 0.0 {
        return vec3f(0.0);
    }
    // Limb darkening: towards the edge one sees the sun's cooler, outer layers, more so in blue.
    let mu = sqrt(max(1.0 - raa_milky_way_sq(min(angle / raa_physical_sky.discs.x, 1.0)), 0.0));
    let limb = (vec3f(1.0) - vec3f(0.55, 0.62, 0.72) * (1.0 - mu)) / (vec3f(1.0) - vec3f(0.55, 0.62, 0.72) / 3.0);
    return air_transmittance(toward) * raa_physical_sky.daylight.rgb * raa_physical_sky.daylight.w
        * raa_physical_sky.discs.y * limb * cover;
}

// ---- The moon ----

// Dark seas of old lava on the moon's near side, as (latitude, longitude, radius across,
// radius north-south) in degrees, sized from their true diameters: east is towards Mare
// Crisium, on the right as seen from the northern hemisphere. Overlapping, they join into the
// great connected dark regions the eye sees.
const raa_celestial_maria_count: u32 = 17u;
const raa_celestial_maria = array<vec4f, 17>(
    vec4f(18.0, -58.0, 22.0, 32.0), // Oceanus Procellarum
    vec4f(-4.0, -43.0, 14.0, 16.0), // its southern reach, towards Mare Nubium
    vec4f(34.0, -17.0, 20.0, 16.0), // Mare Imbrium
    vec4f(27.5, 17.5, 12.0, 11.0), // Mare Serenitatis
    vec4f(8.0, 30.0, 15.0, 12.0), // Mare Tranquillitatis
    vec4f(17.0, 59.0, 9.0, 8.0), // Mare Crisium
    vec4f(-7.0, 52.0, 9.0, 13.0), // Mare Fecunditatis
    vec4f(-15.0, 35.0, 6.0, 6.0), // Mare Nectaris
    vec4f(-20.0, -15.0, 12.0, 9.0), // Mare Nubium
    vec4f(-10.0, -23.0, 6.0, 5.0), // Mare Cognitum
    vec4f(7.5, -30.0, 9.0, 7.0), // Mare Insularum
    vec4f(-24.0, -39.0, 7.0, 6.5), // Mare Humorum
    vec4f(56.0, 0.0, 40.0, 5.0), // Mare Frigoris
    vec4f(13.0, 4.0, 5.5, 4.0), // Mare Vaporum
    vec4f(2.0, 1.0, 5.0, 3.5), // Sinus Medii
    vec4f(45.0, -31.0, 5.0, 4.0), // Sinus Iridum
    vec4f(52.0, -52.0, 10.0, 6.0), // Sinus Roris
);

// A young crater's bright rays: splashed rock flung out in long thin streaks, plainest at full
// moon. `crater` is its position on the unit sphere, `reach` how far the rays run (radians).
fn crater_rays(p: vec3f, crater: vec3f, reach: f32, salt: u32) -> f32 {
    let distance = acos(clamp(dot(p, crater), -1.0, 1.0));
    // Bearing around the crater, in a frame on its tangent plane.
    let east = normalize(cross(vec3f(0.0, 1.0, 0.0), crater));
    let north = cross(crater, east);
    let around = normalize(vec2f(dot(p, east), dot(p, north)) + vec2f(1.0e-6, 0.0));
    let circle = vec3f(around, 0.0);
    // Many thin streaks of different strengths, each running its own length.
    let streaks = smoothstep(0.62, 0.85, raa_milky_way_value_noise(circle * 16.0, salt)) * 0.7
        + smoothstep(0.6, 0.9, raa_milky_way_value_noise(circle * 45.0, salt + 7u)) * 0.45;
    let length = reach * mix(0.35, 1.4, raa_milky_way_value_noise(circle * 16.0, salt + 13u));
    let fade = exp(-distance / length) * smoothstep(0.0, 0.02, distance);
    // The crater itself, and the bright blanket of rock thrown out around it.
    let halo = exp(-raa_milky_way_sq(distance / 0.025));
    return streaks * fade + halo;
}

// A point on the moon in its own frame (x towards lunar east, y north, z towards the Earth),
// to selenographic latitude and longitude, in degrees.
fn selenographic(p: vec3f) -> vec2f {
    return vec2f(asin(clamp(p.y, -1.0, 1.0)), atan2(p.x, p.z)) * (180.0 / raa_milky_way_pi);
}

// The moon's surface at `p`: its albedo, and a bump that tilts the normal across x and y.
// `full` is 1 near full moon, when the bright rays stand out.
fn moon_surface(p: vec3f, full: f32) -> vec3f {
    // The seas' outlines, warped so that no shore is a clean curve: broad bends of up to ten
    // degrees, then finer ones.
    let warp = vec2f(raa_milky_way_fbm(p * 3.0, 3u, 5u), raa_milky_way_fbm(p * 3.0, 3u, 7u)) - vec2f(0.5);
    let fine_warp = vec2f(raa_milky_way_fbm(p * 9.0, 3u, 9u), raa_milky_way_fbm(p * 9.0, 3u, 10u)) - vec2f(0.5);
    let coords = selenographic(p) + warp * 20.0 + fine_warp * 7.0;
    let shore = (raa_milky_way_value_noise(p * 40.0, 13u) - 0.5) * 0.08;
    var dry = 1.0;
    for (var i = 0u; i < raa_celestial_maria_count; i++) {
        let mare = raa_celestial_maria[i];
        let offset = vec2f(
            (coords.y - mare.y) * cos(radians(coords.x)) / mare.z,
            (coords.x - mare.x) / mare.w,
        );
        dry *= 1.0 - smoothstep(1.0, 0.86, length(offset) + shore);
    }
    let sea = 1.0 - dry;
    // Highlands pale, seas dark, each varied in tone across their breadth, and mottled by
    // craters too small to see one by one.
    let tone = raa_milky_way_fbm(p * 4.0, 3u, 17u);
    let lava = raa_milky_way_fbm(p * 8.0, 4u, 19u);
    let mottle = raa_milky_way_fbm(p * 26.0, 4u, 23u);
    var albedo = mix(0.92 + 0.16 * tone, 0.33 + 0.2 * lava, sea) * (0.9 + 0.2 * mottle);

    // The rayed craters: Tycho in the southern highlands, Copernicus and Kepler on the seas, and
    // Aristarchus, the brightest spot on the moon.
    let tycho = vec3f(-0.1433, -0.6859, 0.7134);
    let copernicus = vec3f(-0.3385, 0.1671, 0.9260);
    let kepler = vec3f(-0.6096, 0.1412, 0.7800);
    let aristarchus = vec3f(-0.6748, 0.4024, 0.6186);
    let rays = crater_rays(p, tycho, 0.5, 31u) * 1.4
        + crater_rays(p, copernicus, 0.2, 37u) * 0.9
        + crater_rays(p, kepler, 0.14, 41u) * 0.6
        + exp(-raa_milky_way_sq(acos(clamp(dot(p, aristarchus), -1.0, 1.0)) / 0.02)) * 0.8;
    albedo += rays * mix(0.12, 0.6, full);

    // Relief: craters and hills that roughen the shadow line between day and night.
    let scale = 34.0;
    let step = 0.4;
    let h = raa_milky_way_fbm(p * scale, 4u, 53u);
    let bump = vec2f(
        raa_milky_way_fbm(p * scale + vec3f(step, 0.0, 0.0), 4u, 53u) - h,
        raa_milky_way_fbm(p * scale + vec3f(0.0, step, 0.0), 4u, 53u) - h,
    );
    return vec3f(albedo, -bump * 0.6);
}

// The moon, lit by the sun at its phase and faintly by the Earth. Returns its light, and in A how
// much of the pixel it covers, to hide the stars behind it.
fn moon_body(direction: vec3f, pixel_angle: f32) -> vec4f {
    if raa_physical_sky.moon.w <= 0.0 {
        return vec4f(0.0);
    }
    let toward = raa_physical_sky.moon.xyz;
    let offset = direction - toward;
    let cover = disc_coverage(length(offset), raa_celestial_moon_radius, pixel_angle);
    if cover <= 0.0 {
        return vec4f(0.0);
    }
    // The moon keeps one face to the Earth, its north towards the celestial pole: to the viewer,
    // its north is up the sky towards the pole and its east to the right.
    var north = raa_physical_sky.pole.xyz - toward * dot(raa_physical_sky.pole.xyz, toward);
    north = select(vec3f(0.0, 1.0, 0.0), normalize(north), dot(north, north) > 1.0e-8);
    let right = cross(toward, north);
    var q = vec2f(dot(offset, right), dot(offset, north)) / raa_celestial_moon_radius;
    q *= min(1.0, 0.999 / max(length(q), 1.0e-6));
    let z = sqrt(max(1.0 - dot(q, q), 0.0));

    // Phase: the angle at the moon between the sun and the viewer, 0 at full.
    let sun = raa_physical_sky.sun.xyz;
    let cos_phase = dot(sun, -toward);
    let full = smoothstep(0.6, 1.0, cos_phase);
    let surface = moon_surface(vec3f(q, z), full);
    let normal = normalize(right * (q.x + surface.y * z) + north * (q.y + surface.z * z) - toward * z);

    // Lommel-Seeliger reflection, as from dusty regolith: the full moon is as bright at its edge
    // as at its centre, and the shadow line is soft.
    let mu_sun = max(dot(normal, sun), 0.0);
    let mu_view = max(dot(normal, -toward), 0.0);
    let sunlit = mu_sun / max(mu_sun + mu_view, 1.0e-4) * 2.0;
    // Close to full the shadows between grains vanish and the moon brightens sharply.
    let surge = 1.0 + 0.2 * pow(max(cos_phase, 0.0), 60.0);
    // Earthshine: sunlight off the Earth, which from the moon is nearly full when the moon is a
    // thin crescent, lighting its dark side a dim ashen grey.
    let earth = raa_milky_way_sq(0.5 - 0.5 * cos_phase);
    let earthshine = 0.035 * earth;
    let light = surface.x * (sunlit * surge + earthshine);
    // Grey rock, a touch warm.
    let color = vec3f(1.0, 0.94, 0.86);
    let radiance = color * light * raa_physical_sky.moon.w * air_transmittance(toward);
    return vec4f(radiance * cover, cover);
}

