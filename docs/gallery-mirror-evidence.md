# Windows gallery mirror evidence

The direct runner uses the gallery's default Performance preset, authored
home camera, two mirror bounces, and confidence threshold 0.55. The UI check
uses the default 1280 × 720 canvas. The fixed-state physical captures use a
320 × 180 display and the native 160 × 90 reflection tap. The camera, time,
exposure and authored scene stay fixed while the lighting policy changes.
An isolated mirror at a separate fixed camera supplies real BVH misses.

Build the pinned gallery using Kong. Run the existing direct driver with
native Windows Node and hardware Chrome on the RTX 3090.

```powershell
node scripts/capture-area-pilot-direct.mjs $Chrome dist/global-illumination out/mirror-coverage gallery-mirror coverage
node scripts/capture-area-pilot-direct.mjs $Chrome dist/global-illumination out/mirror-validation gallery-mirror fixed
```

The coverage run saves its evidence even when no useful default mirror
samples are resident. It compares cold and warm counts at the current
128-cell cap, including actual off-screen secondary receivers that fall in
the grid and the subset the cache supports. These are distinct from
screen-space rejection counts.

The physical bracket raises the confidence threshold only for forced BVH and
verifies that every eligible mirror ray takes that route. The original
threshold is used for default measurements. A/B cases preserve the first ray,
geometry hit count and GPU digest of triangle and world position. The learned
rough tap additionally stores those values and first-hit direct RGB per pixel.
Its fixed seed lets the comparison check both identity and direct light
exactly and measure the downstream tail separately.

The spherical lighting mask isolates sky, sun, and other light. Volume,
spherical and cache-bypass measurements use linear HDR. The overlay check
measures the forward visualization separately from raw reflected radiance.
The learned package is identified by its SHA-256 and dispatch counts. The
reference uses an independent seed with spherical resources resident and
then released, and is checked for equal HDR and zero grid reads.

The direct driver validates all plane lengths and digests before writing
`mirror.json`. The rough evidence buffer is mixed FP32/U32 little-endian with
a 48-byte stride. Its lanes are first-hit XYZ, triangle ID, first-hit direct
RGB, grid trust, accepted grid RGB, and source kind.
