# raa-dof

Depth of field for the lit renderer (S12 `dof`, order 110; DOF values).

## Public items

- `DofDesc { focus_distance_m, f_stop, focal_length_mm, autofocus, max_coc_px }` with
  `Default`: focus 10 m, f/2.8, 50 mm, `autofocus: None`, `max_coc_px: 24` (at 720p).
  `autofocus` is an optional screen UV (x right, y down) each in `[0, 1]`.
- `DofDesc::validate` refuses with `RaaError::FrameGraph { code: "post.invalid" }`, and the
  message names the field: every value must be finite, `focus_distance_m` must exceed the
  focal length in metres, `f_stop` is in `[0.7, 64]`, `focal_length_mm` in `[8, 600]`,
  `autofocus` in `[0, 1]`, and `max_coc_px` in `(0, 64]`.
- `coc_px(z, &DofDesc, viewport_h) -> f32`: the signed thin-lens circle of confusion in
  pixels. It is negative in front of the focus plane and positive behind it, and is clamped
  to `+-max_coc_px * viewport_h / 720`. A non-finite or non-positive `z` returns 0.
- `SENSOR_WIDTH_MM` (36) and `REFERENCE_VIEWPORT_H` (720), the constants the formula uses.
- `DofFeature` and `DofHandle` (including `set_camera`) arrive with the feature task; they
  install the `dof` pass (autofocus readback, tile max/min CoC dilation, half-res near/far
  prefilter, spiral gather, bilateral composite) and are not part of this skeleton.
- `POST_INVALID` is the `post.invalid` code.

## Circle of confusion

The 36 mm sensor is mapped onto the viewport height, so the same lens and f-stop give the
same relative blur at any resolution:

```text
A      = f / N                              aperture diameter (m)
coc_mm = A * f * (z - s) / (z * (s - f))    signed CoC on the sensor (mm)
px     = coc_mm / 36 * viewport_h
```

`f` is the focal length in metres, `N` the f-number, `s` the focus distance and `z` the
sample depth in metres. Reducing `f_stop` widens the aperture and grows `|coc|`. The clamp
is expressed at 720p: 1440 pixels of viewport height double the pixel limit.

## Test

`cargo test -p raa-dof`: `tests/coc.rs` checks the defaults and every refusal code, zero CoC
at the focus distance, the sign of the layers, growth with a wider aperture, the 36 mm
mapping, and the clamp scaling with viewport height.
