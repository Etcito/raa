# RAA API m15: animation hooks

Additions of chain milestone 9 to `@etcito/raa`: squash and stretch and jiggle springs in the
vertex stage, per-instance tint and parameters, and a batched per-instance record. Rigid part
hierarchies keep their v0 form. Types: `packages/raa/types/m15-animation.d.ts` (appended to
`index.d.ts` when the package is staged; `Engine` merges with the v0 declaration). Everything in
v0 and `docs/api/m07`..`m14` stays valid. Failures throw a `RaaError` with a `code`.

`setSquash` and `setJiggle` replace the settings of their instance: the fields you omit take the
defaults below. A refusal changes nothing. Lengths are metres, angles are radians, colours are as
noted per field.

## Exports

- `Rgba`: `[r, g, b, a]`; `rgb` is sRGB, `a` is a linear multiplier.
- `SquashSettings`: `{ axis?, amount }`.
- `JiggleSettings`: `{ stiffness?, damping?, amount?, height }`.
- `setInstanceTint(instance, rgba)`, `setInstanceParams(instance, params)`, `setSquash(instance, settings)`,
  `setJiggle(instance, settings)`, `setInstanceExtras(instances, data)`: methods of `Engine`.

`Vec3` (from `index.d.ts`) is reused, not redefined.

## Rigid hierarchies

A rigid hierarchy is a tree of nodes: the body is a root node, the head and foot are children.
Keep the v0 pattern: `addNode` once, `setTransforms` every frame. Children inherit the transform of
their parent, so moving the body moves the head and foot without any extra call.

```js
const body = engine.addNode(null);
const head = engine.addNode(body, [0, 1.6, 0]);
const foot = engine.addNode(body, [0.2, 0, 0]);
engine.addInstance(body, bodyMesh, material);
engine.addInstance(head, headMesh, material);
engine.addInstance(foot, footMesh, material);

const nodes = new Uint32Array([body, head, foot]);
const pose = new Float32Array(3 * 10); // t (3), r (4, [x, y, z, w]), s (3) per node

function update(nowMs) {
  // fill `pose` from the animation, then:
  engine.setTransforms(nodes, pose);
  engine.render(nowMs);
}
```

## Squash and stretch

`setSquash(instance, { axis, amount })` squashes an instance along a local axis and keeps its
volume: a positive `amount` shortens the instance along `axis` and widens the two other axes; a
negative one stretches it. `axis` defaults to `[0, 1, 0]` and is normalised by the engine. The
shader clamps `amount` to at least -0.9.

The squash is about the instance's local origin. Build the mesh with its origin at the contact
point (the feet on the ground for a character), so the contact stays where it is while the body
squashes above it.

```js
engine.setSquash(instance, { amount: 0.3 }); // squash along +Y
engine.setSquash(instance, { axis: [1, 0, 0], amount: -0.2 }); // stretch along X
```

Calling `setSquash` again replaces the squash of the instance. Squash is applied in the vertex
stage, so a squashed instance still draws with its own mesh and material.

## Jiggle springs

`setJiggle(instance, { stiffness, damping, amount, height })` attaches a spring to an instance.
The spring is driven by the world motion of the instance's node: each `render` call (the frame
update) steps the springs of the configured instances from their node world matrices and writes the
result to the instance's jiggle record. Defaults: `stiffness` 120, `damping` 0.35 (a damping ratio,
1 is critical), `amount` 1. `height` is required.

```js
engine.setJiggle(instance, { height: 1.2 });
engine.setJiggle(instance, { stiffness: 80, damping: 0.2, amount: 0.5, height: 1.2 });
engine.setJiggle(instance, null); // removes the spring
```

Springs are deterministic: the same sequence of node transforms and time steps gives the same
result, and a node that stops moves the spring to rest, so the motion settles. A step of 0 seconds
(a fixed-time capture) does not move a spring, so a capture shows the spring where it was last
stepped. For that reason a lookdev scene that shows jiggle in a capture does not rely on a spring:
it writes the jiggle record directly, see below. Springs are for interactive loops with a real time
step.

## Direct jiggle through `setInstanceExtras`

`setInstanceExtras` writes the animation record of many instances at once. A scene that needs a
fixed jiggle in a capture writes the jiggle fields (floats 8 to 11) itself:

```js
const ids = new Uint32Array([instance]);
const data = new Float32Array(32);
data.set([1, 1, 1, 1], 0); // tint: sRGB rgb, linear alpha
data.set([0, 1, 0, 0], 4); // squash: axis, amount
data.set([0, 0, 0, 1.2], 8); // jiggle: local offset (m), height (m)
engine.setInstanceExtras(ids, data);
```

A jiggle record written here stays until the next `render`; at that `render`, an instance with a
`setJiggle` spring has its record overwritten by the spring.

## Tint

`setInstanceTint(instance, [r, g, b, a])` multiplies the base colour of an instance by `rgb` and its
opacity by `a`. `rgb` is sRGB, as the colours of the rest of the API that are written as sRGB: each
component is finite and >= 0, and `[1, 1, 1]` is no tint. `a` is a linear multiplier in [0, 1]. The
default is `[1, 1, 1, 1]`.

```js
engine.setInstanceTint(instance, [1.2, 0.9, 0.8, 1]); // warmer
engine.setInstanceTint(instance, [1, 1, 1, 0.5]); // half opacity
```

## Instance parameters

`setInstanceParams(instance, params)` sets 16 free floats per instance (`Float32Array` or an array of
16 numbers). Material functions read them as `instance_params` (four vec4). The default is all zero.

## Batched layout

`setInstanceExtras(instances, data)` takes one `Uint32Array` of instance handles and one
`Float32Array` of 32 floats per instance, in the order of `instances`. The layout of each group of
32 floats:

| floats | field | meaning |
|---|---|---|
| 0 to 3 | tint | `rgb` sRGB, `a` linear alpha; default 1 |
| 4 to 7 | squash | `axis` xyz (unit), `amount`; default `(0, 1, 0, 0)` |
| 8 to 11 | jiggle | local offset xyz (m), `height` (m); default 0 |
| 12 to 27 | params | the 16 instance parameters of `setInstanceParams` |
| 28 to 31 | reserved | write 0 |

Every handle and value is checked before anything is written, so a refusal changes nothing. The
batch is the same record the other setters write: `setInstanceTint`, `setSquash`, `setInstanceParams`
and a jiggle spring each update their own floats.

Rigid hierarchies are not part of the batch: keep using `addNode` and `setTransforms` for them.

## Refusals

Every method throws a `RaaError` with one of these codes.

| code | raised by |
|---|---|
| `api.unknown_handle` | every method taking an instance: the handle is not a live instance |
| `api.invalid_argument` | a value is not finite, a shape is wrong (not an object, not three numbers), or a tint component is negative or alpha is outside [0, 1] |
| `api.length_mismatch` | `setInstanceParams`: not 16 values; `setInstanceExtras`: `data.length !== 32 * instances.length` |
| `deform.invalid` | `setSquash`: an axis of zero length; `setJiggle`: stiffness <= 0, damping < 0, or height < 0; `setInstanceExtras`: a squash axis of zero length |

After `dispose` every method throws `api.disposed`, as in v0.
