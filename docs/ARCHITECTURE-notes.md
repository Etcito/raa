# Architecture notes for contributors

These notes describe the current code path and extension points across the workspace.

## Asset loading to frame submission

1. An asset starts as prims v0 text (`raa.prims`, version 0). `raa-loader` parses JSON, validates the prims document, then builds it at the requested quality.
2. During build, primitive kinds and parameters are sent to providers registered in `raa_geom::GeometryRegistry`. The built-in eight providers are installed by `raa_prims::register_all`; additional kinds can be registered through the same `GeometryProvider` interface. Boolean child operations are passed to the configured `BooleanResolver`; `raa-csg` supplies the CSG implementation, while `NoBooleans` refuses operations that need a resolver.
3. The loader returns a `LoadedAsset` containing its generated meshes, material descriptions, anchors, and colliders. Each `LoadedMesh` includes mesh geometry and its transform in the asset.
4. `Renderer::upload_mesh` validates and uploads each `MeshData`, returning a `MeshId`. The application creates `Scene` nodes and instances that refer to mesh and material IDs. Scene updates world transforms and fills a reusable `RenderList` with visible instances.
5. `Renderer::prepare` updates the scene, builds the render list, and passes it to `FramePrep`. `FramePrep::prepare_cpu` writes instance data, frame uniforms, and draw ranges into its preallocated buffers.
6. `Renderer::install` adds the `ForwardPass` to the `FrameGraph`. The graph compiles declared resources and pass dependencies, then executes the forward pass, which reads the prepared draw ranges and submits indexed instanced draws to the backbuffer.
7. In the browser, `WebRuntime::frame` updates frame input and counters, calls the app's frame callback, and executes the installed frame graph. The `raa-js` `Engine` app callback prepares the renderer for the current scene before the runtime executes the graph.

## Extension hooks

- **Geometry:** implement `raa_geom::GeometryProvider` for a kind, then register it in `GeometryRegistry`. A provider receives parameters and quality and returns mesh data or a typed geometry error. The loader uses this registry when it builds primitive nodes.
- **Asset formats:** implement `raa_loader::AssetLoader` and register it in `LoaderRegistry`. A loader can recognize its format and produce a `LoadedAsset` using the supplied `LoadContext`.
- **Boolean operations:** implement `raa_loader::BooleanResolver` and provide it in `LoadContext`; it receives a host and tool description and returns the resolved mesh or a load refusal. The built-in CSG resolver is used by the JS engine.
- **Render passes:** implement `raa_core::RenderPass` with its name, resource reads and writes, and encode operation. Add it to a `FrameGraph`, or place it relative to an existing named pass with `add_pass_before` or `add_pass_after`. The renderer installs its forward pass under the name `forward`.

## Frame-path allocation rule

The engine frame path is expected to allocate no heap memory after warm-up in RAA-owned code: `Scene::update_world`, `Scene::build_render_list`, `FramePrep::prepare_cpu`, RAA code in `FrameGraph::execute`, and `WebRuntime::frame`. Scene order and render-list storage are reused, and frame-preparation buffers are allocated up front. Use the scene and list capacity constructors when the expected size is known. Calls that create or remove scene objects may allocate. Allocations inside wgpu or web-sys are outside this RAA-owned-code rule.

## Errors and refusals

Public operations return typed errors rather than silently producing invalid output. Crate errors implement `Display` and `Error` and expose a stable `code()` string; loader errors also carry a JSON path for asset diagnostics. The JS API converts failures to JavaScript `Error` objects named `RaaError` with a `code` property. Refusals such as unknown handles, invalid geometry, unsupported Boolean operations, or exceeded capacities retain their specific codes and messages through the boundary.

## JavaScript state and handles

`raa-js` shares engine state between the JS-owned `Engine` and the runtime-owned app using `Rc<RefCell<Option<State>>>`. This lets both sides access the same renderer, scene, registries, and reusable frame buffers without global state. JavaScript receives monotonically increasing `u32` handles mapped to internal IDs. Removed handles are not reused, so a stale numeric handle cannot accidentally refer to a later object.
