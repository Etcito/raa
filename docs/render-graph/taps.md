# Render graph taps

Each row names the Rust field or accessor that holds the view or buffer a pass publishes under its (pass entry, port) key in the pipeline's `ResourceTable`, and the file that holds that field. The row set is every catalog output port of every pass node type in `crates/raa-graph/src/nodes/passes.rs`, 81 rows in all. `crates/raa-graph/tests/view_nodes.rs` reads this table: every line after the header that starts with `| ` is a row, the separator is skipped, and cells 1 and 2 are the pass and the port.

## Rules

- Every pass publishes each of its output ports in `prepare`, after its (re)allocation, on every frame it runs.
- Chain `color` ports are published at the end of `record`, with `Targets::hdr().src()` after the pass's swap, which is the view it wrote. The chain's src and dst are decided during `record`, so the publish waits for it.
- `tonemap` and `overlay` `color` are published by the pipeline, with the display target `Targets::output`, only while at least one graph frame is added.
- `skin`'s `skinned` is published in `record`, because the raa-js wrapper `SkinPipelinePass` forwards only `set_enabled` and `record`.
- When a field holds a texture rather than a view (`Textures::_raw`, `DofTargets::_coc`, `SsrTargets::_hiz`), the publisher creates the view from it.
- Where two ports name one texture (`gi.ssgi` `ssgi` and `output`, `shadow.contact` `contact` and `output`), both rows name it. For `gi.area-shadow`, `areaShadow` and `mask` both name the two-layer mask array.
- The parity pairs of `gi.screen` (`output`, `meta`, `filter`) name the half of the current parity. `gi` is the screen output the forward pass reads.
- `sky.env` `envSpecular` names the SKY_ENV source cube, whose mips the IBL filter writes. The lighting's `env_specular` is the IBL filter output (`crates/raa-ibl/src/filter.rs`), which this table does not name.

## Taps

| pass | port | field | file |
|---|---|---|---|
| sky.atmosphere | transmittanceLut | AtmosphereResources::transmittance | crates/raa-sky/src/atmosphere.rs |
| sky.atmosphere | multiScatteringLut | AtmosphereResources::multiple_scattering | crates/raa-sky/src/atmosphere.rs |
| sky.atmosphere | skyView | ViewGpu::sky_view | crates/raa-sky/src/atmosphere.rs |
| sky.cloud-lighting | cloudPanorama | CloudViews::panorama | crates/raa-sky/src/cloud_backend.rs |
| sky.cloud-lighting | cloudTransmittance | CloudViews::transmittance | crates/raa-sky/src/cloud_backend.rs |
| sky.env | envSpecular | SkyEnvPass::cube | crates/raa-sky/src/env_pass.rs |
| skin | skinned | SkinnedView::stream0 | crates/raa-lit/src/skin_views.rs |
| shadow.csm | shadowMap | Casters::array_view | crates/raa-shadow/src/caster.rs |
| vg.cull | cullList | CullWork::visible | crates/raa-vg/src/culling.rs |
| vg.raster | visibility | VisibilityBuffer::depth | crates/raa-vg/src/raster.rs |
| vg.raster | hzb | VgResources::current | crates/raa-vg/src/frame_passes.rs |
| vg.resolve | surfaceIds | VgSurface::ids | crates/raa-vg/src/resolve.rs |
| vg.resolve | surfaceNormal | VgSurface::normal | crates/raa-vg/src/resolve.rs |
| vg.resolve | surfaceUvMotion | VgSurface::uv_motion | crates/raa-vg/src/resolve.rs |
| vg.resolve | surfaceAlbedo | VgSurface::albedo | crates/raa-vg/src/resolve.rs |
| gi.volume | volumeAtlas | GiVolume::atlas_view | crates/raa-gi/src/volume.rs |
| prepass | depth | Targets::depth | crates/raa-pipeline/src/targets.rs |
| prepass | normal | Targets::normal | crates/raa-pipeline/src/targets.rs |
| prepass | material | Targets::material | crates/raa-pipeline/src/targets.rs |
| prepass | motion | Targets::motion | crates/raa-pipeline/src/targets.rs |
| vg.gbuffer | depth | Targets::depth | crates/raa-pipeline/src/targets.rs |
| vg.gbuffer | normal | Targets::normal | crates/raa-pipeline/src/targets.rs |
| vg.gbuffer | material | Targets::material | crates/raa-pipeline/src/targets.rs |
| vg.gbuffer | motion | Targets::motion | crates/raa-pipeline/src/targets.rs |
| light.tiles | lightTiles | ViewTiles::fragment | crates/raa-lit/src/tiles.rs |
| ao.gtao | ao | Textures::out_view | crates/raa-gtao/src/feature.rs |
| ao.gtao | raw | Textures::_raw | crates/raa-gtao/src/feature.rs |
| ao.gtao | blur | Textures::_blur | crates/raa-gtao/src/feature.rs |
| gi.screen | gi | ScreenResources::screen_view | crates/raa-gi/src/parts/screen.rs |
| gi.screen | pyramid | ScreenResources::pyramid_view | crates/raa-gi/src/parts/screen.rs |
| gi.screen | atlas | ScreenResources::atlas_view | crates/raa-gi/src/parts/screen.rs |
| gi.screen | probesRaw | ScreenResources::probes_raw | crates/raa-gi/src/parts/screen.rs |
| gi.screen | probesFiltered | ScreenResources::probes_filtered | crates/raa-gi/src/parts/screen.rs |
| gi.screen | output | ScreenResources::output_views[parity & 1] | crates/raa-gi/src/parts/screen.rs |
| gi.screen | meta | ScreenResources::meta_views[parity & 1] | crates/raa-gi/src/parts/screen.rs |
| gi.screen | filter | ScreenResources::filter_views[parity & 1] | crates/raa-gi/src/parts/screen.rs |
| gi.screen | display | ScreenResources::display_view | crates/raa-gi/src/parts/screen.rs |
| gi.screen | debug | ScreenResources::debug_view | crates/raa-gi/src/parts/screen.rs |
| gi.area-shadow | areaShadow | AreaShadowResources::mask_view | crates/raa-gi/src/parts/area_shadow/resources.rs |
| gi.area-shadow | raw | AreaShadowResources::shared_views (raw) | crates/raa-gi/src/parts/area_shadow/resources.rs |
| gi.area-shadow | depthHistory | AreaShadowResources::shared_views (depth_write) | crates/raa-gi/src/parts/area_shadow/resources.rs |
| gi.area-shadow | mask | AreaShadowResources::mask_view | crates/raa-gi/src/parts/area_shadow/resources.rs |
| gi.area-shadow | restirAtlas | RestirResources::atlas_view | crates/raa-gi/src/parts/area_shadow/restir_resources.rs |
| gi.ssgi | ssgi | SsgiResources::output_view | crates/raa-gi/src/parts/ssgi/resources.rs |
| gi.ssgi | raw | SsgiResources::raw_view | crates/raa-gi/src/parts/ssgi/resources.rs |
| gi.ssgi | history | SsgiResources::history_write_view | crates/raa-gi/src/parts/ssgi/resources.rs |
| gi.ssgi | output | SsgiResources::output | crates/raa-gi/src/parts/ssgi/resources.rs |
| shadow.contact | contact | Output::view | crates/raa-contact/src/feature.rs |
| shadow.contact | output | Output::view | crates/raa-contact/src/feature.rs |
| particles.sim | particles | SimBuffers::particles | crates/raa-particles/src/sim.rs |
| trace.sdf | trace | TraceTargets::color_view | crates/raa-sdf-trace/src/pass.rs |
| sky.cloud-screen | cloudScreen | CloudViews::screen | crates/raa-sky/src/cloud_backend.rs |
| forward | color | Targets::hdr().src() | crates/raa-pipeline/src/targets.rs |
| gi.history | color | Targets::hdr().src() | crates/raa-pipeline/src/targets.rs |
| gi.history | history | GiCore::history | crates/raa-gi/src/feature.rs |
| ssr | color | Targets::hdr().src() | crates/raa-pipeline/src/targets.rs |
| ssr | hiz | SsrTargets::_hiz | crates/raa-ssr/src/resources.rs |
| ssr | trace | SsrTargets::trace_view | crates/raa-ssr/src/resources.rs |
| fog | color | Targets::hdr().src() | crates/raa-pipeline/src/targets.rs |
| sunshafts | color | Targets::hdr().src() | crates/raa-pipeline/src/targets.rs |
| opaque.copy | color | Targets::hdr().src() | crates/raa-pipeline/src/targets.rs |
| opaque.copy | opaqueColor | Copies::color_view | crates/raa-transmission/src/copy.rs |
| opaque.copy | opaqueDepth | Copies::depth_view | crates/raa-transmission/src/copy.rs |
| transparent.oit | color | Targets::hdr().src() | crates/raa-pipeline/src/targets.rs |
| particles.blend | color | Targets::hdr().src() | crates/raa-pipeline/src/targets.rs |
| taa | color | Targets::hdr().src() | crates/raa-pipeline/src/targets.rs |
| taa | history | TaaPass::history (write view) | crates/raa-taa/src/feature.rs |
| dof | color | Targets::hdr().src() | crates/raa-pipeline/src/targets.rs |
| dof | coc | DofTargets::_coc | crates/raa-dof/src/resources.rs |
| dof | tiles | DofTargets::_tiles | crates/raa-dof/src/resources.rs |
| dof | dilated | DofTargets::_dilated | crates/raa-dof/src/resources.rs |
| dof | nearPrefilter | DofTargets::near_prefilter_view | crates/raa-dof/src/resources.rs |
| dof | farPrefilter | DofTargets::far_prefilter_view | crates/raa-dof/src/resources.rs |
| dof | nearGather | DofTargets::near_gather_view | crates/raa-dof/src/resources.rs |
| dof | farGather | DofTargets::far_gather_view | crates/raa-dof/src/resources.rs |
| bloom | color | Targets::hdr().src() | crates/raa-pipeline/src/targets.rs |
| bloom | chain | Chain::views | crates/raa-bloom/src/feature.rs |
| grading | color | Targets::hdr().src() | crates/raa-pipeline/src/targets.rs |
| cel.resolve | color | Targets::hdr().src() | crates/raa-pipeline/src/targets.rs |
| tonemap | color | Targets::output | crates/raa-pipeline/src/targets.rs |
| overlay | color | Targets::output | crates/raa-pipeline/src/targets.rs |
