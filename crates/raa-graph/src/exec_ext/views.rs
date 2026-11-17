//! Executor extension `views` (S55): the sixteen `view.image` shader variants and the node input
//! checks the views share applies. The compositor passes that draw the views are later cards' work.

use raa_pipeline::{ImageDim, SampleKind};

use crate::catalog::{Catalog, GraphFormat, PortSpec, PortType};
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles};
use crate::plan::{GraphPlan, InputSource, PlanInput};

#[allow(dead_code)] // the compositor card and the tests call these; this share installs nothing
/// The sixteen (dimension, sample kind) pairs `view.image` has a shader for, in this order.
pub(crate) const IMAGE_VARIANTS: [(ImageDim, SampleKind); 16] = [
    (ImageDim::D2, SampleKind::Float),
    (ImageDim::D2, SampleKind::UnfilterableFloat),
    (ImageDim::D2, SampleKind::Depth),
    (ImageDim::D2, SampleKind::Uint),
    (ImageDim::D2, SampleKind::Sint),
    (ImageDim::D2Array, SampleKind::Float),
    (ImageDim::D2Array, SampleKind::UnfilterableFloat),
    (ImageDim::D2Array, SampleKind::Depth),
    (ImageDim::D2Array, SampleKind::Uint),
    (ImageDim::D2Array, SampleKind::Sint),
    (ImageDim::D3, SampleKind::Float),
    (ImageDim::D3, SampleKind::UnfilterableFloat),
    (ImageDim::D3, SampleKind::Uint),
    (ImageDim::D3, SampleKind::Sint),
    (ImageDim::Cube, SampleKind::Float),
    (ImageDim::Cube, SampleKind::UnfilterableFloat),
];

#[allow(dead_code)] // the compositor card and the tests call these; this share installs nothing
/// The WGSL shared by every variant; `//#SOURCE` marks where the variant's binding and fetch go.
const TEMPLATE: &str = include_str!("../wgsl/view_image.wgsl");

#[allow(dead_code)] // the compositor card and the tests call these; this share installs nothing
/// The `view_fetch` body of the cube variants: face `view.layer` in WebGPU order (+X, -X, +Y, -Y,
/// +Z, -Z), sampled at the screen `uv`.
const CUBE_FETCH: &str = "\
    let s = uv * 2.0 - 1.0;
    let face = view.layer % 6u;
    var dir = vec3<f32>(-s.x, -s.y, -1.0);
    if (face == 0u) { dir = vec3<f32>(1.0, -s.y, -s.x); }
    else if (face == 1u) { dir = vec3<f32>(-1.0, -s.y, s.x); }
    else if (face == 2u) { dir = vec3<f32>(s.x, 1.0, s.y); }
    else if (face == 3u) { dir = vec3<f32>(s.x, -1.0, -s.y); }
    else if (face == 4u) { dir = vec3<f32>(s.x, -s.y, 1.0); }
    return textureSampleLevel(view_source, view_sampler, dir, f32(view.mip));
";

#[allow(dead_code)] // the compositor card and the tests call these; this share installs nothing
/// The texture declaration and `textureLoad` call of a variant, `None` outside [`IMAGE_VARIANTS`].
fn variant(dim: ImageDim, sample: SampleKind) -> Option<(&'static str, &'static str)> {
    use ImageDim::{Cube, D2, D2Array, D3};
    use SampleKind::{Depth, Float, Sint, Uint, UnfilterableFloat};
    let ty = match (dim, sample) {
        (D2, Float | UnfilterableFloat) => "texture_2d<f32>",
        (D2, Depth) => "texture_depth_2d",
        (D2, Uint) => "texture_2d<u32>",
        (D2, Sint) => "texture_2d<i32>",
        (D2Array, Float | UnfilterableFloat) => "texture_2d_array<f32>",
        (D2Array, Depth) => "texture_depth_2d_array",
        (D2Array, Uint) => "texture_2d_array<u32>",
        (D2Array, Sint) => "texture_2d_array<i32>",
        (D3, Float | UnfilterableFloat) => "texture_3d<f32>",
        (D3, Uint) => "texture_3d<u32>",
        (D3, Sint) => "texture_3d<i32>",
        (Cube, Float | UnfilterableFloat) => "texture_cube<f32>",
        _ => return None,
    };
    let load = match dim {
        D2 => "textureLoad(view_source, p, i32(view.mip))",
        D2Array => "textureLoad(view_source, p, i32(view.layer), i32(view.mip))",
        D3 => "textureLoad(view_source, vec3<i32>(p, i32(view.slice)), i32(view.mip))",
        Cube => "",
    };
    Some((ty, load))
}

/// The lane `channels` reads: `r`/`rgb`/`rgba-alpha` read x, `g` y, `b` z and `a` w.
fn integer_body(load: &str, signed: bool) -> String {
    let lane = |component: char| {
        if signed {
            format!("bitcast<u32>(texel.{component})")
        } else {
            format!("texel.{component}")
        }
    };
    format!(
        "    let texel = {load};\n    var lane = {};\n    if (view.channels == 2u) {{ lane = {}; }}\n    else if (view.channels == 3u) {{ lane = {}; }}\n    else if (view.channels == 4u) {{ lane = {}; }}\n    var v = f32(lane);\n    if (view.decode == 1u) {{ v = unpack2x16float(lane).x; }}\n    else if (view.decode == 2u) {{ v = unpack2x16float(lane).y; }}\n    return vec4<f32>(v, v, v, 1.0);\n",
        lane('x'),
        lane('y'),
        lane('z'),
        lane('w'),
    )
}

#[allow(dead_code)] // the compositor card and the tests call these; this share installs nothing
/// Handoff: `view_image_source` is called by card 'View nodes 4/6: ViewPass compositor FeaturePass'
/// to build the module of one variant. It returns the text of `wgsl/view_image.wgsl` with
/// `//#SOURCE` replaced by the `(dim, sample)` binding and `view_fetch`, `None` outside
/// [`IMAGE_VARIANTS`]; no state is passed.
pub(crate) fn view_image_source(dim: ImageDim, sample: SampleKind) -> Option<String> {
    let (ty, load) = variant(dim, sample)?;
    let fetch = if dim == ImageDim::Cube {
        CUBE_FETCH.to_string()
    } else {
        let head = "    let dims = vec2<i32>(textureDimensions(view_source, view.mip).xy);\n    let p = clamp(px, vec2<i32>(0), dims - 1);\n";
        let tail = match sample {
            SampleKind::Float | SampleKind::UnfilterableFloat => format!("    return {load};\n"),
            SampleKind::Depth => {
                format!("    let d = {load};\n    return vec4<f32>(d, d, d, 1.0);\n")
            }
            SampleKind::Uint => integer_body(load, false),
            SampleKind::Sint => integer_body(load, true),
        };
        format!("{head}{tail}")
    };
    let source = format!(
        "@group(0) @binding(1) var view_source: {ty};\n\nfn view_fetch(px: vec2<i32>, uv: vec2<f32>) -> vec4<f32> {{\n{fetch}}}\n"
    );
    Some(TEMPLATE.replace("//#SOURCE", &source))
}

/// Handoff: `ViewsExtension` is extended by card 'View nodes 5/6: views graph frame, pass
/// installation and node stats', which extends `ViewsExtension::apply` and `ViewsExtension::clear`
/// and calls `check_plan`; the generated registry attaches the extension through `extension()`.
/// State: the extension's own `Catalog`, built once in `ViewsExtension::new()`.
pub(crate) struct ViewsExtension {
    catalog: Catalog,
}

impl ViewsExtension {
    /// The extension with its own catalog.
    pub(crate) fn new() -> ViewsExtension {
        ViewsExtension {
            catalog: Catalog::build(),
        }
    }
}

/// The `views` extension, discovered by build.rs.
pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(ViewsExtension::new())
}

impl ExecExtension for ViewsExtension {
    fn name(&self) -> &'static str {
        "views"
    }

    fn apply(
        &mut self,
        plan: &GraphPlan,
        _previous: Option<&GraphPlan>,
        _h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let issues = check_plan(plan, &self.catalog);
        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    fn clear(&mut self, _h: &mut FeatureHandles<'_>) {}
}

/// One apply-stage error of `node` on `port`.
fn issue(node: &str, port: &str, code: &'static str, message: String, fix: &str) -> Issue {
    Issue {
        code,
        severity: Severity::Error,
        stage: Stage::Apply,
        node: Some(node.to_string()),
        port: Some(port.to_string()),
        param: None,
        link: None,
        message,
        fix: fix.to_string(),
    }
}

/// The catalog output spec that feeds a planned input, `None` when it is not a link.
fn output_spec<'a>(
    plan: &'a GraphPlan,
    catalog: &'a Catalog,
    node: &str,
    port: &str,
) -> Option<&'a PortSpec> {
    let producer = plan.nodes.iter().find(|candidate| candidate.id == node)?;
    catalog
        .node(&producer.type_id)?
        .outputs
        .iter()
        .find(|spec| spec.key == port)
}

/// The input checks of the view nodes (contract 'view-shader-image'): `view.image` links exactly
/// one of its six inputs, `view.grid` at least one tile, and `view.compare`/`view.grid` read
/// float-sampleable image2d sources only. Every issue is an apply-stage error.
pub(crate) fn check_plan(plan: &GraphPlan, catalog: &Catalog) -> Vec<Issue> {
    let mut issues = Vec::new();
    for node in &plan.nodes {
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
        let linked: Vec<&PlanInput> = node
            .inputs
            .iter()
            .filter(|input| matches!(input.source, InputSource::Link { .. }))
            .collect();
        if spec.id == "view.image" {
            if linked.is_empty() {
                issues.push(issue(
                    &node.id,
                    "src",
                    "graph.input_missing",
                    format!(
                        "node `{}` (view.image) links none of its image inputs src, srcDepth, srcArray, srcDepthArray, src3d and srcCube, and exactly one must be linked",
                        node.id
                    ),
                    "Link one image into the node's `src` input.",
                ));
            } else if linked.len() > 1 {
                let (first, second) = (linked[0].port.as_str(), linked[1].port.as_str());
                issues.push(issue(
                    &node.id,
                    second,
                    "graph.input_taken",
                    format!(
                        "node `{}` (view.image) links two image inputs, `{first}` and `{second}`, and exactly one is allowed",
                        node.id
                    ),
                    &format!("Unlink `{second}` so that only `{first}` stays linked."),
                ));
            }
        }
        if spec.id == "view.grid" && linked.is_empty() {
            issues.push(issue(
                &node.id,
                "t0",
                "graph.input_missing",
                format!(
                    "node `{}` (view.grid) links none of its tile inputs t0..t15, so it has nothing to show",
                    node.id
                ),
                "Link an image into the node's `t0` input.",
            ));
        }
        if spec.id != "view.compare" && spec.id != "view.grid" {
            continue;
        }
        for input in linked {
            let InputSource::Link {
                node: source_node,
                port: source_port,
                ..
            } = &input.source
            else {
                continue;
            };
            let Some(source) = output_spec(plan, catalog, source_node, source_port) else {
                continue;
            };
            let refused = source.ty != PortType::Image2d
                || matches!(
                    source.format,
                    Some(
                        GraphFormat::R32Uint
                            | GraphFormat::Rg32Uint
                            | GraphFormat::Rgba32Uint
                            | GraphFormat::Depth32Float
                    )
                );
            if refused {
                issues.push(issue(
                    &node.id,
                    &input.port,
                    "graph.link_type",
                    format!(
                        "node `{}` input `{}` reads `{source_node}.{source_port}`, a {:?} with format {}, and view.compare and view.grid take float-sampleable image2d sources only",
                        node.id,
                        input.port,
                        source.ty,
                        source.format.as_ref().map_or("none", GraphFormat::name)
                    ),
                    &format!(
                        "Insert a view.image node between `{source_node}.{source_port}` and `{}.{}`.",
                        node.id, input.port
                    ),
                ));
            }
        }
    }
    issues
}

/// Returns the `view.compare` module (crates/raa-graph/src/wgsl/view_compare.wgsl) unchanged.
/// Handoff: `view_compare_source`, `view_grid_source` and `pack_label` are called by card
/// 'View nodes 4/6: ViewPass compositor FeaturePass'.
#[allow(dead_code)] // called by the compositor and the tests
pub(crate) fn view_compare_source() -> String {
    include_str!("../wgsl/view_compare.wgsl").to_string()
}

/// Returns the `view.grid` module (crates/raa-graph/src/wgsl/view_grid.wgsl) unchanged.
#[allow(dead_code)] // called by the compositor and the tests
pub(crate) fn view_grid_source() -> String {
    include_str!("../wgsl/view_grid.wgsl").to_string()
}

/// Packs the first 32 bytes of `text` into `out`, 4 bytes per word little-endian, zero after the
/// end; a char outside ASCII 32..=126 becomes one `?`.
#[allow(dead_code)] // called by the compositor and the tests
pub(crate) fn pack_label(text: &str, out: &mut [u32; 8]) {
    *out = [0; 8];
    for (i, c) in text.chars().take(32).enumerate() {
        let byte = if (' '..='~').contains(&c) {
            c as u32
        } else {
            u32::from(b'?')
        };
        out[i / 4] |= byte << ((i % 4) * 8);
    }
}

#[cfg(test)]
mod tests {
    use raa_loader::json::{JsonValue, parse, write_pretty};
    use raa_shader::Composer;

    use super::*;

    #[test]
    fn every_image_variant_validates() {
        for (dim, sample) in IMAGE_VARIANTS {
            let text = view_image_source(dim, sample)
                .unwrap_or_else(|| panic!("{dim:?} {sample:?} has a variant"));
            Composer::validate(&text).unwrap_or_else(|error| panic!("{dim:?} {sample:?}: {error}"));
        }
        assert_eq!(IMAGE_VARIANTS.len(), 16);
        assert!(view_image_source(ImageDim::D3, SampleKind::Depth).is_none());
        assert!(view_image_source(ImageDim::Cube, SampleKind::Uint).is_none());
    }

    /// The Simple forward starter as JSON, from the generated registry table.
    fn starter() -> JsonValue {
        let (_, _, text) = crate::registry::STARTERS
            .iter()
            .find(|(_, stem, _)| *stem == "simple-forward")
            .expect("the simple-forward starter is registered");
        parse(text).expect("the Simple forward starter parses")
    }

    /// The `key` field of a JSON object.
    fn text_field<'a>(value: &'a JsonValue, key: &str) -> &'a str {
        let JsonValue::Object(fields) = value else {
            panic!("a JSON object is expected")
        };
        match fields
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
        {
            Some(JsonValue::String(text)) => text,
            _ => panic!("the object names `{key}` as a string"),
        }
    }

    /// The array stored under `key` of the starter document.
    fn array<'a>(doc: &'a mut JsonValue, key: &str) -> &'a mut Vec<JsonValue> {
        let JsonValue::Object(fields) = doc else {
            panic!("the starter is a JSON object")
        };
        match fields.iter_mut().find(|(name, _)| name == key) {
            Some((_, JsonValue::Array(items))) => items,
            _ => panic!("the starter holds a `{key}` array"),
        }
    }

    fn link(from: &str, to: &str) -> JsonValue {
        JsonValue::Object(vec![
            ("from".to_string(), JsonValue::String(from.to_string())),
            ("to".to_string(), JsonValue::String(to.to_string())),
        ])
    }

    /// A Simple forward plan with a `view` node of `view_type` displayed in place of `tonemap`,
    /// reading `inputs`.
    fn plan_with(view_type: &str, inputs: &[(&str, &str)]) -> GraphPlan {
        let mut doc = starter();
        let links = array(&mut doc, "links");
        let display = links
            .iter()
            .position(|link| text_field(link, "to") == "out.color")
            .expect("the starter links into out.color");
        let JsonValue::Object(fields) = &mut links[display] else {
            panic!("a link is a JSON object")
        };
        let from = fields
            .iter_mut()
            .find(|(name, _)| name == "from")
            .expect("a link names its source");
        from.1 = JsonValue::String("view1.color".to_string());
        for (source, port) in inputs {
            links.push(link(source, port));
        }
        array(&mut doc, "nodes").push(JsonValue::Object(vec![
            ("id".to_string(), JsonValue::String("view1".to_string())),
            ("type".to_string(), JsonValue::String(view_type.to_string())),
        ]));
        let mut text = String::new();
        write_pretty(&doc, &mut text);
        crate::plan::plan_from_json(&text).expect("the test document plans")
    }

    #[test]
    fn one_linked_image_input_is_accepted() {
        let plan = plan_with("view.image", &[("tonemap.color", "view1.src")]);
        assert_eq!(check_plan(&plan, &Catalog::build()), Vec::new());
    }

    #[test]
    fn a_second_linked_image_input_is_taken() {
        let plan = plan_with(
            "view.image",
            &[
                ("tonemap.color", "view1.src"),
                ("prepass.depth", "view1.srcDepth"),
            ],
        );
        let issues = check_plan(&plan, &Catalog::build());
        assert_eq!(issues.len(), 1, "{issues:?}");
        let issue = &issues[0];
        assert_eq!(issue.code, "graph.input_taken");
        assert_eq!(issue.node.as_deref(), Some("view1"));
        assert_eq!(issue.port.as_deref(), Some("srcDepth"));
        assert_eq!(issue.stage, Stage::Apply);
        assert_eq!(issue.severity, Severity::Error);
    }

    #[test]
    fn missing_view_inputs_are_named() {
        let grid = plan_with("view.grid", &[]);
        let issues = check_plan(&grid, &Catalog::build());
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].code, "graph.input_missing");
        assert_eq!(issues[0].port.as_deref(), Some("t0"));

        let image = plan_with("view.image", &[]);
        let issues = check_plan(&image, &Catalog::build());
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].code, "graph.input_missing");
        assert_eq!(issues[0].port.as_deref(), Some("src"));
    }

    #[test]
    fn compare_and_grid_sources_validate() {
        Composer::validate(&view_compare_source()).unwrap_or_else(|e| panic!("compare: {e}"));
        Composer::validate(&view_grid_source()).unwrap_or_else(|e| panic!("grid: {e}"));
    }

    #[test]
    fn labels_pack_little_endian_and_truncate() {
        let mut out = [7u32; 8];
        pack_label("ao", &mut out);
        assert_eq!(out, [0x6f61, 0, 0, 0, 0, 0, 0, 0]);
        let long: String = (0..40u8).map(|i| char::from(b'A' + i % 26)).collect();
        pack_label(&long, &mut out);
        assert_eq!(out[7] >> 24, u32::from(long.as_bytes()[31]));
        assert_eq!(out[0], u32::from_le_bytes([b'A', b'B', b'C', b'D']));
        pack_label("\u{e9}", &mut out);
        assert_eq!(out, [0x3f, 0, 0, 0, 0, 0, 0, 0]);
    }

    fn font_glyphs() -> Vec<u64> {
        let source = view_grid_source();
        let start = source.find("const VIEW_FONT").expect("VIEW_FONT");
        let mut glyphs = Vec::new();
        for line in source[start..].lines().skip(1) {
            let Some(rest) = line.trim().strip_prefix("vec2<u32>(") else {
                break;
            };
            let inner = &rest[..rest.find(')').expect("closing parenthesis")];
            let lanes: Vec<u64> = inner
                .split(',')
                .map(|n| {
                    let hex = n.trim().trim_start_matches("0x").trim_end_matches('u');
                    u64::from_str_radix(hex, 16).expect("hex lane")
                })
                .collect();
            glyphs.push(lanes[0] | (lanes[1] << 32));
        }
        glyphs
    }

    #[test]
    fn font_has_blank_space_and_visible_digits_and_capitals() {
        let glyphs = font_glyphs();
        assert_eq!(glyphs.len(), 95);
        assert_eq!(glyphs[0], 0);
        for c in ('0'..='9').chain('A'..='Z') {
            assert_ne!(glyphs[c as usize - 32], 0, "glyph {c}");
        }
    }
}
