//! CPU coverage of the shared draw path used by prepass, forward, shadow and OIT.
use raa_core::{MaterialId, MeshId};
use raa_lit::{DrawClass, DrawPrep, MaterialClass, PrepInput, SurfaceMaterial};
use raa_math::Mat4;
use raa_scene::{RenderList, Scene, Transform};
use raa_shader::{Composer, PassKind};
use std::collections::HashMap;

#[test]
fn pass_selection_preserves_shared_tangent_handedness() {
    for pass in [
        PassKind::Prepass,
        PassKind::Forward,
        PassKind::Shadow,
        PassKind::Transparent,
    ] {
        for posed in [false, true] {
            let entry = raa_lit::skin_draw::entry_points(pass, posed);
            assert_eq!(entry.contains("@location(13)"), posed);
            if pass != PassKind::Shadow {
                assert!(entry.contains("v1.tangent.w"));
                assert_eq!(entry.contains("vec4<f32>(raa_posed_tangent, 0.0)"), posed);
                assert!(entry.contains("out.uv = v1.uv"));
                assert!(entry.contains("out.color = v1.color"));
            }
            let source = format!("{}\n{}\n{}\n{}\n{}", raa_shader::PRELUDE_WGSL,
                raa_shader::LIB_WGSL, include_str!("../src/wgsl/slots_default.wgsl"),
                "const RAA_MSAA: u32 = 1u; const RAA_VARIANT_FLAGS: u32 = 0u;
                fn material_main(in: MaterialInput) -> MaterialOutput { return raa_default_output(in); }
                fn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput {
                    return VertexMaterialOutput(in.local_position, in.normal);
                }", entry);
            Composer::validate(&source)
                .unwrap_or_else(|e| panic!("{pass:?}, posed={posed}: {e:?}"));
        }
    }
}

#[test]
fn submeshes_keep_material_and_instance_identity_without_merging_poses() {
    for class in [DrawClass::Opaque, DrawClass::Fur, DrawClass::Transparent] {
        let mut scene = Scene::with_capacity(8, 8);
        for mesh in 0..2 {
            for tag in [17, 29, 31, 32] {
                let node = scene.add_node(None, Transform::IDENTITY).unwrap();
                let instance = scene
                    .add_instance(node, MeshId(mesh), MaterialId(mesh))
                    .unwrap();
                scene.set_instance_tag(instance, tag).unwrap();
            }
        }
        scene.update_world();
        let mut list = RenderList::with_capacity(8);
        let mut tags = Vec::with_capacity(8);
        scene.build_render_list_tagged(&mut list, &mut tags);
        let materials = [MaterialClass {
            class,
            ..MaterialClass::of(&SurfaceMaterial::default())
        }; 2];
        let extras = HashMap::new();
        let input = PrepInput {
            list: &list,
            tags: &tags,
            materials: &materials,
            extras: &extras,
            view: Mat4::IDENTITY,
            known_mesh: &|m| m.0 < 2,
            class_handled: &|_| true,
        };
        let mut prep = DrawPrep::new(8, 2);
        prep.prepare_skinned(&input, &|tag| tag == 17 || tag == 29)
            .unwrap();
        let mut posed = Vec::new();
        for draw in prep.draws(class) {
            let (mesh, tag) = prep.mesh_binding(draw.mesh);
            assert_eq!(mesh.0, draw.material.0);
            if let Some(tag) = tag {
                assert_eq!(draw.instance_count, 1);
                posed.push((mesh.0, tag));
                let offset = draw.first_instance as usize * 128;
                let record = &prep.instance_bytes()[offset..offset + 128];
                assert_eq!(
                    u32::from_le_bytes(record[116..120].try_into().unwrap()),
                    tag
                );
                assert_eq!(
                    u32::from_le_bytes(record[112..116].try_into().unwrap()),
                    mesh.0
                );
            } else if class != DrawClass::Transparent {
                assert_eq!(draw.instance_count, 2, "rigid instances still batch");
            }
        }
        posed.sort_unstable();
        assert_eq!(posed, [(0, 17), (0, 29), (1, 17), (1, 29)]);
        prep.prepare(&input).unwrap();
        assert!(
            prep.draws(class)
                .iter()
                .all(|d| prep.mesh_binding(d.mesh).1.is_none())
        );
    }
}
