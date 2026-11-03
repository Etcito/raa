//! CPU coverage of the shared draw path used by prepass, forward, shadow, OIT and fur:
//! the posed (instanced skinning) entry points and the posed batching of the draw prep.
use raa_core::{MaterialId, MeshId};
use raa_lit::{
    DrawClass, DrawPrep, INSTANCE_PALETTE_OFFSET, MaterialClass, PrepInput, SkinDraw,
    SurfaceMaterial,
};
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
        PassKind::FurShell,
    ] {
        for posed in [false, true] {
            let entry = raa_lit::skin_draw::entry_points(pass, posed);
            for location in ["@location(13)", "@location(14)", "@location(15)"] {
                assert_eq!(entry.contains(location), posed, "{pass:?} {location}");
            }
            assert_eq!(entry.contains("raa_skin_palette"), posed);
            assert_eq!(entry.contains("@group(3)"), posed);
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

fn u32_at(record: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(record[at..at + 4].try_into().unwrap())
}

#[test]
fn posed_instances_of_one_mesh_and_material_share_one_instanced_draw() {
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
        // Tags 17 and 29 are posed (palette blocks 0 and 4, one skin per mesh id).
        let skinned = |tag: u32| match tag {
            17 => Some(SkinDraw { base: 0, skin: 7 }),
            29 => Some(SkinDraw { base: 4, skin: 7 }),
            _ => None,
        };
        prep.prepare_skinned(&input, &skinned).unwrap();
        let mut posed = Vec::new();
        for draw in prep.draws(class) {
            let (mesh, tag) = prep.mesh_binding(draw.mesh);
            assert_eq!(mesh.0, draw.material.0);
            let records: Vec<&[u8]> = (draw.first_instance
                ..draw.first_instance + draw.instance_count)
                .map(|i| &prep.instance_bytes()[i as usize * 128..i as usize * 128 + 128])
                .collect();
            for record in &records {
                assert_eq!(u32_at(record, 112), mesh.0, "material index");
            }
            if tag.is_some() {
                for record in &records {
                    let t = u32_at(record, 116);
                    let base = u32_at(record, INSTANCE_PALETTE_OFFSET);
                    posed.push((mesh.0, t, base));
                }
                if class != DrawClass::Transparent {
                    assert_eq!(draw.instance_count, 2, "posed instances batch");
                }
            } else {
                for record in &records {
                    assert_eq!(u32_at(record, INSTANCE_PALETTE_OFFSET), 0);
                }
                if class != DrawClass::Transparent {
                    assert_eq!(draw.instance_count, 2, "rigid instances still batch");
                }
            }
        }
        posed.sort_unstable();
        assert_eq!(
            posed,
            [(0, 17, 0), (0, 29, 4), (1, 17, 0), (1, 29, 4)],
            "{class:?}"
        );
        let expected_draws = if class == DrawClass::Transparent {
            8
        } else {
            4
        };
        assert_eq!(prep.draws(class).len(), expected_draws, "{class:?}");
        prep.prepare(&input).unwrap();
        assert!(
            prep.draws(class)
                .iter()
                .all(|d| prep.mesh_binding(d.mesh).1.is_none())
        );
    }
}

#[test]
fn draw_count_does_not_grow_with_posed_instances() {
    for n in [1u32, 2, 60, 200] {
        let mut scene = Scene::with_capacity(n as usize, n as usize);
        for tag in 0..n {
            let node = scene.add_node(None, Transform::IDENTITY).unwrap();
            let instance = scene.add_instance(node, MeshId(0), MaterialId(0)).unwrap();
            scene.set_instance_tag(instance, tag).unwrap();
        }
        scene.update_world();
        let mut list = RenderList::with_capacity(n as usize);
        let mut tags = Vec::new();
        scene.build_render_list_tagged(&mut list, &mut tags);
        let materials = [MaterialClass::of(&SurfaceMaterial::default())];
        let extras = HashMap::new();
        let input = PrepInput {
            list: &list,
            tags: &tags,
            materials: &materials,
            extras: &extras,
            view: Mat4::IDENTITY,
            known_mesh: &|m| m.0 == 0,
            class_handled: &|_| true,
        };
        let mut prep = DrawPrep::new(n, 1);
        prep.prepare_skinned(&input, &|tag| {
            Some(SkinDraw {
                base: tag * 4,
                skin: 1,
            })
        })
        .unwrap();
        assert_eq!(prep.draws(DrawClass::Opaque).len(), 1, "n = {n}");
        assert_eq!(prep.draws(DrawClass::Opaque)[0].instance_count, n);
    }
}
