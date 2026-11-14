//! The read accessors raa-gi driver A needs (P4-M1-LIT): `LitHandle::mesh_storage`,
//! `LitHandle::lod_chain_into` and `LitHandle::extras_storage`. The software-GPU test
//! (SKIP without an adapter) uploads two-sub-mesh meshes and checks the sub-mesh ranges,
//! the STORAGE usage of the vertex and index buffers, the per-recreation key, removal and
//! the `LodRegistry` chain in order.
//!
//! Nothing is rendered, so no depth or HDR input is read: the test only uploads, updates,
//! removes and registers LOD chains through the public `LitRenderer` and `LitHandle`.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::wgpu;
use raa_geom::builtin::DebugCube;
use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
use raa_lit::{AttributedMesh, LitOptions, LitRenderer, LodLevel, SubMesh};
use raa_pipeline::Pipeline;

const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A DebugCube of `half` extent cut to 12 indices split into two sub-meshes of 6 each.
fn two_submesh_mesh(half: f32) -> AttributedMesh {
    let mut base = MeshData::new();
    let params = Params::new().with("half", ParamValue::Scalar(half));
    DebugCube
        .generate(&params, Quality::LOD0, &mut base)
        .expect("debug cube");
    base.indices.truncate(12);
    let mut mesh = AttributedMesh::from_mesh(base);
    mesh.submeshes = vec![
        SubMesh {
            first_index: 0,
            index_count: 6,
            material: None,
        },
        SubMesh {
            first_index: 6,
            index_count: 6,
            material: None,
        },
    ];
    mesh
}

#[test]
fn mesh_storage_lod_chain_and_extras() {
    let test = "mesh_storage_lod_chain_and_extras";
    let Some(gpu) = software_gpu() else {
        return skip(test, "software_gpu returned None");
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
    let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
    let handle = lit.handle();

    let mesh = two_submesh_mesh(0.5);
    let id = lit.upload(&gpu, &mesh).expect("upload");
    let ids = lit.submesh_ids(id).expect("sub-meshes").to_vec();
    assert_eq!(ids.len(), 2);

    let first = handle.mesh_storage(ids[0]).expect("first sub-mesh");
    let second = handle.mesh_storage(ids[1]).expect("second sub-mesh");
    assert_eq!((first.first_index, first.index_count), (0, 6));
    assert_eq!((second.first_index, second.index_count), (6, 6));
    assert_eq!(first.vertex_count, mesh.vertex_count() as u32);
    assert_eq!(second.vertex_count, first.vertex_count);
    assert_eq!(first.key, second.key);
    let storage_vertex = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX;
    let storage_index = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX;
    assert!(first.stream0.usage().contains(storage_vertex));
    assert!(first.index.usage().contains(storage_index));
    assert!(
        handle
            .extras_storage()
            .usage()
            .contains(wgpu::BufferUsages::STORAGE)
    );

    let key = first.key;
    lit.update(&gpu, id, &mesh).expect("update");
    let updated = handle.mesh_storage(ids[0]).expect("updated sub-mesh");
    assert_ne!(updated.key, key);
    assert_eq!(
        handle.mesh_storage(ids[1]).expect("updated sibling").key,
        updated.key
    );

    lit.remove(id);
    for id in &ids {
        assert!(handle.mesh_storage(*id).is_none());
    }

    let full_id = lit
        .upload(&gpu, &two_submesh_mesh(0.5))
        .expect("full upload");
    let coarse_id = lit
        .upload(&gpu, &two_submesh_mesh(0.25))
        .expect("coarse upload");
    let full = lit.submesh_ids(full_id).expect("full sub-meshes")[0];
    let coarse = lit.submesh_ids(coarse_id).expect("coarse sub-meshes")[0];

    let mut chain = vec![LodLevel {
        mesh: full,
        error: 1.0,
    }];
    assert!(!handle.lod_chain_into(full, &mut chain));
    assert!(chain.is_empty());

    lit.set_mesh_lods(
        full,
        &[LodLevel {
            mesh: coarse,
            error: 0.05,
        }],
    )
    .expect("lod chain");
    assert!(handle.lod_chain_into(full, &mut chain));
    assert_eq!(
        chain,
        [
            LodLevel {
                mesh: full,
                error: 0.0,
            },
            LodLevel {
                mesh: coarse,
                error: 0.05,
            },
        ]
    );
}
