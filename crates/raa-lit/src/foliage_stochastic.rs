//! Stochastic foliage assembly (M7, contract `m7-stochastic`).
//!
//! [`build_stochastic_foliage`] assembles the opaque foliage mesh and the native volume
//! of one tree from the sampler's strata. The non-foliage source triangles stay opaque
//! wood in source order; every [`FoliageSample`] then emits one independent proxy
//! triangle at [`FoliageSample::corners`] whose three vertices carry the sampled source
//! shading normal, so the normal never follows the proxy winding. The same traversal
//! accumulates the unlit albedo and represented area into the
//! [`FoliageVolumeAccumulator`]. No resampling and no second area compensation happen:
//! the deterministic area compensation is the one `sample_foliage` already did.

use raa_core::RaaError;
use raa_geom::MeshData;

use super::foliage_resources::FoliageSource;
use super::foliage_sampling::FoliageSample;
use super::foliage_volume::FoliageVolumeAccumulator;

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: message.into(),
    }
}

fn push_triangle(
    mesh: &mut MeshData,
    positions: [[f32; 3]; 3],
    normals: [[f32; 3]; 3],
) -> Result<(), RaaError> {
    let base = u32::try_from(mesh.positions.len())
        .map_err(|_| invalid("foliage mesh exceeds the 32-bit vertex index range"))?;
    mesh.positions.extend_from_slice(&positions);
    mesh.normals.extend_from_slice(&normals);
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
    Ok(())
}

/// Reserves, at their exact size, the three corners and indices of `triangles` emitted
/// triangles in `mesh`. A size that overflows or an allocation that is refused returns
/// `foliage.invalid` naming the byte count instead of aborting on a later push.
pub(crate) fn reserve_triangles(mesh: &mut MeshData, triangles: usize) -> Result<(), RaaError> {
    let corners = triangles
        .checked_mul(3)
        .ok_or_else(|| invalid("foliage mesh corner count overflows"))?;
    let bytes = corners.saturating_mul(12 + 12 + 4);
    let refused = |_| invalid(format!("cannot reserve {bytes} bytes of foliage mesh"));
    mesh.positions.try_reserve_exact(corners).map_err(refused)?;
    mesh.normals.try_reserve_exact(corners).map_err(refused)?;
    mesh.indices.try_reserve_exact(corners).map_err(refused)?;
    Ok(())
}

/// The assembled stochastic foliage mesh and its native volume payload.
///
/// `mesh` has three independent vertices per emitted triangle, triangle-constant source
/// shading normals on every foliage triangle and no vertex colours. `volume` is the
/// x-fastest RGBA8 payload of [`FoliageVolumeAccumulator::finish`].
/// `foliage_triangles` and `wood_triangles` count the triangles actually emitted, never a
/// requested budget.
#[derive(Clone, Debug, PartialEq)]
pub struct FoliageMeshBake {
    pub mesh: MeshData,
    pub volume: Vec<u8>,
    pub foliage_triangles: u32,
    pub wood_triangles: u32,
}

/// Assembles the stochastic foliage of `source` from its `samples` and unlit `albedo`.
///
/// `samples` must be the `sample_foliage` strata of the same source in stratum order and
/// `albedo` holds one linear RGB value per sample. The wood triangles of `source` come
/// first with their source positions and normals, then each sample emits its proxy
/// triangle, so the emitted foliage area equals the eligible source area. `resolution` is
/// the 16 or 32 voxels-per-axis resolution of [`FoliageVolumeAccumulator`].
///
/// An albedo length that differs from the sample count is refused with `foliage.invalid`
/// before any output exists. Non-finite or out-of-range albedo and a degenerate source box
/// are refused the same way by the accumulator. Empty foliage with retained wood emits the
/// wood and a zeroed volume; repeated calls with equal inputs return equal meshes and
/// equal volume bytes.
///
/// Handoff: 'Publish completed foliage assets in the renderer' (card #9135) calls
/// `build_stochastic_foliage`, which today assembles the opaque stochastic mesh and the
/// native volume bytes from the sampler's strata in one traversal. The earlier handoff for
/// the same downstream card is `crates/raa-lit/src/foliage_hull.rs:248` `build_foliage_hull`
/// and, for the source snapshot it consumes,
/// `crates/raa-lit/src/handle.rs:1641` `LitHandle::foliage_source`.
pub fn build_stochastic_foliage(
    source: &FoliageSource,
    samples: &[FoliageSample],
    albedo: &[[f32; 3]],
    resolution: u32,
) -> Result<FoliageMeshBake, RaaError> {
    if samples.len() != albedo.len() {
        return Err(invalid(format!(
            "foliage albedo length {} does not match sample count {}",
            albedo.len(),
            samples.len()
        )));
    }
    let mut volume = FoliageVolumeAccumulator::new(source.min, source.max, resolution)?;
    let mut mesh = MeshData::new();
    let wood_count = source.triangles.iter().filter(|t| !t.foliage).count();
    reserve_triangles(&mut mesh, wood_count.saturating_add(samples.len()))?;
    let mut wood = 0usize;
    for triangle in &source.triangles {
        if triangle.foliage {
            continue;
        }
        push_triangle(&mut mesh, triangle.positions, triangle.normals)?;
        wood += 1;
    }
    let wood_triangles = u32::try_from(wood)
        .map_err(|_| invalid("foliage wood triangle count exceeds the reported count"))?;
    let foliage_triangles = u32::try_from(samples.len())
        .map_err(|_| invalid("foliage sample count exceeds the reported triangle count"))?;
    for (sample, albedo) in samples.iter().zip(albedo) {
        push_triangle(&mut mesh, sample.corners, [sample.normal; 3])?;
        volume.add(sample.position, *albedo, sample.represented_area)?;
    }
    Ok(FoliageMeshBake {
        mesh,
        volume: volume.finish(),
        foliage_triangles,
        wood_triangles,
    })
}
