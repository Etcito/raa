// S18 additions; see docs/api/m18-skinning.md. build-package.sh appends this file
// to index.d.ts. These interfaces merge with the facade and m07 mesh declarations.

/** Optional skin attributes for addMesh/updateMesh; n is the rest vertex count. */
export interface MeshDesc {
  /**
   * 4n joint indices in S16 bone order, each < 256. Supply together with weights.
   * All slots count toward the pose joint count, including slots with zero weight.
   */
  joints?: Uint16Array;
  /**
   * 4n finite, non-negative weights, four per vertex, summing to 1 within 1e-4.
   * Supply together with joints; omitted pairs describe a rigid mesh.
   */
  weights?: Float32Array;
}

export interface Engine {
  /**
   * Replaces one instance's pose; instances sharing a mesh keep independent poses.
   * Matrices are mesh-space model(pose) * inverse_bind, column-major, in joint order.
   * Exactly 16 * J finite floats, J = 1 + max(mesh.joints), including zero-weight slots.
   * Without a pose, or with the skin pass disabled, the instance renders at rest.
   * Validation failure leaves the pose unchanged. Use identity matrices for rest.
   * @throws RaaError `api.unknown_handle`, `skin.not_skinned`, `skin.pose_length`,
   *   `skin.joint_range`, `skin.cap_joints`, `api.feature_missing`, `pipeline.busy`.
   */
  setSkinPose(instance: number, matrices: Float32Array): void;

  /**
   * Concatenates one pose slice per instances entry in handle order, without padding.
   * Slice i has 16 * J_i floats; J_i = 1 + max(that instance's mesh.joints).
   * Mixed joint counts are supported. Total length is 16 * sum(J_i).
   * Every handle and slice is validated before any pose changes. Repeated handles
   * use the last slice. Each matrix has the space/order of setSkinPose.
   * @throws RaaError `api.unknown_handle`, `skin.not_skinned`, `skin.pose_length`,
   *   `skin.joint_range`, `skin.cap_joints`, `api.feature_missing`, `pipeline.busy`.
   */
  setSkinPoses(instances: Uint32Array, matrices: Float32Array): void;
}
