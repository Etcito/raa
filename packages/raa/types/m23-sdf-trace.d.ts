// M23 packed SDF tracing. build-package.sh merges this fragment into index.d.ts.
export interface TraceStats {
  sources: number;
  instances: number;
  bvhNodes: number;
  width: number;
  height: number;
  placeholders: number;
}

export interface Engine {
  /** Installs a complete little-endian TraceProgram v1 byte stream. A bent instance
   * requires its matching packed bend record. A rejected program leaves the scene intact. */
  setTraceScene(program: Uint8Array): void;
  /** Replaces the 32-word records, preserving their bit patterns. The count and existing
   * bend-record associations must remain valid; a rejected update leaves the scene intact. */
  updateTraceInstances(instances: Float32Array): void;
  /** Removes the trace scene. */
  clearTraceScene(): void;
  /** Returns scene counts and full trace target dimensions. The world volume has a
   * separate longest-axis limit of 1024 voxels. */
  traceStats(): TraceStats;
}
