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
  /** Installs a complete little-endian TraceProgram v1 or v2 byte stream (v2 adds sampled SDF leaves).
   * Samples live in the packed data buffer; normals are evaluated from the SDF gradient.
   * A bent instance requires its matching packed bend record. A rejected program leaves the scene intact. */
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


/** Geometry reflections use the screen result first and trace only rejected pixels.
 * The full-frame path tracer is an explicit reference mode. */
export interface MeshTracingSettings {
  reflections?: boolean;
  reference?: boolean;
  confidenceThreshold?: number;
  reflectionBounces?: number;
  pathBounces?: number;
}

export interface MeshTracingStats extends Required<Pick<MeshTracingSettings,
  'confidenceThreshold' | 'reflectionBounces' | 'pathBounces'>> {
  frame: number;
  width: number;
  height: number;
  eligible: number;
  acceptedScreen: number;
  traced: number;
  screenMiss: number;
  offscreen: number;
  disocclusion: number;
  lowConfidence: number;
  nestedMirror: number;
  geometryHit: number;
  mirrorRays: number;
  pathRays: number;
  referencePixels: number;
}

export interface Engine {
  /** Merges settings. Threshold 0–1, mirror bounces 1–6, reference path bounces 0–6. */
  setMeshTracing(options: MeshTracingSettings): void;
  /** Counts actual half-resolution GPU samples from the latest completed tracing frame. */
  meshTracingStats(): MeshTracingStats | null;
}
