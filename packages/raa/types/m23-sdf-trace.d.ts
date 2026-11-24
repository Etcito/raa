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
  /** Single-draw and independent 128-draw controls are intended for comparisons. */
  panelSampling?: 'stable' | 'single' | 'independent';
  diffuseOutput?: 'off' | 'direct' | 'direct-and-indirect';
  /** Selects the lighting contribution for the forward or full-frame mesh output. Default combined. */
  lightingOutput?: 'combined' | 'direct' | 'indirect' | 'bounce';
  /** With lightingOutput 'bounce' and reference enabled, selects an exact depth from 0 to 12. */
  lightingDepth?: number;
  pathMode?: 'off' | 'short' | 'traced' | 'learned';
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
  modelPixels: number;
  refreshPixels: number;
  diffuseMissingPixels: number;
  diffuseRays: number;
  specularRays: number;
  pathForegroundPixels: number;
  modelRejectedPixels: number;
  primaryBvhPixels: number;
  referenceSpp: number;
  hybridLobeSpp: number;
}

export interface Engine {
  /** Merges settings. Threshold 0–1, mirror bounces 1–6, reference path bounces 0–6. */
  setMeshTracing(options: MeshTracingSettings): void;
  /** Latest completed GPU counters for a mesh graph node, or for legacy tracing when omitted. */
  meshTracingStats(nodeId?: string): MeshTracingStats | null;
  /** Loads a specular model in validated 12→8→3 little-endian FP32 tensor order. */
  setMeshPathModel(weights: Uint8Array): void;
  /** Reads the renderer's tight, little-endian linear RGBA16F scene color. */
  captureMeshLighting(nowMs: number, traced: boolean, raw?: boolean): Promise<{
    size: [number, number];
    data: Uint8Array;
    /** The same-frame combined radiance while capturing an isolated forward view. */
    combinedData?: Uint8Array;
  }>;
  /** Captures one fixed-seed specular RGB sample and guides from short or traced mode. */
  captureMeshPathSample(nowMs: number, seed: number): Promise<{ size: [number, number]; data: Uint8Array }>;
  /** Offline readback of the latest submitted single view, without advancing its history. */
  captureTrainingTaps(frameId: number): Promise<{
    hdr: Uint8Array;
    depth: Uint8Array;
    normal: Uint8Array;
    motion: Uint8Array;
    reflectionHit?: Uint8Array;
    jitterNdc: [number, number];
  }>;
  /** Offline seeded reference trace, with an independent batch of 1 to 32 samples per pixel. */
  captureTrainingReference(nowMs: number, seed: number, samples: number): Promise<{
    size: [number, number];
    data: Uint8Array;
  }>;
}
